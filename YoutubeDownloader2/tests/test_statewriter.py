"""Tests for ytdl_core.statewriter: coalescing, durability and shutdown."""

from __future__ import annotations

import json
import threading
import time

from ytdl_core.statewriter import CoalescingStateWriter


def _writer(tmp_path, **kwargs):
    kwargs.setdefault("flush_interval", 0.05)
    kwargs.setdefault("flush_batch", 25)
    saves: list[dict] = []

    def save(state, output_dir, filename=None):
        saves.append(json.loads(json.dumps(state)))

    state = {"downloads": {}}
    writer = CoalescingStateWriter(
        state,
        tmp_path,
        ".state.json",
        save_fn=save,
        **kwargs,
    )
    return state, writer, saves


def _record(state, writer, count, prefix="a"):
    for index in range(count):
        key = f"{prefix}::{index}"
        with writer.lock:
            state["downloads"][key] = {"status": "downloaded"}
        writer.record()


class TestCoalescing:
    def test_a_burst_becomes_few_writes(self, tmp_path):
        """The point of the writer: N updates must not become N fsyncs."""
        state, writer, saves = _writer(tmp_path, flush_interval=10.0, flush_batch=1000)
        writer.start()
        _record(state, writer, 200)
        writer.close()
        assert len(saves) <= 2
        assert len(saves[-1]["downloads"]) == 200

    def test_batch_threshold_triggers_an_early_write(self, tmp_path):
        state, writer, saves = _writer(tmp_path, flush_interval=3600.0, flush_batch=10)
        writer.start()
        _record(state, writer, 10, prefix="first")
        # Let the writer thread act on the threshold before the next burst.
        time.sleep(0.2)
        _record(state, writer, 10, prefix="second")
        writer.close()
        assert len(saves) >= 2
        assert len(saves[-1]["downloads"]) == 20

    def test_interval_bounds_how_long_work_can_be_lost(self, tmp_path):
        state, writer, saves = _writer(tmp_path, flush_interval=0.05, flush_batch=1000)
        writer.start()
        _record(state, writer, 1)
        time.sleep(0.4)
        assert len(saves) >= 1
        assert "a::0" in saves[-1]["downloads"]
        writer.close()


class TestDurability:
    def test_flush_is_synchronous(self, tmp_path):
        state, writer, saves = _writer(tmp_path, flush_interval=3600.0)
        writer.start()
        _record(state, writer, 3)
        writer.flush()
        assert len(saves) == 1
        assert len(saves[-1]["downloads"]) == 3
        writer.close()

    def test_flush_with_nothing_pending_is_a_no_op(self, tmp_path):
        state, writer, saves = _writer(tmp_path)
        writer.start()
        writer.flush()
        assert saves == []
        writer.close()

    def test_close_persists_a_long_interval_run(self, tmp_path):
        """close() must not lose work even when the interval has not elapsed."""
        state, writer, saves = _writer(tmp_path, flush_interval=3600.0)
        writer.start()
        _record(state, writer, 5)
        writer.close()
        assert len(saves[-1]["downloads"]) == 5

    def test_close_is_idempotent(self, tmp_path):
        state, writer, saves = _writer(tmp_path)
        writer.start()
        writer.close()
        writer.close()

    def test_start_is_idempotent(self, tmp_path):
        state, writer, saves = _writer(tmp_path)
        writer.start()
        thread = writer._thread
        writer.start()
        assert writer._thread is thread
        writer.close()

    def test_writes_reach_the_real_file(self, tmp_path):
        state = {"downloads": {}}
        writer = CoalescingStateWriter(state, tmp_path, ".state.json", flush_interval=3600.0)
        writer.start()
        with writer.lock:
            state["downloads"]["a::b"] = {"status": "downloaded"}
        writer.record()
        writer.close()
        on_disk = json.loads((tmp_path / ".state.json").read_text(encoding="utf-8"))
        assert on_disk["downloads"]["a::b"]["status"] == "downloaded"


class TestSnapshot:
    def test_snapshot_does_not_alias_the_live_state(self, tmp_path):
        """A write in progress must not see later mutations."""
        state, writer, saves = _writer(tmp_path, flush_interval=3600.0)
        writer.start()
        with writer.lock:
            state["downloads"]["a"] = {"status": "downloaded"}
        writer.record()
        writer.flush()
        with writer.lock:
            state["downloads"]["b"] = {"status": "failed"}
        assert "b" not in saves[-1]["downloads"]
        writer.close()

    def test_top_level_keys_survive(self, tmp_path):
        state, writer, saves = _writer(tmp_path, flush_interval=3600.0)
        state["version"] = 3
        writer.start()
        _record(state, writer, 1)
        writer.flush()
        assert saves[-1]["version"] == 3
        writer.close()


class TestConcurrency:
    def test_concurrent_writers_do_not_lose_entries(self, tmp_path):
        state, writer, saves = _writer(tmp_path, flush_interval=0.02, flush_batch=50)

        def worker(prefix: str) -> None:
            for index in range(40):
                with writer.lock:
                    state["downloads"][f"{prefix}::{index}"] = {"status": "downloaded"}
                writer.record()

        writer.start()
        threads = [threading.Thread(target=worker, args=(f"t{n}",)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        writer.close()
        assert len(saves[-1]["downloads"]) == 160

    def test_failed_write_does_not_kill_the_writer(self, tmp_path):
        state = {"downloads": {}}
        attempts = []

        def flaky(state, output_dir, filename=None):
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError("disk full")

        writer = CoalescingStateWriter(
            state, tmp_path, ".state.json", flush_interval=0.02, save_fn=flaky
        ).start()
        with writer.lock:
            state["downloads"]["a"] = {"status": "downloaded"}
        writer.record()
        time.sleep(0.3)
        assert len(attempts) >= 2, "the writer should retry after a failed write"
        assert writer._thread is not None and writer._thread.is_alive()
        writer.close()


class TestContextManager:
    def test_closes_on_exit(self, tmp_path):
        state = {"downloads": {}}
        with CoalescingStateWriter(state, tmp_path, ".state.json", flush_interval=3600.0) as w:
            with w.lock:
                state["downloads"]["a"] = {"status": "downloaded"}
            w.record()
        assert json.loads((tmp_path / ".state.json").read_text())["downloads"] == {
            "a": {"status": "downloaded"}
        }
        assert w._thread is None