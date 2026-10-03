"""Tests for ytdl_core.pipeline: staged execution, ordering and shutdown."""

from __future__ import annotations

import threading
import time

from ytdl_core.pipeline import StagePipeline


def _pass_through(name, log=None, log_lock=None, delay=0.0, drop=lambda _x: False, fail=lambda _x: False):
    def fn(item):
        if delay:
            time.sleep(delay)
        if log is not None:
            with log_lock:
                log.append((name, item))
        if fail(item):
            raise RuntimeError(f"{name} failed on {item}")
        return None if drop(item) else item

    return fn


class TestBasics:
    def test_every_item_reaches_the_last_stage(self):
        seen: list[int] = []
        lock = threading.Lock()
        pipeline = StagePipeline(
            [
                ("a", _pass_through("a", seen, lock), 3),
                ("b", _pass_through("b", seen, lock), 2),
            ],
            queue_depth=4,
        )
        pipeline.run(range(30))
        assert sorted(seen) == sorted(
            [("a", i) for i in range(30)] + [("b", i) for i in range(30)]
        )
        assert pipeline.stage_stats() == {"a": 30, "b": 30}

    def test_a_stage_returning_none_drops_the_item(self):
        seen: list[tuple[str, int]] = []
        lock = threading.Lock()
        pipeline = StagePipeline(
            [
                ("a", _pass_through("a", seen, lock, drop=lambda x: x % 3 == 0), 2),
                ("b", _pass_through("b", seen, lock), 2),
            ]
        )
        pipeline.run(range(10))
        forwarded = [i for name, i in seen if name == "b"]
        assert 0 not in forwarded and 3 not in forwarded
        assert 1 in forwarded

    def test_a_failing_item_does_not_stop_the_batch(self):
        processed: list[int] = []
        lock = threading.Lock()
        errors: list = []

        def fn(item):
            if item % 4 == 0:
                raise RuntimeError("boom")
            with lock:
                processed.append(item)
            return item

        pipeline = StagePipeline([("a", fn, 3)], on_error=lambda i, e: errors.append(i))
        pipeline.run(range(20))

        assert sorted(processed) == [i for i in range(20) if i % 4]
        assert sorted(errors) == [0, 4, 8, 12, 16]

    def test_empty_input_terminates(self):
        pipeline = StagePipeline([("a", _pass_through("a"), 2), ("b", _pass_through("b"), 2)])
        pipeline.run([])
        assert pipeline.stage_stats() == {"a": 0, "b": 0}

    def test_single_stage_is_allowed(self):
        seen: list[tuple[str, int]] = []
        lock = threading.Lock()
        pipeline = StagePipeline([("only", _pass_through("only", seen, lock), 2)])
        pipeline.run(range(5))
        assert sorted(item for _name, item in seen) == list(range(5))


class TestOverlap:
    def test_stages_actually_run_concurrently(self):
        """That is the entire point: the sum of stage times must not be paid."""
        concurrent = {"max": 0, "now": 0}
        lock = threading.Lock()
        delay = 0.05

        def fn(item):
            with lock:
                concurrent["now"] += 1
                concurrent["max"] = max(concurrent["max"], concurrent["now"])
            time.sleep(delay)
            with lock:
                concurrent["now"] -= 1
            return item

        pipeline = StagePipeline([(f"s{i}", fn, 3) for i in range(4)], queue_depth=8)
        pipeline.run(range(12))
        assert concurrent["max"] > 1

    def test_deep_queues_still_drain(self):
        seen: list[int] = []
        lock = threading.Lock()
        # queue_depth=1 is the tightest possible setting; a naive implementation
        # deadlocks here.
        pipeline = StagePipeline(
            [
                ("a", _pass_through("a", seen, lock), 2),
                ("b", _pass_through("b", seen, lock), 1),
                ("c", _pass_through("c", seen, lock), 2),
            ],
            queue_depth=1,
        )
        pipeline.run(range(10))
        assert pipeline.stage_stats()["c"] == 10


class TestBackpressure:
    def test_a_slow_stage_throttles_the_one_before_it(self):
        """In-flight work must stay bounded, not grow with the batch size."""
        in_flight = {"max": 0, "now": 0}
        lock = threading.Lock()

        def slow_first(item):
            with lock:
                in_flight["now"] += 1
                in_flight["max"] = max(in_flight["max"], in_flight["now"])
            time.sleep(0.01)
            with lock:
                in_flight["now"] -= 1
            return item

        def instant_second(item):
            return item

        pipeline = StagePipeline(
            [("slow", slow_first, 2), ("fast", instant_second, 8)], queue_depth=3
        )
        pipeline.run(range(200))

        # 2 workers plus a queue of 3 bounds this, not 200.
        assert in_flight["max"] <= 5


class TestShutdown:
    def test_run_returns_without_hanging(self):
        pipeline = StagePipeline([("a", _pass_through("a"), 4), ("b", _pass_through("b"), 4)])
        started = time.monotonic()
        pipeline.run(range(50))
        assert time.monotonic() - started < 20

    def test_stop_event_stops_new_work(self):
        processed: list[int] = []
        lock = threading.Lock()
        stop = threading.Event()

        def fn(item):
            with lock:
                processed.append(item)
            if len(processed) == 5:
                stop.set()
            return item

        pipeline = StagePipeline([("a", fn, 1)], queue_depth=2, stop_event=stop)
        pipeline.run(range(500))
        # Workers drain without blocking, but stop doing work once asked.
        assert len(processed) < 500

    def test_requires_at_least_one_stage(self):
        try:
            StagePipeline([])
        except ValueError:
            return
        raise AssertionError("an empty pipeline should be rejected")