"""
Coalescing writer for the download state file.

The state file is a whole-file document: one JSON object, rewritten from
scratch, with an ``fsync``, on every persisted outcome -- including every
failure path. With N songs that is O(N^2) bytes written and O(N) fsyncs, and
because the write happened while the state lock was held, every one of those
fsyncs blocked all the other workers from recording anything at all. The lock
protecting a shared dictionary ended up serialising the entire batch behind the
disk.

This module keeps the lock where it belongs -- around the in-memory mutation --
and moves the expensive part off the hot path:

* callers mutate the state under the lock, then signal;
* one background thread owns the file, coalescing a burst of updates into a
  single write, triggered either by *pending* updates or by elapsed time;
* serialisation and I/O happen against a snapshot taken outside the lock, so a
  slow disk never stalls the pipeline.

Writes remain atomic (``write_json_atomic``), and :meth:`flush` gives callers a
synchronous guarantee that everything recorded so far is on disk -- which is what
the interrupt and disk-full paths need.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .state import save_state

__all__ = ["CoalescingStateWriter"]


class CoalescingStateWriter:
    """Serialises state mutations in memory; writes them out on a background thread.

    Parameters
    ----------
    state:
        The live state dict. Mutated in place under :attr:`lock`.
    output_dir / state_filename:
        Where :func:`ytdl_core.state.save_state` writes.
    flush_interval / flush_batch:
        A write happens once *either* ``flush_batch`` updates are pending *or*
        ``flush_interval`` seconds have passed since the last write. The interval
        is the important one: it bounds how long a crash can lose work.
    save_fn:
        Injection point for tests.
    """

    def __init__(
        self,
        state: dict[str, Any],
        output_dir: Path | str,
        state_filename: Optional[str] = None,
        *,
        flush_interval: float = 1.0,
        flush_batch: int = 25,
        save_fn: Optional[Callable[..., None]] = None,
    ) -> None:
        self.state = state
        self.output_dir = Path(output_dir)
        self.state_filename = state_filename
        self.flush_interval = max(0.0, float(flush_interval))
        self.flush_batch = max(1, int(flush_batch))
        self._save = save_fn or save_state

        self.lock = threading.Lock()
        self._wake = threading.Condition()
        self._pending = 0
        self._dirty = False
        self._closed = False
        self._due_at: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._writes = 0

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> "CoalescingStateWriter":
        """Start the background writer. Idempotent."""
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run, name="ytdl-state-writer", daemon=True
        )
        self._thread.start()
        return self

    def close(self) -> None:
        """Flush everything pending and stop the writer."""
        with self._wake:
            self._closed = True
            self._dirty = True
            self._wake.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=30)
            self._thread = None

    def __enter__(self) -> "CoalescingStateWriter":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- mutation -------------------------------------------------------------

    def record(self) -> None:
        """Note that the state changed under the caller's own lock.

        The caller has already mutated ``self.state`` while holding
        :attr:`lock` (the same discipline the rest of the pipeline uses when it
        reads state). This only signals the writer.
        """
        self._signal()

    def update(self, mutator: Callable[[dict[str, Any]], Any]) -> Any:
        """Run *mutator* against the state under the lock, then signal."""
        with self.lock:
            result = mutator(self.state)
        self._signal()
        return result

    def _signal(self) -> None:
        with self._wake:
            self._pending += 1
            self._dirty = True
            if self._due_at is None:
                # Arm the deadline on the first update of a burst, not on every
                # one, so a steady stream of results coalesces into one write
                # per interval instead of one write per result.
                self._due_at = time.monotonic() + self.flush_interval
            self._wake.notify_all()

    # -- writing --------------------------------------------------------------

    def _snapshot(self) -> dict[str, Any]:
        """Copy just enough of the state to serialise it outside the lock.

        Entries under ``downloads`` are replaced wholesale on every write rather
        than mutated in place, so a one-level copy of that mapping is a stable
        snapshot. Without this the serialisation -- which is the expensive part --
        would have to happen under the lock.
        """
        with self.lock:
            snapshot = dict(self.state)
            downloads = self.state.get("downloads")
            if isinstance(downloads, dict):
                snapshot["downloads"] = dict(downloads)
            return snapshot

    def _write_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._save(snapshot, self.output_dir, self.state_filename)
        self._writes += 1

    def _take_pending_locked(self) -> None:
        """Take ownership of the pending writes and re-arm the deadline."""
        self._pending = 0
        self._dirty = False
        self._due_at = None

    def flush(self, force: bool = True) -> None:
        """Write pending state to disk now.

        With ``force=False`` this is a no-op when nothing is pending, which lets
        hot paths call it defensively without paying for a write.
        """
        with self._wake:
            if not self._dirty:
                return
            if not force and self._pending < self.flush_batch:
                return
            self._take_pending_locked()
        self._write_snapshot(self._snapshot())

    def _wait_until_due_locked(self) -> bool:
        """Block until a write is due. False means "closing, nothing to do"."""
        while True:
            if not self._dirty:
                # Idle. Wake periodically so a close() is noticed promptly.
                self._wake.wait(timeout=0.5)
                # Re-check dirty *before* honouring close: close() raises the
                # dirty flag itself, and exiting here would silently discard it.
                if self._closed and not self._dirty:
                    return False
                continue
            if self._pending >= self.flush_batch or self.flush_interval <= 0:
                return True
            now = time.monotonic()
            due_at = self._due_at if self._due_at is not None else now
            if now >= due_at:
                return True
            if self._closed:
                return True
            self._wake.wait(timeout=due_at - now)

    def _run(self) -> None:
        while True:
            with self._wake:
                if not self._wait_until_due_locked():
                    return
                closing = self._closed
                self._take_pending_locked()

            snapshot = self._snapshot()
            try:
                self._write_snapshot(snapshot)
            except Exception:
                # A failed write must not kill the writer. Put the work back so
                # the next tick retries with whatever accumulated since.
                with self._wake:
                    self._dirty = True
                    self._pending += 1
                    if self._due_at is None:
                        self._due_at = time.monotonic() + self.flush_interval

            if closing:
                # Anything that arrived while we were writing gets one last pass.
                with self._wake:
                    if not self._dirty:
                        return

    @property
    def write_count(self) -> int:
        """How many times the file has actually been written this run."""
        return self._writes