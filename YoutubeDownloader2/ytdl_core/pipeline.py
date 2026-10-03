"""
Stage-bounded execution: a pipeline of queues, one pool per stage.

The batch runner used to be a plain fork-join: one task per song, one worker
slot per song, held for the song's entire life. That is the wrong shape for this
workload, because a song's stages contend for completely different resources:

* searching is network-bound and wants many threads;
* the decision model is latency-bound and wants few;
* the post-download checks decode audio and want about one worker per core;
* AcoustID publishes a process-wide request budget that no amount of thread
  count can buy past.

Sharing one pool between them means the slowest stage sets the size of every
other one. Twelve ffmpeg decodes running next to twelve full downloads do not
make the downloads faster; they make the decodes slower too.

Running the stages as separate queues with separate pools decouples them. Each
stage's size can be set against the resource it actually contends for, and
because the queues are bounded, a slow stage applies backpressure to the stage
feeding it instead of letting an unbounded backlog accumulate. A batch of 500
songs does not put 500 searches, 500 downloads and 500 ffmpeg decodes in flight
simultaneously -- it puts a few of each, which is the whole point.

Nothing here knows what a "song" is. The stages are plain callables over
callables, which keeps the machinery testable on its own.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Generic, Optional, TypeVar

__all__ = ["Stage", "StagePipeline", "StageFailure"]

T = TypeVar("T")


class StageFailure:
    """Wraps an exception raised by a stage so the pipeline can report it."""

    __slots__ = ("item", "error")

    def __init__(self, item: Any, error: BaseException) -> None:
        self.item = item
        self.error = error


class Stage(Generic[T]):
    """One pipeline stage: *workers* threads draining *input_queue*.

    A stage returns the item to pass on, or ``None`` to drop it (the stage
    already decided the item's fate). Raising aborts just that item: the
    :class:`StageFailure` is handed to ``on_error`` and the pipeline keeps going,
    because one song failing must never stop the batch.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[T], Optional[T]],
        *,
        workers: int,
        input_queue: "queue.Queue[Any]",
        output_queue: Optional["queue.Queue[Any]"],
        stop_event: threading.Event,
        on_error: Optional[Callable[[Any, BaseException], None]] = None,
        sentinel: Any = None,
    ) -> None:
        self.name = name
        self.fn = fn
        self.workers = max(1, int(workers))
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.on_error = on_error
        self.sentinel = sentinel
        self._threads: list[threading.Thread] = []
        self.processed = 0
        self._counter_lock = threading.Lock()

    def _emit(self, item: Any) -> None:
        if self.output_queue is not None:
            self.output_queue.put(item)

    def _work(self) -> None:
        while True:
            item = self.input_queue.get()
            try:
                if item is self.sentinel:
                    return
                if self.stop_event.is_set():
                    # Still drain, so the upstream stages never block on a full
                    # queue, but stop doing work.
                    continue
                try:
                    outcome = self.fn(item)
                except BaseException as error:  # noqa: BLE001 - reported, not swallowed
                    if self.on_error is not None:
                        self.on_error(item, error)
                    continue
                with self._counter_lock:
                    self.processed += 1
                if outcome is not None:
                    self._emit(outcome)
            finally:
                self.input_queue.task_done()

    def start(self) -> None:
        for index in range(self.workers):
            thread = threading.Thread(
                target=self._work, name=f"ytdl-{self.name}-{index}", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def join(self) -> None:
        for thread in self._threads:
            thread.join()


class StagePipeline(Generic[T]):
    """Runs a sequence of stages over items, each stage with its own pool.

    Parameters
    ----------
    stages:
        ``[(name, fn, workers), ...]`` in order.
    queue_depth:
        Per-queue bound. This is the backpressure knob: a depth of one would
        serialise the pipeline, an unbounded queue would let the first stage run
        the entire batch ahead of the rest.
    on_error:
        Called with ``(item, exception)`` when a stage raises.
    stop_event:
        Shared cancellation. Checked by every stage before doing work.
    """

    def __init__(
        self,
        stages: Sequence[tuple[str, Callable[[T], Optional[T]], int]],
        *,
        queue_depth: int = 32,
        on_error: Optional[Callable[[Any, BaseException], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        if not stages:
            raise ValueError("a pipeline needs at least one stage")
        self.stage_specs = list(stages)
        self.queue_depth = max(1, int(queue_depth))
        self.on_error = on_error
        self.stop_event = stop_event or threading.Event()
        self.queues: list[queue.Queue[Any]] = []
        self.stages: list[Stage[T]] = []
        self._build()

    def _build(self) -> None:
        count = len(self.stage_specs)
        for index in range(count):
            self.queues.append(queue.Queue(maxsize=self.queue_depth))

        sentinel = object()
        for index, (name, fn, workers) in enumerate(self.stage_specs):
            output = self.queues[index + 1] if index + 1 < count else None
            stage = Stage(
                name,
                fn,
                workers=workers,
                input_queue=self.queues[index],
                output_queue=output,
                stop_event=self.stop_event,
                on_error=self.on_error,
                sentinel=sentinel,
            )
            self.stages.append(stage)

    def run(self, items: Iterable[T]) -> None:
        """Feed *items* through the pipeline and block until every stage is idle."""
        for stage in self.stages:
            stage.start()

        try:
            for item in items:
                self.queues[0].put(item)
            for queue_ in self.queues:
                queue_.join()
        finally:
            # One sentinel per worker per queue: a single sentinel would release
            # exactly one thread and leave the rest parked in ``get()`` forever,
            # so ``join()`` would never return. Workers exit on the first one
            # they see and the rest are simply never picked up.
            for index, stage in enumerate(self.stages):
                for _ in range(stage.workers):
                    self.queues[index].put(stage.sentinel)
            for stage in self.stages:
                stage.join()

    def stop(self) -> None:
        """Ask every stage to stop taking new work (an interrupt, say)."""
        self.stop_event.set()

    def stage_stats(self) -> dict[str, int]:
        return {stage.name: stage.processed for stage in self.stages}