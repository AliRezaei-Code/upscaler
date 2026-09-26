"""One driver for the four producers, and the three failure modes it exists to stop.

The Upscale tab runs a job; the Models tab downloads a checkpoint; the Community
tab scrapes OpenModelDB; the Runtimes tab unpacks a wheel. All four block for
minutes, all four have progress, and none of them has a thread, a stop path or a
place in a job-scoped design. `UpscalerApp` is that place: `submit` is the
channel every producer goes through, `start` is the job-specific wrapper, and
both front-ends read one queue of `PipelineEvent` and nothing else.

**One bounded queue, and a sentinel per task.** `events()` yields from a
`queue.Queue(maxsize=queue_size)` and *returns* when it pulls the `None`
sentinel that each producing thread pushes in its `finally`. Without that
sentinel the consumer blocks forever after the last `done`, the thread is never
joined, and the app cannot shut down. Because every task pushes one, a consumer
must re-enter after each return — a drain is one task's worth of events:

```python
while True:
    for ev in app.events():
        widget.show(ev)
```

**Emit must never block, and must be honest when it drops.** An emit waits up to
`QUEUE_PUT_TIMEOUT_SECONDS` for room — long enough that a UI pausing for a modal
dialog loses nothing — and then drops the event and counts it. A blocking put
from a pipeline thread would waste a whole machine's work behind a log widget
nobody is reading, so once the queue has been full the app stops waiting until
an event gets through again. The count is reported as one `log` event, at most
once a second, and it is the one event guaranteed to be delivered: a saturated
queue is exactly when the user needs to be told the numbers are lying.

**A re-entrancy guard, because two jobs over one work directory corrupt output.**
`start` raises `RuntimeError` while `_active` is set, checked under a lock. A
double-clicked Start would otherwise run two `ProcessPoolExecutor` pools over
one `work_dir`, writing the same `frame_%08d.png` names non-atomically and
racing on `os.replace`. `_active` is released when the job thread *finishes*,
not when `stop` is asked for: a worker mid-frame can take seconds to notice the
cancel event, and releasing early would let a second job start over a work
directory the first one is still writing.

**Two cancel flags, both set by `stop`.** `TaskHandle.stop` is a
`threading.Event`: the contract every task polls, including the ones that never
touch the process pool. The job additionally needs a `multiprocessing.Event`,
because a `threading.Event` cannot cross a `spawn` boundary into the pool
workers and even a delivered copy is one the parent can never mutate. The app
creates that event up front, in the parent, and hands it to `run_job`, which
passes it to the pool's `initializer`; `stop` sets both and returns without
joining, because a UI thread that joins freezes the window for exactly as long
as a slow frame takes.

**Per-task state lives here, not on the handle.** `TaskHandle` is frozen — it is
a value handed to a front-end — so the return value of a task and the message of
a task that failed are kept in `results` and `last_error`, both keyed by
`task_id`. A failure is also emitted as a `kind="error"` event, because a
producer thread that dies silently shows a progress bar that stops at 40% with
no explanation; `run_job` reports its own failures and returns, so only a
raised exception produces an event here and nothing is double-emitted.

`core.pipeline` is imported inside the wrapper below, never at module scope:
both front-ends import this module on every repaint, and `core.pipeline` drags
in the process pool, the backends and therefore torch and onnxruntime.
"""

from __future__ import annotations

import multiprocessing
import queue
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import JobConfig
from .errors import ConfigError
from .events import PipelineEvent

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as MpEvent

#: What a task is handed to report progress, and the shape `submit` takes.
Emit = Callable[[PipelineEvent], None]
TaskFn = Callable[[Emit, threading.Event], Any]

#: How long one emit waits for room before dropping the event. Long enough that
#: a UI which pauses — a modal dialog, a window being dragged — loses nothing.
QUEUE_PUT_TIMEOUT_SECONDS = 5.0
#: How often the dropped-event count is reported, so a stalled consumer is told
#: once a second rather than once per event.
DROP_REPORT_INTERVAL_SECONDS = 1.0
#: The `task_id` of an upscale job. Front-ends dispatch on it, so a model
#: download's progress cannot land in the job's widgets.
JOB_TASK_ID = "job"


@dataclass(frozen=True)
class TaskHandle:
    """A running task: what to call to stop it, and where to see it.

    Attributes:
        task_id: Which producer this is — `"job"`, `"model:<id>"`, and so on.
        stop: Set by `stop()` and by whoever holds the handle; every task is
            expected to poll it.
        thread: The daemon thread running the task.
    """

    task_id: str
    stop: threading.Event
    thread: threading.Thread


@dataclass
class _ActiveJob:
    """Mutable state for the one job `start` allows at a time.

    `TaskHandle` is frozen and is only assigned once `submit` returns, so the
    reservation needs a cell of its own: the job thread may finish before that
    assignment happens, and both orders have to end with `_active` released.
    """

    handle: TaskHandle | None = None
    cancel: MpEvent | None = None
    stop_requested: bool = False
    finished: bool = False


def run_job(cfg: JobConfig, emit: Emit, cancel: MpEvent) -> Path:
    """Call `core.pipeline.run_job`, importing it at call time.

    This wrapper is the seam `start` goes through, and it exists so that
    `core.app` stays importable without torch, onnxruntime or the process pool:
    the import is deferred to the moment a job actually runs, and a test can
    replace this one name to drive `start` without a pipeline.

    Args:
        cfg: The job to run.
        emit: Progress sink; the job's own events already carry
            `task_id="job"`, and `start`'s emitter stamps it if one is empty.
        cancel: The event `stop()` sets; `run_job` hands it to each spawned
            pool worker through the pool's `initializer`, the only channel that
            survives `spawn`.

    Returns:
        The path of the encoded video.
    """
    from .pipeline import run_job as pipeline_run_job

    return pipeline_run_job(cfg, emit, cancel)


class UpscalerApp:
    """Runs tasks on threads and reports them through one event queue.

    Attributes:
        last_error: `task_id` to the message of the failure that ended it, so a
            front-end that has already drained past the `error` event can still
            show what went wrong. A task that succeeds has no entry.
        results: `task_id` to what its function returned — the output path for a
            job, `None` for a task that returns nothing.
    """

    def __init__(self, queue_size: int = 1024) -> None:
        """Create an app whose queue holds `queue_size` events.

        Args:
            queue_size: Events that may wait for a consumer before emit starts
                dropping. Bounded on purpose: a UI that stops consuming must not
                be able to grow the producer's memory without limit.
        """
        self._queue: queue.Queue[PipelineEvent | None] = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._drop_lock = threading.Lock()
        self._active: _ActiveJob | None = None
        self._dropped = 0
        self._saturated = False
        self._report_due = False
        self._report_task_id = ""
        self._last_report_at: float | None = None
        self.last_error: dict[str, str] = {}
        self.results: dict[str, Any] = {}

    def submit(self, task_id: str, fn: TaskFn) -> TaskHandle:
        """Run `fn(emit, stop)` on a daemon thread and return its handle.

        This is the channel for every producer that is not a job: a model
        download, a community fetch, a runtime install. The thread is a daemon
        so a task the user walked away from cannot keep the interpreter alive.

        Args:
            task_id: How the front-end recognises this task's events.
            fn: Called as `fn(emit, stop)`; whatever it returns is recorded in
                `results[task_id]`, and any exception it raises becomes one
                `error` event and an entry in `last_error[task_id]`.

        Returns:
            The handle, whose `stop` event the caller may set at any time.
        """
        stop = threading.Event()
        thread = threading.Thread(
            target=self._run_task,
            args=(task_id, fn, stop),
            name=f"upscaler:{task_id}",
            daemon=True,
        )
        thread.start()
        return TaskHandle(task_id=task_id, stop=stop, thread=thread)

    def events(self) -> Iterator[PipelineEvent]:
        """Yield events until the next task ends, then return.

        Returns — rather than blocks — at the `None` sentinel that every task
        pushes when it finishes, so a consumer can tell "this task is over" from
        "nothing has happened yet". A front-end therefore re-enters, because one
        return is one task, not the end of the stream:

        ```python
        while True:
            for ev in app.events():
                widget.show(ev)
        ```

        A consumer that stops early simply abandons the generator; producers are
        unaffected either way, because they never wait on this side.
        """
        while True:
            item = self._queue.get()
            if item is None:
                return
            yield item

    def start(self, cfg: JobConfig) -> TaskHandle:
        """Start one upscale job.

        Args:
            cfg: The job to run; `run_job` validates it and reports any problem
                as an `error` event rather than raising here.

        Returns:
            The job's handle.

        Raises:
            RuntimeError: If a job is already running. Two jobs over one work
                directory write the same frame names non-atomically.
        """
        with self._lock:
            if self._active is not None:
                raise RuntimeError("A job is already running")
            state = _ActiveJob()
            self._active = state

        # Created here, in the parent, because a `multiprocessing.Event` is only
        # picklable while a child is being spawned: handing it to the pool as an
        # argument is exactly the mistake the initializer exists to avoid.
        cancel = multiprocessing.get_context("spawn").Event()
        state.cancel = cancel

        def job(emit: Emit, stop: threading.Event) -> Path:
            try:
                return run_job(cfg, emit, cancel)
            finally:
                self._finish_job(state)

        try:
            handle = self.submit(JOB_TASK_ID, job)
        except BaseException:
            self._finish_job(state)
            raise

        with self._lock:
            state.handle = handle
            # `stop` may have been called before this line: it saw no handle, so
            # it recorded the request for the `threading.Event` to catch up on.
            if state.stop_requested:
                handle.stop.set()
            # A job that finished before this assignment released nothing,
            # because `_finish_job` had no handle to compare against.
            if state.finished and self._active is state:
                self._active = None
        return handle

    def stop(self) -> None:
        """Ask the running job to stop, and return immediately.

        Deliberately does not join: a pool worker part-way through a frame can
        take seconds to notice, and a UI thread that waits for that freezes the
        window for the whole of it. `is_running` stays true until the job thread
        really has finished.
        """
        with self._lock:
            state = self._active
            if state is None:
                return
            state.stop_requested = True
            handle = state.handle
            cancel = state.cancel
        if handle is not None:
            handle.stop.set()
        if cancel is not None:
            cancel.set()

    def is_running(self) -> bool:
        """True while a job's thread is alive."""
        with self._lock:
            state = self._active
        return (
            state is not None
            and state.handle is not None
            and state.handle.thread.is_alive()
        )

    def clean_work_dir(self, cfg: JobConfig) -> None:
        """Delete `cfg.work_dir` and nothing else.

        This is the second half of a confirmation the front-end must show first,
        naming this exact path and the gigabytes it will remove: on a long job
        the work directory is hundreds of gigabytes and the deletion cannot be
        undone. The input video, the output and the model are never touched.
        A work directory that is already gone is the common case, not an error.

        Raises:
            ConfigError: If a job is running, because it is writing into that
                directory right now.
        """
        if self.is_running():
            raise ConfigError(
                f"A job is still running; stop it before deleting {cfg.work_dir}"
            )
        try:
            shutil.rmtree(cfg.work_dir, ignore_errors=False)
        except FileNotFoundError:
            return

    def _run_task(self, task_id: str, fn: TaskFn, stop: threading.Event) -> None:
        """Run one task, turning a failure into an event and ending the drain.

        A producer thread that dies silently is the worst outcome available
        here, so every exception becomes exactly one `error` event carrying the
        message verbatim — and is recorded in `last_error` for a UI that has
        already drained past it. `run_job` reports its own failures and returns
        normally, so nothing is double-emitted.
        """
        try:
            result = fn(self._emitter(task_id), stop)
        except Exception as exc:
            # Emitted before the sentinel, so a consumer draining this task sees
            # the failure; recorded as well, for a consumer that already has.
            self._push(
                task_id,
                PipelineEvent(kind="error", task_id=task_id, message=str(exc)),
            )
            with self._lock:
                self.last_error[task_id] = str(exc)
        else:
            with self._lock:
                self.results[task_id] = result
        finally:
            self._push_sentinel()

    def _emitter(self, task_id: str) -> Emit:
        """Build the `emit` a task is handed.

        Events that arrive without a `task_id` are stamped with this task's,
        because the front-ends dispatch on it and a job's unlabelled progress
        would otherwise be dispatched nowhere.
        """

        def emit(ev: PipelineEvent) -> None:
            self._push(task_id, ev if ev.task_id else replace(ev, task_id=task_id))

        return emit

    def _push(self, task_id: str, ev: PipelineEvent) -> None:
        """Put one event, dropping it rather than stalling the producer.

        The first emit into a full queue waits for room, so a consumer that is
        merely busy loses nothing. After a drop the app stops waiting entirely
        until an event gets through: waiting the full timeout on every event of
        a 165,303-frame job would turn a stalled log into days of extra work.
        """
        with self._drop_lock:
            saturated = self._saturated
        queued = self._put_nowait(ev) if saturated else self._put(ev)
        if not queued:
            self._on_drop(task_id)
            return
        with self._drop_lock:
            self._saturated = False
        self._deliver_report(wait=False)

    def _put(self, ev: PipelineEvent | None) -> bool:
        """Put with the full timeout. False means the queue stayed full."""
        try:
            self._queue.put(ev, timeout=QUEUE_PUT_TIMEOUT_SECONDS)
        except queue.Full:
            return False
        return True

    def _put_nowait(self, ev: PipelineEvent | None) -> bool:
        """Put without waiting. False means the queue is full."""
        try:
            self._queue.put_nowait(ev)
        except queue.Full:
            return False
        return True

    def _on_drop(self, task_id: str) -> None:
        """Count the drop, and owe a summary at most once a second."""
        now = time.monotonic()
        with self._drop_lock:
            self._saturated = True
            self._dropped += 1
            if (
                self._last_report_at is not None
                and now - self._last_report_at < DROP_REPORT_INTERVAL_SECONDS
            ):
                return
            self._last_report_at = now
            self._report_due = True
            self._report_task_id = task_id

    def _report_event(self) -> PipelineEvent | None:
        """Build the owed summary, reading the count at this moment.

        Built here rather than at the drop, because a burst of 48 drops inside
        one report interval is one report — and it must say 48, the number the
        consumer actually missed, not 1, the number that happened to fall due.
        """
        with self._drop_lock:
            if not self._report_due:
                return None
            return PipelineEvent(
                kind="log",
                task_id=self._report_task_id,
                message=(
                    f"{self._dropped} progress events dropped — "
                    "the UI is not keeping up"
                ),
            )

    def _deliver_report(self, wait: bool) -> None:
        """Put the owed summary, forgetting the debt only once it is queued."""
        report = self._report_event()
        if report is None:
            return
        if self._put(report) if wait else self._put_nowait(report):
            with self._drop_lock:
                self._report_due = False

    def _push_sentinel(self) -> None:
        """End one task's drain: the drop summary first, then `None`.

        The summary is put before the sentinel and with the same wait, because a
        queue that is still full is precisely when the user needs to be told the
        numbers are stale. The sentinel itself is never dropped — a consumer
        that never sees it blocks forever — and the wait is unbounded for the
        same reason; a task blocked here is a daemon thread, reclaimed at exit.
        """
        self._deliver_report(wait=True)
        self._put(None)

    def _finish_job(self, state: _ActiveJob) -> None:
        """Release the `start` reservation, now that the job thread is done."""
        with self._lock:
            state.finished = True
            if self._active is state:
                self._active = None
