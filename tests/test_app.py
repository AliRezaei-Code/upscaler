"""Behaviour of the task driver: threads, queue, cancellation, guards.

Every test runs real threads against a real `queue.Queue` and no GPU, and
`core.pipeline` is never imported: `start` calls `core.app.run_job`, and that
name is replaced here with a fake of the same shape, which is exactly the seam
the deferred import exists to provide.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Protocol

import pytest

from core.app import Emit, TaskHandle, UpscalerApp
from core.config import Device, JobConfig
from core.errors import ConfigError
from core.events import PipelineEvent

DRAIN_TIMEOUT = 10.0


class Cancel(Protocol):
    """What a task is handed in place of a `multiprocessing.Event`.

    Declared here rather than imported from `core.worker` so that this module
    stays clear of the pipeline's dependency graph.
    """

    def is_set(self) -> bool: ...


def make_device() -> Device:
    """One usable CPU device — enough for a `JobConfig` the fakes never run."""
    return Device(
        index=0,
        name="CPU",
        vendor="cpu",
        backend="cpu",
        total_memory_bytes=0,
        pci_bus_id=None,
        compute_capability=None,
        usable=True,
        unusable_reason=None,
    )


def make_config(tmp_path: Path) -> JobConfig:
    """A job configuration pointing entirely inside `tmp_path`."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"not really a video")
    return JobConfig(
        input_path=source,
        output_path=tmp_path / "source.upscaled.mp4",
        model_path=tmp_path / "model.onnx",
        work_dir=tmp_path / "work",
        devices=(make_device(),),
    )


def drain(app: UpscalerApp, timeout: float = DRAIN_TIMEOUT) -> list[PipelineEvent]:
    """Consume one task's events, re-entering when the sentinel arrives.

    The generator runs on a thread so a missing sentinel fails the test in
    `timeout` seconds instead of hanging the suite.
    """
    collected: list[PipelineEvent] = []
    failures: list[BaseException] = []

    def consume() -> None:
        try:
            for ev in app.events():
                collected.append(ev)
        except BaseException as exc:  # surfaced on the calling thread
            failures.append(exc)

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail(f"events() never terminated; collected {collected}")
    if failures:
        raise failures[0]
    return collected


def drain_n(
    app: UpscalerApp, count: int, timeout: float = DRAIN_TIMEOUT
) -> list[PipelineEvent]:
    """Consume `count` events across as many tasks as it takes.

    This is the front-end's loop: one `events()` call covers one task, so a
    second producer's events arrive only after re-entering.
    """
    collected: list[PipelineEvent] = []
    failures: list[BaseException] = []

    def consume() -> None:
        try:
            while len(collected) < count:
                for ev in app.events():
                    collected.append(ev)
                    if len(collected) >= count:
                        return
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail(f"only {len(collected)} of {count} arrived; got {collected}")
    if failures:
        raise failures[0]
    return collected


def test_a_submitted_task_delivers_its_events_in_order_then_ends_the_drain() -> None:
    app = UpscalerApp()

    def task(emit: Emit, stop: threading.Event) -> str:
        for index in range(3):
            emit(PipelineEvent(kind="log", message=f"line {index}"))
        return "finished"

    handle = app.submit("model:RealESRGAN_x4plus", task)

    events = drain(app)

    assert [ev.message for ev in events] == ["line 0", "line 1", "line 2"]
    assert {ev.task_id for ev in events} == {"model:RealESRGAN_x4plus"}
    assert app.results["model:RealESRGAN_x4plus"] == "finished"
    assert app.last_error == {}
    handle.thread.join(timeout=DRAIN_TIMEOUT)
    assert not handle.thread.is_alive()


def test_a_task_that_raises_reports_one_error_and_still_ends_the_drain() -> None:
    app = UpscalerApp()

    def task(emit: Emit, stop: threading.Event) -> None:
        emit(PipelineEvent(kind="stage", stage="probe"))
        raise ConfigError("Input video not found: /tmp/gone.mp4")

    app.submit("community", task)

    events = drain(app)

    errors = [ev for ev in events if ev.kind == "error"]
    assert len(errors) == 1
    assert errors[0].message == "Input video not found: /tmp/gone.mp4"
    assert errors[0].task_id == "community"
    assert app.last_error == {"community": "Input video not found: /tmp/gone.mp4"}
    assert "community" not in app.results


def test_a_non_upscaler_exception_is_reported_too() -> None:
    """A non-`UpscalerError` is still a failure a user has to be shown."""

    def boom(emit: Emit, stop: threading.Event) -> None:
        raise ValueError("r_frame_rate was 'N/A'")

    app = UpscalerApp()
    app.submit("job", boom)

    events = drain(app)

    assert [ev.message for ev in events if ev.kind == "error"] == [
        "r_frame_rate was 'N/A'"
    ]
    assert app.last_error["job"] == "r_frame_rate was 'N/A'"


def test_a_full_queue_drops_and_counts_instead_of_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.app.QUEUE_PUT_TIMEOUT_SECONDS", 0.05)
    app = UpscalerApp(queue_size=2)
    emitted = threading.Event()

    def task(emit: Emit, stop: threading.Event) -> None:
        for index in range(50):
            emit(PipelineEvent(kind="log", message=f"line {index}"))
        emitted.set()

    handle = app.submit("job", task)

    # Two events fit and the other 48 must be dropped rather than waited on.
    # The task cannot set `emitted` until it has tried to emit all of them, so
    # this is also the assertion that a stalled consumer did not stall it.
    assert emitted.wait(timeout=2.0), "emitting into a full queue blocked"
    assert handle.thread.is_alive(), "the sentinel was dropped rather than queued"

    events = drain(app)

    assert [ev.message for ev in events[:2]] == ["line 0", "line 1"]
    reports = [ev for ev in events if "progress events dropped" in ev.message]
    assert len(reports) == 1
    assert reports[0].kind == "log"
    assert reports[0].task_id == "job"
    assert reports[0].message == "48 progress events dropped — the UI is not keeping up"
    handle.thread.join(timeout=DRAIN_TIMEOUT)
    assert not handle.thread.is_alive()


def test_a_second_start_raises_while_a_job_runs_then_works_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    ran: list[JobConfig] = []

    def fake_run_job(cfg: JobConfig, emit: Emit, cancel: Cancel) -> Path:
        ran.append(cfg)
        started.set()
        release.wait(timeout=DRAIN_TIMEOUT)
        emit(PipelineEvent(kind="done", message=str(cfg.output_path)))
        return cfg.output_path

    monkeypatch.setattr("core.app.run_job", fake_run_job)
    app = UpscalerApp()
    cfg = make_config(tmp_path)

    first = app.start(cfg)
    assert started.wait(timeout=DRAIN_TIMEOUT)

    with pytest.raises(RuntimeError, match="A job is already running"):
        app.start(cfg)
    assert ran == [cfg], "the refused start must not have run a job"

    release.set()
    first.thread.join(timeout=DRAIN_TIMEOUT)
    assert not first.stop.is_set()

    # The reservation is released when the thread finishes, not when it is
    # asked to stop: a second job over one work directory would write the same
    # frame names non-atomically.
    second = app.start(cfg)
    second.thread.join(timeout=DRAIN_TIMEOUT)

    assert ran == [cfg, cfg]
    assert not app.is_running()
    done = drain_n(app, 2)
    assert [ev.kind for ev in done] == ["done", "done"]
    assert {ev.message for ev in done} == {str(cfg.output_path)}


def test_a_job_that_ends_before_start_returns_still_frees_the_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job short enough to finish inside `start` must not wedge the app.

    `start` stores the handle after `submit` has already started the thread, so
    a thread that ends in between has to release the reservation some other way.
    """
    monkeypatch.setattr("core.app.run_job", lambda cfg, emit, cancel: cfg.output_path)
    app = UpscalerApp()
    cfg = make_config(tmp_path)

    for _ in range(3):
        handle = app.start(cfg)
        handle.thread.join(timeout=DRAIN_TIMEOUT)
        assert not app.is_running()


def test_a_failed_job_reports_the_error_and_frees_the_start_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(cfg: JobConfig, emit: Emit, cancel: Cancel) -> Path:
        raise ConfigError("Need ~9750 GB for frames, 0 GB free at /tmp")

    def succeeding(cfg: JobConfig, emit: Emit, cancel: Cancel) -> Path:
        return cfg.output_path

    monkeypatch.setattr("core.app.run_job", failing)
    app = UpscalerApp()
    cfg = make_config(tmp_path)

    handle = app.start(cfg)
    events = drain(app)

    assert [ev.message for ev in events if ev.kind == "error"] == [
        "Need ~9750 GB for frames, 0 GB free at /tmp"
    ]
    assert app.last_error["job"] == "Need ~9750 GB for frames, 0 GB free at /tmp"
    assert "job" not in app.results
    handle.thread.join(timeout=DRAIN_TIMEOUT)
    assert not app.is_running()

    monkeypatch.setattr("core.app.run_job", succeeding)
    retry = app.start(cfg)
    retry.thread.join(timeout=DRAIN_TIMEOUT)
    assert app.results["job"] == cfg.output_path


def test_stop_sets_both_cancel_flags_and_the_job_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    running = threading.Event()
    cancel_seen: list[bool] = []

    def fake_run_job(cfg: JobConfig, emit: Emit, cancel: Cancel) -> Path:
        running.set()
        while not cancel.is_set():
            emit(PipelineEvent(kind="device_progress", device_index=0, processed=1))
            time.sleep(0.01)
        cancel_seen.append(cancel.is_set())
        return cfg.output_path

    monkeypatch.setattr("core.app.run_job", fake_run_job)
    app = UpscalerApp()

    handle = app.start(make_config(tmp_path))
    assert running.wait(timeout=DRAIN_TIMEOUT)
    assert app.is_running()

    app.stop()

    # `stop` returns without joining; the job ends on its own shortly after.
    assert handle.stop.is_set(), "the threading.Event every task polls was not set"
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and app.is_running():
        time.sleep(0.01)
    assert cancel_seen == [True], (
        "the multiprocessing.Event the pool shares was not set"
    )
    assert not app.is_running()
    handle.thread.join(timeout=DRAIN_TIMEOUT)
    assert not handle.thread.is_alive()


def test_stop_with_no_job_running_is_harmless() -> None:
    app = UpscalerApp()
    app.stop()
    assert not app.is_running()


def test_clean_work_dir_refuses_while_a_job_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    running = threading.Event()
    release = threading.Event()

    def fake_run_job(cfg: JobConfig, emit: Emit, cancel: Cancel) -> Path:
        running.set()
        release.wait(timeout=DRAIN_TIMEOUT)
        return cfg.output_path

    monkeypatch.setattr("core.app.run_job", fake_run_job)
    app = UpscalerApp()
    cfg = make_config(tmp_path)
    cfg.work_dir.mkdir(parents=True)
    (cfg.work_dir / "frame_00000001.png").write_bytes(b"frame")

    handle = app.start(cfg)
    assert running.wait(timeout=DRAIN_TIMEOUT)
    try:
        with pytest.raises(ConfigError, match="stop it before deleting"):
            app.clean_work_dir(cfg)
    finally:
        release.set()
        handle.thread.join(timeout=DRAIN_TIMEOUT)
    assert cfg.work_dir.is_dir(), "the work directory was deleted under a running job"


def test_clean_work_dir_removes_only_the_work_directory(tmp_path: Path) -> None:
    app = UpscalerApp()
    cfg = make_config(tmp_path)
    cfg.output_path.write_bytes(b"output video")
    cfg.work_dir.mkdir(parents=True)
    (cfg.work_dir / "frames_out").mkdir()
    (cfg.work_dir / "frames_out" / "frame_00000001.png").write_bytes(b"frame")
    cfg.model_path.write_bytes(b"model weights")

    app.clean_work_dir(cfg)

    assert not cfg.work_dir.exists()
    assert cfg.output_path.read_bytes() == b"output video"
    assert cfg.input_path.read_bytes() == b"not really a video"
    assert cfg.model_path.read_bytes() == b"model weights"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "model.onnx",
        "source.mp4",
        "source.upscaled.mp4",
    ]


def test_cleaning_a_work_dir_that_is_already_gone_is_not_an_error(
    tmp_path: Path,
) -> None:
    app = UpscalerApp()
    cfg = make_config(tmp_path)
    assert not cfg.work_dir.exists()

    app.clean_work_dir(cfg)

    assert not cfg.work_dir.exists()


def test_two_producers_submitted_at_once_both_reach_the_consumer() -> None:
    app = UpscalerApp()
    handles: list[TaskHandle] = []
    lock = threading.Lock()

    def submit_task(task_id: str) -> None:
        def task(emit: Emit, stop: threading.Event) -> str:
            for index in range(3):
                emit(PipelineEvent(kind="log", message=f"{task_id} {index}"))
            return task_id

        handle = app.submit(task_id, task)
        with lock:
            handles.append(handle)

    submitters = [
        threading.Thread(target=submit_task, args=(f"model:{i}",)) for i in range(2)
    ]
    for thread in submitters:
        thread.start()
    for thread in submitters:
        thread.join(timeout=DRAIN_TIMEOUT)

    assert sorted(h.task_id for h in handles) == ["model:0", "model:1"]

    events = drain_n(app, 6)
    assert sorted(ev.message for ev in events) == sorted(
        f"{task_id} {index}" for task_id in ("model:0", "model:1") for index in range(3)
    )
    assert {ev.task_id for ev in events} == {"model:0", "model:1"}
    for handle in handles:
        handle.thread.join(timeout=DRAIN_TIMEOUT)
        assert not handle.thread.is_alive()
    assert app.results == {"model:0": "model:0", "model:1": "model:1"}
