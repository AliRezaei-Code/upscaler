"""Stopping a job stops it: no orphan processes, no half-written video.

Two things are asserted, because they fail separately. The frames that were in
flight are abandoned and the encode never runs, so there is no truncated video;
and the worker processes are gone, so a second run can start without two pools
fighting over the same work directory.

The second point is the one the ported script got wrong. It reached into
`executor._processes` — a private dictionary, populated asynchronously and
mutated by another thread — and iterated it from a signal handler, which raises
`RuntimeError: dictionary changed size during iteration` at exactly the moment
it is most needed. This pipeline uses the public API and a bounded grace period.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from core import pipeline
from core.config import JobConfig
from core.errors import UpscalerError
from core.pipeline import run_job
from tests.fake_backend import FakeBackend
from tests.support import collector, make_config, quiet

#: Long enough that a cancel lands between frames, short enough for a test.
FRAME_DELAY = 0.4


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "select_backend", lambda devices, model: FakeBackend)


def frames_out(cfg: JobConfig) -> Path:
    return cfg.work_dir / "frames_out"


def run_until_first_frame(cfg: JobConfig, cancel: threading.Event) -> list[object]:
    """Run the job on a thread, cancel when one frame lands, return what happened."""
    events, emit = collector()
    failure: list[BaseException] = []

    def _target() -> None:
        try:
            run_job(cfg, emit, cancel)
        except BaseException as exc:  # the test asserts on which exception this is
            failure.append(exc)

    worker = threading.Thread(target=_target, name="job", daemon=True)
    worker.start()
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and worker.is_alive():
        if list(frames_out(cfg).glob("frame_*.png")):
            cancel.set()
            break
        time.sleep(0.05)
    worker.join(timeout=90)
    assert not worker.is_alive(), "the job thread did not finish after a stop"
    return [*events, *failure]


def test_a_cancelled_job_stops_early_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPSCALER_FAKE_BACKEND_DELAY", str(FRAME_DELAY))
    cfg = make_config(tmp_path, frames=8, devices=1)
    outcome = run_until_first_frame(cfg, threading.Event())

    failures = [item for item in outcome if isinstance(item, BaseException)]
    assert failures, "a stopped job must not report success"
    assert isinstance(failures[0], UpscalerError)
    assert "Re-run to resume" in str(failures[0])
    assert not cfg.output_path.exists(), "a stopped job must not encode a video"


def test_a_cancelled_job_leaves_no_worker_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPSCALER_FAKE_BACKEND_DELAY", str(FRAME_DELAY))
    cfg = make_config(tmp_path, frames=8, devices=1)
    before = {child.pid for child in multiprocessing.active_children()}
    run_until_first_frame(cfg, threading.Event())
    time.sleep(0.5)
    after = {child.pid for child in multiprocessing.active_children()}
    assert not after - before, f"a worker outlived the job: {after - before}"


def test_a_cancelled_job_leaves_the_finished_frames_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever it managed to write must be resumable, not corrupt."""
    monkeypatch.setenv("UPSCALER_FAKE_BACKEND_DELAY", str(FRAME_DELAY))
    cfg = make_config(tmp_path, frames=8, devices=1)
    run_until_first_frame(cfg, threading.Event())
    written = sorted(frames_out(cfg).glob("frame_*.png"))
    assert written, "the fake backend should have written at least one frame"
    assert len(written) < 8, "a stopped 8-frame job wrote all of them"

    # A second run, without the delay, must finish the job from where it stopped.
    monkeypatch.setenv("UPSCALER_FAKE_BACKEND_DELAY", "0")
    run_job(cfg, quiet(), threading.Event())
    assert len(list(frames_out(cfg).glob("frame_*.png"))) == 8
    assert cfg.output_path.is_file()


def test_a_stop_requested_before_the_pool_starts_still_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPSCALER_FAKE_BACKEND_DELAY", "0")
    cfg = make_config(tmp_path, frames=4)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(UpscalerError, match="Re-run to resume"):
        run_job(cfg, quiet(), cancel)
    assert not cfg.output_path.exists()


def test_a_stopped_job_never_emits_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPSCALER_FAKE_BACKEND_DELAY", str(FRAME_DELAY))
    cfg = make_config(tmp_path, frames=8, devices=1)
    events, emit = collector()
    cancel = threading.Event()

    def _target() -> None:
        with pytest.raises(UpscalerError):
            run_job(cfg, emit, cancel)

    worker = threading.Thread(target=_target, name="job", daemon=True)
    worker.start()
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and worker.is_alive():
        if list(frames_out(cfg).glob("frame_*.png")):
            cancel.set()
            break
        time.sleep(0.05)
    worker.join(timeout=90)
    assert not worker.is_alive()
    assert not [event for event in events if event.kind == "done"]
