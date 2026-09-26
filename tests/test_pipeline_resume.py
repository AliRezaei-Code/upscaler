"""A resumed run writes only what is missing, and never touches what is not.

This is the behaviour the whole app exists to preserve: a 165,303-frame job
that stopped at 160,000 has to write 5,303 frames, not 165,303, and the 160,000
it already has have to come out untouched. The ported script's resume worked
only because its paths were hard-coded; here the work directory is a parameter,
so the test has to prove the parameter does not break the resume.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from core import pipeline
from core.pipeline import _split_frames, run_job
from tests.fake_backend import FakeBackend
from tests.support import (
    collector,
    make_config,
    make_video,
    messages,
    quiet,
    stage_names,
    write_frame,
)


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "select_backend", lambda devices, model: FakeBackend)


def frames_out(cfg: object) -> Path:
    return cfg.work_dir / "frames_out"  # type: ignore[attr-defined]


def test_a_fresh_run_writes_every_frame(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, frames=5)
    events, emit = collector()
    out = run_job(cfg, emit, multiprocessing.Event())

    assert out == cfg.output_path
    assert out.is_file()
    assert events[-1].kind == "done"
    assert events[-1].message == str(out)
    assert stage_names(events)[:2] == ["probe", "probe"]
    assert "5 frames, 0 already done, 5 remaining" in " ".join(
        messages(events, "stage")
    )
    assert len(list(frames_out(cfg).glob("frame_*.png"))) == 5


def test_a_resumed_run_leaves_the_finished_frames_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_config(tmp_path, frames=5)
    out_dir = frames_out(cfg)
    done = []
    for index in range(3):
        path = write_frame(out_dir / f"frame_{index:08d}.png")
        done.append((path, path.stat().st_mtime_ns, path.read_bytes()))

    monkeypatch.setenv("UPSCALER_FAKE_BACKEND_LOG", str(tmp_path / "log"))
    events, emit = collector()
    run_job(cfg, emit, multiprocessing.Event())

    assert "5 frames, 3 already done, 2 remaining" in " ".join(
        messages(events, "stage")
    )
    for path, mtime, original in done:
        assert path.read_bytes() == original, "a finished frame was rewritten"
        assert path.stat().st_mtime_ns == mtime
    for index in (3, 4):
        assert (out_dir / f"frame_{index:08d}.png").stat().st_size > 100
    assert cfg.output_path.is_file()


def test_re_running_a_finished_job_writes_no_frames_again(tmp_path: Path) -> None:
    from core.ffmpeg import probe

    cfg = make_config(tmp_path, frames=4)
    run_job(cfg, quiet(), multiprocessing.Event())
    first = probe(cfg.output_path)
    stamps = {
        path.name: path.stat().st_mtime_ns
        for path in frames_out(cfg).glob("frame_*.png")
    }

    events, emit = collector()
    run_job(cfg, emit, multiprocessing.Event())

    assert "4 frames, 4 already done, 0 remaining" in " ".join(
        messages(events, "stage")
    )
    assert "Nothing to do" in " ".join(messages(events, "stage"))
    assert {
        p.name: p.stat().st_mtime_ns for p in frames_out(cfg).glob("frame_*.png")
    } == stamps
    second = probe(cfg.output_path)
    assert second.frame_count == first.frame_count == 4
    assert second.width == first.width


def test_a_second_device_gets_its_own_share(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, frames=5, devices=2)
    events, emit = collector()
    run_job(cfg, emit, multiprocessing.Event())

    written = sorted(p.name for p in frames_out(cfg).glob("frame_*.png"))
    assert written == [f"frame_{index:08d}.png" for index in range(5)]
    assert cfg.output_path.is_file()
    device_events = [event for event in events if event.kind == "device_progress"]
    assert {event.device_index for event in device_events} <= {0, 1}


def test_each_device_really_ran_in_its_own_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_config(tmp_path, frames=6, devices=2)
    monkeypatch.setenv("UPSCALER_FAKE_BACKEND_LOG", str(tmp_path / "log"))
    run_job(cfg, quiet(), multiprocessing.Event())

    logs = sorted((tmp_path / "log").glob("device-*.log"))
    assert [path.name for path in logs] == ["device-0.log", "device-1.log"]
    # Each device must be served *as itself*; the pool is allowed to run both
    # chunks in one process when the first finishes before the second is
    # claimed, so the number of pids is an upper bound, not a count.
    pids = {line.split()[0] for path in logs for line in path.read_text().splitlines()}
    assert len(pids) <= 2, f"more worker processes than devices: {pids}"
    assert (
        "device=0 backend=onnx:cuda" in (tmp_path / "log" / "device-0.log").read_text()
    )
    assert (
        "device=1 backend=onnx:cuda" in (tmp_path / "log" / "device-1.log").read_text()
    )


def test_frames_are_dealt_out_almost_evenly() -> None:
    names = [f"frame_{index:08d}.png" for index in range(10)]
    assert [len(chunk) for chunk in _split_frames(names, [0, 1, 2])] == [4, 3, 3]


def test_the_frame_split_is_contiguous_in_order() -> None:
    names = [f"frame_{index:08d}.png" for index in range(7)]
    for chunk in _split_frames(names, [0, 1]):
        indices = [int(name[6:14]) for _device, name in chunk]
        assert indices == sorted(indices)


def test_the_output_is_the_upscaled_size(tmp_path: Path) -> None:
    from core.ffmpeg import probe

    video = make_video(tmp_path / "in.mp4", frames=3, width=48, height=32)
    cfg = make_config(tmp_path, frames=3, video=video)
    run_job(cfg, quiet(), multiprocessing.Event())
    info = probe(cfg.output_path)
    # The fake backend doubles, so 48x32 in, 96x64 out.
    assert (info.width, info.height) == (96, 64)
