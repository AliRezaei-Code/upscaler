"""A work directory from a different video is refused, not merged.

The ported script could never hit this: its `WORK_DIR` was a constant, so the
directory and the video were always the same pair by construction. The moment a
work directory becomes a parameter — which is the whole point of this
application — a mistyped path or a second video lands in the same scratch space,
and the failure is silent: the output video contains frames from two different
sources, in the wrong order, with no error anywhere.

The guard is therefore a refusal, before any inference and before the encode.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from core import pipeline
from core.errors import UpscalerError
from core.pipeline import _existing_frames, run_job
from tests.fake_backend import FakeBackend
from tests.support import (
    collector,
    make_config,
    messages,
    quiet,
    stage_names,
    write_frame,
)


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "select_backend", lambda devices, model: FakeBackend)


def test_frames_from_another_video_are_refused(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, frames=4)
    frames_out = cfg.work_dir / "frames_out"
    frames_out.mkdir(parents=True)
    # A 10-frame video's leftovers: frame_00000004.png is not this video's.
    for index in (0, 1, 4, 9):
        (frames_out / f"frame_{index:08d}.png").write_bytes(b"from elsewhere")

    events, emit = collector()
    with pytest.raises(UpscalerError) as excinfo:
        run_job(cfg, emit, multiprocessing.Event())

    message = str(excinfo.value)
    assert message == (
        f"Work directory {cfg.work_dir} contains 2 frames that do not match this "
        "video. Move or delete it before starting."
    )
    assert "upscale" not in stage_names(events), "the guard let inference start"
    assert not cfg.output_path.exists()


def test_a_refusal_names_the_directory_and_leaves_the_frames_alone(
    tmp_path: Path,
) -> None:
    cfg = make_config(tmp_path, frames=3)
    frames_out = cfg.work_dir / "frames_out"
    frames_out.mkdir(parents=True)
    stray = frames_out / "frame_00000007.png"
    stray.write_bytes(b"not mine")

    with pytest.raises(UpscalerError, match=str(cfg.work_dir)):
        run_job(cfg, quiet(), multiprocessing.Event())

    assert stray.read_bytes() == b"not mine", "the guard deleted the user's data"
    assert _existing_frames(frames_out) == {"frame_00000007.png"}


def test_frames_beyond_the_end_of_the_video_count_as_foreign(tmp_path: Path) -> None:
    """A five-frame video must not silently accept a six-frame directory."""
    cfg = make_config(tmp_path, frames=5)
    frames_out = cfg.work_dir / "frames_out"
    frames_out.mkdir(parents=True)
    for index in range(6):
        (frames_out / f"frame_{index:08d}.png").write_bytes(b"x")

    with pytest.raises(UpscalerError, match="1 frames that do not match"):
        run_job(cfg, quiet(), multiprocessing.Event())


def test_a_directory_of_this_video_is_accepted(tmp_path: Path) -> None:
    """The guard must not fire on a legitimate partial run."""
    cfg = make_config(tmp_path, frames=5)
    frames_out = cfg.work_dir / "frames_out"
    for index in (0, 1):
        write_frame(frames_out / f"frame_{index:08d}.png")

    events, emit = collector()
    run_job(cfg, emit, multiprocessing.Event())

    assert "5 frames, 2 already done, 3 remaining" in " ".join(
        messages(events, "stage")
    )
    assert cfg.output_path.is_file()


def test_unrelated_files_in_the_work_directory_are_not_frames(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, frames=2)
    frames_out = cfg.work_dir / "frames_out"
    frames_out.mkdir(parents=True)
    (frames_out / "notes.txt").write_text("a user left this here", encoding="utf-8")
    write_frame(frames_out / "frame_00000000.png")

    run_job(cfg, quiet(), multiprocessing.Event())
    assert (frames_out / "notes.txt").read_text(
        encoding="utf-8"
    ) == "a user left this here"
    assert (frames_out / "frame_00000001.png").is_file()
