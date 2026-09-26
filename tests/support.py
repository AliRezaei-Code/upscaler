"""Shared fixtures for the pipeline tests: a real video, model file and fake engine.

The pipeline's whole job is to get real frames through real processes into a
real file, so these tests use ffmpeg-produced video and a spawn pool. What is
faked is only the upscaler, and it is faked in `tests/fake_backend.py` because a
class has to be importable to cross a `spawn` boundary.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from core.config import Device, JobConfig
from core.errors import UpscalerError
from core.events import PipelineEvent
from core.models import MIN_MODEL_BYTES


def ffmpeg_or_skip() -> str:
    """The ffmpeg to build fixtures with, or skip the test."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    pytest.skip("no ffmpeg on PATH; the pipeline tests need real video")


def make_video(
    path: Path,
    *,
    frames: int = 5,
    width: int = 48,
    height: int = 32,
    fps: int = 10,
    audio: bool = False,
) -> Path:
    """A real, decodable video of `frames` distinct frames."""
    ffmpeg = ffmpeg_or_skip()
    duration = frames / fps
    command = [
        ffmpeg,
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
    ]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    # yuv24p does not exist; yuv444p is the closest real format and it
    # decodes to exactly the frames this test expects.
    command += ["-pix_fmt", "yuv444p", "-c:v", "libx264", "-c:a", "aac", str(path)]
    subprocess.run(command, check=True, capture_output=True)
    return path


#: The size the fake backend writes: its nearest-neighbour 2x of a 48x32 clip.
UPSCALED_WIDTH = 96
UPSCALED_HEIGHT = 64


def write_frame(
    path: Path, width: int = UPSCALED_WIDTH, height: int = UPSCALED_HEIGHT
) -> Path:
    """A real, decodable PNG of the given size.

    A resume test cannot seed the work directory with a placeholder: the encode
    checks the first frame's PNG header to refuse a directory of mixed
    resolutions, which is exactly the failure a half-finished job can leave.
    """
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    gradient = np.linspace(0, 255, width, dtype=np.uint8)[None, :, None]
    image = np.tile(gradient, (height, 1, 3))
    assert cv2.imwrite(str(path), image)
    return path


def make_model(path: Path) -> Path:
    """A file big enough to pass `verify_model`'s size check.

    Not a real checkpoint: the pipeline never opens it, because the backend is
    faked, and a 2 MB write per test is cheaper than loading a 67 MB model.
    """
    path.write_bytes(b"fake model weights" * 60_000)
    assert path.stat().st_size > MIN_MODEL_BYTES
    return path


def make_device(index: int = 0, name: str = "Tesla P40") -> Device:
    """A device the pipeline will spread frames across."""
    return Device(
        index=index,
        name=name,
        vendor="nvidia",
        backend="onnx:cuda",
        total_memory_bytes=0,
        pci_bus_id=f"0:0{index + 1}:00.0",
        compute_capability=None,
        usable=True,
        unusable_reason=None,
    )


def make_config(
    tmp_path: Path,
    *,
    frames: int = 5,
    devices: int = 1,
    audio: bool = False,
    crf: int = 30,
    video: Path | None = None,
) -> JobConfig:
    """A `JobConfig` wired to `tmp_path`, ready for `run_job`."""
    source = (
        video
        if video is not None
        else make_video(tmp_path / "in.mp4", frames=frames, audio=audio)
    )
    return JobConfig(
        input_path=source,
        output_path=tmp_path / "out.mp4",
        model_path=make_model(tmp_path / "model.pth"),
        work_dir=tmp_path / "work",
        devices=tuple(make_device(index) for index in range(devices)),
        crf=crf,
    )


def quiet() -> Callable[[PipelineEvent], None]:
    """An `emit` that throws the events away, for tests that assert on files."""

    def _emit(_event: PipelineEvent) -> None:
        return None

    return _emit


def collector() -> tuple[list[PipelineEvent], Iterator[PipelineEvent]]:
    """An `emit` that records everything, for asserting on the log."""
    events: list[PipelineEvent] = []
    return events, events.append


def messages(events: list[PipelineEvent], kind: str) -> list[str]:
    return [event.message for event in events if event.kind == kind]


def stage_names(events: list[PipelineEvent]) -> list[str]:
    return [event.stage for event in events if event.kind == "stage"]


def error_message(events: list[PipelineEvent]) -> str:
    for event in events:
        if event.kind == "error":
            return event.message
    raise AssertionError(f"no error event in {[(e.kind, e.message) for e in events]}")


__all__ = [
    "UpscalerError",
    "collector",
    "error_message",
    "ffmpeg_or_skip",
    "make_config",
    "make_device",
    "make_model",
    "make_video",
    "messages",
    "quiet",
    "stage_names",
    "write_frame",
]
