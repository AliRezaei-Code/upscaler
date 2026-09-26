"""The worker, called directly.

`upscale_chunk` normally runs in a spawned child, where coverage cannot see it
and where a test cannot assert on a return value. Both are reasons to call it
in-process as well: the thread orchestration, the resume skip, the cancel check
and the error propagation are all real behaviour a mocked pool never exercises.
"""

from __future__ import annotations

import multiprocessing
import queue
import threading
import time
from pathlib import Path

import pytest

from core.config import Device
from core.worker import QUEUE_DEPTH, ChunkArgs, _worker_init, upscale_chunk
from tests.fake_backend import FakeBackend
from tests.support import UPSCALED_HEIGHT, UPSCALED_WIDTH, make_model, write_frame


class DiscardQueue:
    """A progress queue that throws everything away."""

    def put_nowait(self, _item: tuple[int, int, int, float]) -> None:
        return None


def make_device(index: int = 0) -> Device:
    return Device(
        index=index,
        name="Tesla P40",
        vendor="nvidia",
        backend="onnx:cuda",
        total_memory_bytes=0,
        pci_bus_id="0:01:00.0",
        compute_capability=None,
        usable=True,
        unusable_reason=None,
    )


def make_args(
    tmp_path: Path, *, frames: int = 4, device: Device | None = None
) -> ChunkArgs:
    frames_in = tmp_path / "frames_in"
    for index in range(frames):
        write_frame(frames_in / f"frame_{index:08d}.png", 48, 32)
    return ChunkArgs(
        backend_cls=FakeBackend,
        model_path=str(make_model(tmp_path / "model.pth")),
        frames_in=str(frames_in),
        frames_out=str(tmp_path / "frames_out"),
        items=tuple((0, f"frame_{index:08d}.png") for index in range(frames)),
        device=device or make_device(),
        export_size=(32, 48),
    )


def frames_written(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "frames_out").glob("frame_*.png"))


def test_a_chunk_writes_every_frame_it_is_given(tmp_path: Path) -> None:
    assert upscale_chunk(make_args(tmp_path, frames=4)) == 4
    assert [path.name for path in frames_written(tmp_path)] == [
        f"frame_{index:08d}.png" for index in range(4)
    ]


def test_the_frames_are_the_upscaled_ones(tmp_path: Path) -> None:
    import cv2

    upscale_chunk(make_args(tmp_path, frames=2))
    image = cv2.imread(str(tmp_path / "frames_out" / "frame_00000000.png"))
    assert image is not None
    assert (image.shape[1], image.shape[0]) == (UPSCALED_WIDTH, UPSCALED_HEIGHT)


def test_frames_already_written_are_skipped(tmp_path: Path) -> None:
    args = make_args(tmp_path, frames=4)
    frames_out = tmp_path / "frames_out"
    for index in (0, 1):
        write_frame(frames_out / f"frame_{index:08d}.png")
    stamps = {path.name: path.stat().st_mtime_ns for path in frames_written(tmp_path)}

    assert upscale_chunk(args) == 2
    for name, stamp in stamps.items():
        assert (frames_out / name).stat().st_mtime_ns == stamp, f"{name} was rewritten"


def test_a_chunk_with_nothing_to_do_returns_zero(tmp_path: Path) -> None:
    args = make_args(tmp_path, frames=2)
    assert upscale_chunk(args) == 2
    assert upscale_chunk(args) == 0


def test_a_cancel_event_stops_the_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPSCALER_FAKE_BACKEND_DELAY", "0.3")
    cancel = threading.Event()
    _worker_init(cancel, DiscardQueue())
    written: list[int] = []
    worker = threading.Thread(
        target=lambda: written.append(upscale_chunk(make_args(tmp_path, frames=20)))
    )
    worker.start()
    time.sleep(1.2)
    cancel.set()
    worker.join(timeout=60)

    assert not worker.is_alive(), "the chunk ignored the cancel event"
    assert 0 < len(frames_written(tmp_path)) < 20, "a cancelled chunk wrote all of them"
    assert written == [len(frames_written(tmp_path))], (
        "a cancelled chunk reported success"
    )


def test_a_missing_input_frame_is_reported(tmp_path: Path) -> None:
    _worker_init(threading.Event(), DiscardQueue())
    args = make_args(tmp_path, frames=2)
    (tmp_path / "frames_in" / "frame_00000001.png").unlink()
    with pytest.raises(OSError, match="could not read"):
        upscale_chunk(args)


def test_a_missing_model_is_reported(tmp_path: Path) -> None:
    _worker_init(threading.Event(), DiscardQueue())
    args = make_args(tmp_path, frames=1)
    object.__setattr__(args, "model_path", str(tmp_path / "absent.pth"))
    with pytest.raises(FileNotFoundError):
        upscale_chunk(args)


def test_a_backend_without_the_optional_keywords_still_works(tmp_path: Path) -> None:
    """The worker passes `export_size`/`tile_size` only to a backend that takes them."""

    class Minimal(FakeBackend):
        def __init__(self) -> None:
            super().__init__()

    _worker_init(threading.Event(), DiscardQueue())
    args = make_args(tmp_path, frames=1)
    object.__setattr__(args, "backend_cls", Minimal)
    assert upscale_chunk(args) == 1


def test_progress_is_reported_per_device(tmp_path: Path) -> None:
    reported: list[tuple[int, int, int, float]] = []

    class CollectingQueue:
        def put_nowait(self, item: tuple[int, int, int, float]) -> None:
            reported.append(item)

    _worker_init(threading.Event(), CollectingQueue())
    assert upscale_chunk(make_args(tmp_path, frames=250)) == 250
    # The last report is the exact total, not the last multiple of 100: a bar
    # that stops at 200 of 250 while the job finishes is a bug.
    assert [item[1] for item in reported] == [100, 200, 250]
    assert reported[-1][2] == 250
    assert {item[0] for item in reported} == {0}
    assert all(item[3] > 0 for item in reported)


def test_a_full_progress_queue_does_not_stop_the_chunk(tmp_path: Path) -> None:
    class FullQueue:
        def put_nowait(self, _item: tuple[int, int, int, float]) -> None:
            raise queue.Full

    _worker_init(threading.Event(), FullQueue())
    assert upscale_chunk(make_args(tmp_path, frames=250)) == 250


def test_the_queue_depth_is_small() -> None:
    """An unbounded hand-off queue would hold every 4K frame in memory."""
    assert QUEUE_DEPTH == 4


def test_a_spawn_event_is_accepted_as_a_cancel_handle(tmp_path: Path) -> None:
    """`run_job` hands the worker a real `multiprocessing.Event`, not a thread one."""
    event = multiprocessing.get_context("spawn").Event()
    _worker_init(event, DiscardQueue())
    assert upscale_chunk(make_args(tmp_path, frames=1)) == 1
    event.set()
