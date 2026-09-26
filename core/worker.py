"""The worker: read on one thread, infer on the main thread, write on a third.

This is the ported pipeline from `upscale_continue.py:74-128`, and the shape of
it is the point. Reading and writing a PNG is CPU work; inference is device
work. Doing them on one thread serialises the two, and overlapping them
measured ~38% faster on the original workload. So:

* the **reader** thread does every `cv2.imread`, and nothing else;
* the **main** thread does every `infer()` call — DirectML permits only one
  concurrent `Run` per session object, and the session is not thread-safe;
* the **writer** thread does every `cv2.imwrite`, and nothing else.

`queue.Queue(maxsize=4)` on both sides is the back-pressure: an unbounded queue
in front of a 30x upscaler would hold 4K PNGs in RAM for every frame in flight.

Frames travel as **strings**, not `Path` objects, because this module is the
thing that crosses a `spawn` boundary and a `Path` pickles fine but a payload of
one does not need the risk.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .backends.base import Backend
from .config import Device

#: Frames between progress reports. Also the interval at which the main thread
#: synchronises the device, because a progress number taken before the queue
#: drains is a number that has not happened yet.
REPORT_EVERY = 100

#: How many frames may sit in each hand-off queue. Small on purpose: see above.
QUEUE_DEPTH = 4


class CancelEvent(Protocol):
    """The only thing this application needs from a cancel handle.

    A `Protocol` rather than `multiprocessing.Event`, because that name is a
    factory on the default context rather than a class, and because a test
    handing in a `threading.Event` is then a first-class citizen instead of a
    cast. `multiprocessing.synchronize.Event` and `threading.Event` both satisfy
    it; `core.signals` sets the handle and the pipeline polls it, so both
    methods are part of the contract.
    """

    def is_set(self) -> bool: ...

    def set(self) -> None: ...


class ProgressQueue(Protocol):
    """The only thing this module needs from a progress handle."""

    def put_nowait(self, item: tuple[int, int, int, float]) -> None: ...


_CANCEL: CancelEvent | None = None
_PROGRESS_QUEUE: ProgressQueue | None = None


@dataclass(frozen=True)
class ChunkArgs:
    """One device's share of the job, as it crosses a `spawn` boundary.

    Attributes:
        backend_cls: The engine to construct. It is pickled **by reference**, so
            it has to be importable in the worker process — which is why the
            test double lives in an importable module rather than inside a test
            file.
        model_path: The checkpoint or `.onnx` this device serves.
        frames_in: Directory of extracted input frames.
        frames_out: Directory of upscaled frames, shared by every device.
        items: `(device_index, frame_name)` pairs, one per frame, in the order
            this device must process them. The device index is carried per item
            so a single worker can serve more than one device if a future
            scheduler wants that; today each process has exactly one.
        device: The device to load the model onto.
        precision: `"auto"`, `"fp16"` or `"fp32"`.
        tile_size: `0` lets the backend decide from the frame size.
        export_size: `(height, width)` of the frames, needed to export a `.pth`
            into an ONNX graph with concrete spatial dims.
        ordinal: This device's position in the selection, 0-based. It is what a
            progress row is keyed on, because `Device.index` is only unique
            within a vendor's device space: GPU 0 and CPU 0 are both 0, and
            reporting the index alone would draw the CPU's progress on the
            first GPU's row.
    """

    backend_cls: type[Backend]
    model_path: str
    frames_in: str
    frames_out: str
    items: tuple[tuple[int, str], ...]
    device: Device
    precision: str = "auto"
    tile_size: int = 0
    export_size: tuple[int, int] | None = None
    ordinal: int = -1


def _worker_init(cancel: CancelEvent, progress_q: ProgressQueue) -> None:
    """Receive the two objects a `spawn` child cannot be given any other way.

    A `threading.Event` cannot be pickled into `submit()`, and even a delivered
    copy is one the parent can never mutate. The pool's `initializer` runs in
    the child, so this is the only channel that works — and it is a shared
    handle, not a copy, which is what makes a stop reach an inference already in
    flight.
    """
    global _CANCEL, _PROGRESS_QUEUE
    _CANCEL = cancel
    _PROGRESS_QUEUE = progress_q


def _cancelled() -> bool:
    """Whether the job has been asked to stop."""
    return _CANCEL is not None and _CANCEL.is_set()


def _report(ordinal: int, processed: int, total: int, fps: float) -> None:
    """Publish one progress update, if anyone is listening."""
    if _PROGRESS_QUEUE is None:
        return
    # Progress is the one thing that may be lost: a full queue means the parent
    # is behind, and the next report carries the count anyway.
    with contextlib.suppress(queue.Full):
        _PROGRESS_QUEUE.put_nowait((ordinal, processed, total, fps))


def _synchronise_device(device: Device) -> None:
    """Block until the device has finished the work queued so far.

    Only CUDA is asynchronous in a way that matters here. `torch.cuda.synchronize`
    is reached through the torch module rather than imported, so a job on the
    CoreML, DirectML or CPU providers pays nothing for it.
    """
    if not device.backend.startswith("onnx:cuda"):
        return
    try:
        import torch

        torch.cuda.synchronize(device.index)
    except Exception:
        # Progress must not depend on a flush succeeding: a device that cannot
        # be synchronised is one the report is already covering.
        return


def _build_backend(args: ChunkArgs) -> Backend:
    """Construct and load this chunk's engine.

    `export_size` and `tile_size` are only accepted by the backends that have
    them; a backend written before this contract (or a test double) gets
    whatever keyword arguments it declares, so the two do not have to move in
    lockstep.
    """
    import inspect

    accepted = inspect.signature(args.backend_cls).parameters
    kwargs: dict[str, object] = {}
    if "export_size" in accepted:
        kwargs["export_size"] = args.export_size
    if "tile_size" in accepted:
        kwargs["tile_size"] = args.tile_size
    backend = args.backend_cls(**kwargs)
    backend.load(Path(args.model_path), args.precision, args.device)
    return backend


def _read_frame(path: Path) -> np.ndarray | None:
    """Read one frame as 8-bit 3-channel BGR.

    `IMREAD_COLOR` is the colour conversion: it normalises 16-bit, greyscale and
    paletted PNGs to 8-bit BGR, which is the only thing either engine accepts.
    """
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


def upscale_chunk(args: ChunkArgs) -> int:
    """Process one device's frames. Returns how many were written.

    Frames already present in `frames_out` are skipped, which is what makes a
    resumed run cheap: a 165,303-frame job that stopped at 160,000 writes 5,303
    frames, not 165,303.
    """
    import cv2

    frames_in = Path(args.frames_in)
    frames_out = Path(args.frames_out)
    frames_out.mkdir(parents=True, exist_ok=True)

    to_do = [
        (device_index, name)
        for device_index, name in args.items
        if not (frames_out / name).is_file()
    ]
    if not to_do:
        return 0

    read_q: queue.Queue[tuple[str, str] | None] = queue.Queue(maxsize=QUEUE_DEPTH)
    write_q: queue.Queue[tuple[str, np.ndarray] | None] = queue.Queue(
        maxsize=QUEUE_DEPTH
    )
    errors: list[BaseException] = []

    def _reader() -> None:
        try:
            for device_index, name in to_do:
                read_q.put((str(device_index), name))
            read_q.put(None)
        except BaseException as exc:  # re-raised on the main thread below
            errors.append(exc)
            read_q.put(None)

    def _writer() -> None:
        try:
            while True:
                item = write_q.get()
                if item is None:
                    return
                name, image = item
                if not cv2.imwrite(str(frames_out / name), image):
                    raise OSError(f"cv2.imwrite failed for {name}")
        except BaseException as exc:  # re-raised on the main thread below
            errors.append(exc)
            write_q.put(None)

    reader = threading.Thread(target=_reader, name="upscale-reader", daemon=True)
    writer = threading.Thread(target=_writer, name="upscale-writer", daemon=True)
    reader.start()
    writer.start()

    backend: Backend | None = None
    written = 0
    started = time.monotonic()
    try:
        backend = _build_backend(args)
        while True:
            if _cancelled():
                break
            if errors:
                break
            item = read_q.get()
            if item is None:
                break
            frame_name = item[1]
            image = _read_frame(frames_in / frame_name)
            if image is None:
                raise OSError(f"could not read {frames_in / frame_name}")
            upscaled = backend.infer_guarded(image)
            write_q.put((frame_name, upscaled))
            written += 1
            if written % REPORT_EVERY == 0:
                _synchronise_device(args.device)
                elapsed = max(1e-6, time.monotonic() - started)
                _report(args.ordinal, written, len(to_do), written / elapsed)
    finally:
        if written and written % REPORT_EVERY:
            # Without this the last report is `written - remainder`, so a
            # 250-frame chunk's progress bar stops at 200 and stays there while
            # the job finishes. The count is exact, so publish it.
            elapsed = max(1e-6, time.monotonic() - started)
            _report(args.ordinal, written, len(to_do), written / elapsed)
        write_q.put(None)
        writer.join(timeout=30)
        reader.join(timeout=30)
        if backend is not None:
            backend.close()
    if errors:
        raise errors[0]
    return written
