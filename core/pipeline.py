"""The job: probe, preflight, extract, resume, spread across devices, check, encode.

Eleven steps, in the order below, each of which exists because skipping it has a
known cost:

1. **Validate, then verify the model.** A 9-byte `Not Found` file saved as
   `RealESRGAN_x4plus_anime_6B.pth` is a real artefact on the machine this was
   built on, and spandrel's response to it is a traceback about tensors. This is
   the only call site that stops it.
2. **Probe the input** for dimensions, frame count and whether it has audio.
3. **Disk preflight**, against a *calibrated* estimate. The naive
   `n * w * h * 4 * scale**2` uncompressed-RGB formula over-estimates by 12-27x
   and would reject this plan's own 165,303-frame job with "Need ~9750 GB".
4. **Extract frames**, skipping the pass when the directory already holds the
   right number.
5. **Resume set** from `frames_out`, with a staleness guard: a work directory
   holding frames from a *different* video is refused, which the hard-coded
   `WORK_DIR` of the ported script could never hit and a generalised tool hits
   on the first typo.
6. **Round-robin** the remaining frames across the selected devices.
7. **One spawned process per device**, with the cancel event passed through the
   pool's `initializer` — the only channel that crosses a `spawn` boundary.
8. **Bounded teardown** on completion or cancel, using the public API. The
   ported script reached into `executor._processes`, which is populated
   asynchronously and mutated by another thread, so iterating it from a signal
   handler raises `RuntimeError: dictionary changed size during iteration`.
9. **Completeness check before encode.** ffmpeg's `image2` demuxer stops at the
   first gap, so a sparse directory silently produces a short video.
10. **Encode with audio**, atomically published.
11. **Emit `done`** with the output path.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .backends.registry import select_backend
from .config import JobConfig
from .errors import UpscalerError
from .events import PipelineEvent
from .models import entry_for, verify_model
from .worker import CancelEvent, ChunkArgs, _worker_init

#: Bytes per **output** pixel of an upscaled PNG frame. Measured, not derived:
#: 820 GiB of `frames_out` across 165,303 frames of 384x288 at 4x is 5.33 MiB
#: per frame, and an output frame is 1536x1152 = 1,769,472 pixels — 3.01 bytes
#: per pixel, which is 6.8x the uncompressed RGBA size. A 4x upscale makes
#: frames *larger*, not smaller: the pixel-shuffle output of a noisy source is
#: close to incompressible for PNG. The plan's 0.30 assumed compression helps,
#: and would have under-estimated the reference run by 10x — the dangerous
#: direction for a preflight.
BYTES_PER_OUTPUT_PIXEL = 3.0
#: Bytes per input pixel of an extracted frame: 26 GiB across the same 165,303
#: frames of 384x288 is 168,878 bytes per frame, or 1.53 bytes per pixel — a
#: third of uncompressed RGB, which is the part compression does win.
BYTES_PER_INPUT_PIXEL = 1.5
#: The encode's temporary file, as a fraction of the output frame bytes. A
#: libx264 CRF encode lands well under the sum of its inputs; 10% is an
#: allowance, not a measurement, and the real run's 846 GiB did not include a
#: temp file at all.
ENCODE_TEMP_FRACTION = 0.10
#: Multiply the estimate by this before comparing against free space, so a job
#: that exactly fits does not start and then fail on its last frame.
DISK_HEADROOM = 1.05

FRAME_PREFIX = "frame_"
FRAME_SUFFIX = ".png"
FRAMES_IN_DIR = "frames_in"
FRAMES_OUT_DIR = "frames_out"

#: How long teardown waits for a pool worker before terminating it.
TEARDOWN_GRACE_SECONDS = 20.0
#: And then how long `terminate()` gets before the child is given up on.
TERMINATE_JOIN_SECONDS = 5.0
#: How often the parent drains progress while the pool runs.
PROGRESS_POLL_SECONDS = 0.5


def frame_name(index: int) -> str:
    """The one naming convention `frame_%08d.png`, in one place."""
    return f"{FRAME_PREFIX}{index:08d}{FRAME_SUFFIX}"


def estimate_frame_bytes(n_frames: int, width: int, height: int, scale: int) -> int:
    """Disk an upscale of this video will need, in bytes.

    Calibrated against one real 165,303-frame run (846 GiB at 384x288, 4x)
    rather than derived from first principles. Both directions of the naive
    derivation are wrong: `n*w*h*4*scale**2` of uncompressed RGBA is 1.17 TB
    here, which is within 15% of the truth by luck rather than by reasoning,
    while assuming compression (0.30 bytes per output pixel) is 10x low.
    """
    if n_frames < 0 or width < 1 or height < 1 or scale < 1:
        raise ValueError(
            f"nonsensical estimate for {n_frames} frames of "
            f"{width}x{height} at {scale}x"
        )
    output_pixels = n_frames * width * scale * height * scale
    input_pixels = n_frames * width * height
    return int(
        output_pixels * BYTES_PER_OUTPUT_PIXEL
        + input_pixels * BYTES_PER_INPUT_PIXEL
        + output_pixels * 4 * ENCODE_TEMP_FRACTION
    )


def resolve_scale(cfg: JobConfig) -> tuple[int, str]:
    """The output multiplier to size the preflight with, and where it came from.

    `JobConfig.scale` is 0 for "read it from the model", and the preflight runs
    before any model is loaded — so without this the estimate would be computed
    for a 1x job and would be **16x short for a 4x model**, which is the
    dangerous direction: it says yes, and then fills the disk. The chain is
    ordered by cost: what the user typed, then the catalogue, then the ONNX
    graph's own shapes (a header read, no torch), and only then a stated
    assumption.
    """
    if cfg.scale > 0:
        return cfg.scale, "the settings"
    entry = entry_for(cfg.model_path)
    if entry is not None:
        return entry.scale, f"the catalogue entry for {entry.filename}"
    if cfg.model_path.suffix.lower() == ".onnx":
        from .backends.export import scale_from_graph, scale_from_metadata

        learned = scale_from_graph(cfg.model_path) or scale_from_metadata(
            cfg.model_path
        )
        if learned:
            return learned, f"the shapes in {cfg.model_path.name}"
    return 1, "an assumption of 1x, which may under-estimate a 4x model"


def encode_tmp_path(output_path: Path) -> Path:
    """Where the encode writes before it is published.

    Always `<output>.part.mp4` **in the output's own directory**, so the
    publishing `os.replace` cannot fail with `EXDEV` across a filesystem
    boundary, and so a failed encode never leaves a temporary next to the work
    directory 800 GB away.
    """
    return output_path.with_suffix(output_path.suffix + ".part.mp4")


def _emit(emit: Callable[[PipelineEvent], None], event: PipelineEvent) -> None:
    emit(event)


def _gib(value: float) -> str:
    """A byte count as GB or GiB, whichever is not a joke at this size.

    `Need ~0 GB, 42 GB free` is what a 60-frame test clip prints with one
    decimal place, and a preflight that says "0 GB" reads as though it did
    nothing.
    """
    if value < 1000**3:
        return f"{value / 1000**2:.0f} MB"
    return f"{value / 1000**3:.1f} GB"


def _stage(emit: Callable[[PipelineEvent], None], stage: str, message: str) -> None:
    """Announce a pipeline step, with the detail the log line should carry."""
    _emit(
        emit,
        PipelineEvent(kind="stage", task_id="job", stage=stage, message=message),
    )


def _split_frames(
    names: Sequence[str], devices: Sequence[Any]
) -> list[list[tuple[int, str]]]:
    """Deal the frames out round-robin, one device per worker.

    The split is by name across the whole pending set, so three devices on a
    55,000-frame video differ by at most one frame — the load is balanced by
    construction rather than by measurement.
    """
    chunks: list[list[tuple[int, str]]] = [[] for _ in devices]
    for position, name in enumerate(names):
        chunks[position % len(devices)].append((position % len(devices), name))
    return chunks


def _existing_frames(directory: Path) -> set[str]:
    return {path.name for path in directory.glob(f"{FRAME_PREFIX}*{FRAME_SUFFIX}")}


def _teardown(pool: ProcessPoolExecutor) -> None:
    """Shut a pool down without waiting for an unbounded join.

    `shutdown(wait=False, cancel_futures=True)` is the public API and is enough
    for the common case. A worker that is mid-frame can still take seconds, so
    the remainder gets a bounded grace period and is then terminated. Nothing
    here reaches into a private attribute, and nothing is called from a signal
    handler.
    """
    pool.shutdown(wait=False, cancel_futures=True)
    deadline = time.monotonic() + TEARDOWN_GRACE_SECONDS
    while time.monotonic() < deadline and multiprocessing.active_children():
        time.sleep(0.1)
    for child in multiprocessing.active_children():
        child.terminate()
        child.join(timeout=TERMINATE_JOIN_SECONDS)


def _drain_progress(
    progress_q: Any, emit: Callable[[PipelineEvent], None]
) -> dict[int, int]:
    """Move whatever progress has arrived into events. Returns counts by ordinal.

    The queue carries the *ordinal* of the selected device, not its
    `Device.index`, because the two are different numbers: `index` is unique
    only within a vendor's device space, so GPU 0 and CPU 0 are both 0, and a
    front-end keying its rows on the index draws the CPU's progress on the first
    GPU's row.
    """
    processed: dict[int, int] = {}
    while True:
        try:
            ordinal, done, total, fps = progress_q.get_nowait()
        except Exception:
            # `queue.Empty` and a closed queue both mean the same thing here.
            break
        processed[ordinal] = done
        _emit(
            emit,
            PipelineEvent(
                kind="device_progress",
                task_id="job",
                device_ordinal=ordinal,
                processed=done,
                total=total,
                fps=fps,
            ),
        )
    return processed


def run_job(
    cfg: JobConfig,
    emit: Callable[[PipelineEvent], None],
    cancel: CancelEvent,
) -> Path:
    """Run one job to completion. Returns the output path.

    The emitted events are the report: on the two early failure paths below the
    return value is the path that was *not* written, so a caller must read the
    `error` event (which `UpscalerApp` also records) rather than trusting the
    return. On every other path the return is a file that exists.
    """
    from .ffmpeg import count_frames, encode_video, extract_frames, probe

    try:
        cfg.validate()
    except UpscalerError as exc:
        _emit(emit, PipelineEvent(kind="error", task_id="job", message=str(exc)))
        return cfg.output_path

    try:
        verify_model(cfg.model_path, entry_for(cfg.model_path))
    except UpscalerError as exc:
        _emit(emit, PipelineEvent(kind="error", task_id="job", message=str(exc)))
        return cfg.output_path

    backend_cls = select_backend(cfg.devices, cfg.model_path)

    _stage(emit, "probe", f"Reading {cfg.input_path.name}")
    info = probe(cfg.input_path)
    scale, scale_source = resolve_scale(cfg)
    if scale_source.startswith("an assumption"):
        _emit(emit, PipelineEvent(kind="log", task_id="job", message=scale_source))
    _stage(
        emit,
        "probe",
        f"{info.width}x{info.height}, {info.frame_count} frames at {info.fps:.3f} fps, "
        f"audio {'yes' if info.has_audio else 'no'}, {scale}x from {scale_source}",
    )

    frames_in = cfg.work_dir / FRAMES_IN_DIR
    frames_out = cfg.work_dir / FRAMES_OUT_DIR
    # The directory is created before the disk is measured: on a brand-new input
    # `work_dir` does not exist yet, and `shutil.disk_usage` raises
    # FileNotFoundError rather than answering the question.
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    estimate = estimate_frame_bytes(info.frame_count, info.width, info.height, scale)
    free = shutil.disk_usage(cfg.work_dir).free
    if free < estimate * DISK_HEADROOM:
        raise UpscalerError(
            f"Need ~{_gib(estimate)} for frames, {_gib(free)} free at "
            f"{cfg.work_dir}. Choose another work directory."
        )
    _stage(
        emit,
        "preflight",
        f"Need ~{_gib(estimate)}, {_gib(free)} free at {cfg.work_dir}",
    )

    if count_frames(frames_in) == info.frame_count and info.frame_count > 0:
        _stage(
            emit,
            "extract",
            f"{info.frame_count} frames already extracted in {frames_in}",
        )
    else:
        _stage(emit, "extract", f"Extracting {info.frame_count} frames to {frames_in}")
        extract_frames(cfg.input_path, frames_in)

    all_names = [frame_name(index) for index in range(info.frame_count)]
    expected = set(all_names)
    existing = _existing_frames(frames_out)
    foreign = existing - expected
    if foreign:
        raise UpscalerError(
            f"Work directory {cfg.work_dir} contains {len(foreign)} frames that do "
            "not match this video. Move or delete it before starting."
        )
    pending = [name for name in all_names if name not in existing]
    _stage(
        emit,
        "upscale",
        f"{len(all_names)} frames, {len(existing)} already done, "
        f"{len(pending)} remaining",
    )

    if not pending:
        _stage(emit, "upscale", "Nothing to do; encoding what is already there")
    else:
        _run_devices(
            cfg,
            emit,
            cancel,
            backend_cls=backend_cls,
            frames_in=frames_in,
            frames_out=frames_out,
            pending=pending,
            frame_size=(info.height, info.width),
        )

    present = _existing_frames(frames_out)
    missing = expected - present
    if missing:
        raise UpscalerError(
            f"{len(missing)} frames missing from {frames_out}; the output would be "
            "truncated. Re-run to resume."
        )

    fps = cfg.fps or info.fps
    audio_source = cfg.input_path if info.has_audio else None
    _stage(
        emit,
        "encode",
        f"Encoding {len(expected)} frames at {fps:.3f} fps, CRF {cfg.crf}",
    )
    tmp_path = encode_tmp_path(cfg.output_path)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        frames_out,
        tmp_path,
        cfg.output_path,
        fps,
        cfg.crf,
        audio_source,
        log=lambda message: _emit(
            emit, PipelineEvent(kind="log", task_id="job", message=message)
        ),
    )

    if cfg.delete_frames_after_encode:
        _stage(emit, "clean", f"Removing {frames_out}")
        shutil.rmtree(frames_out, ignore_errors=True)

    _emit(
        emit,
        PipelineEvent(
            kind="done", task_id="job", message=str(cfg.output_path), stage="done"
        ),
    )
    return cfg.output_path


def _run_devices(
    cfg: JobConfig,
    emit: Callable[[PipelineEvent], None],
    cancel: CancelEvent,
    *,
    backend_cls: type[Any],
    frames_in: Path,
    frames_out: Path,
    pending: list[str],
    frame_size: tuple[int, int],
) -> None:
    """Fan the pending frames out over one process per selected device.

    `frame_size` goes to the worker because a `.pth` has to be exported before
    it can be sessioned, and an export is specific to one frame size.
    """
    """Fan the pending frames out over one process per selected device."""
    devices = list(cfg.devices)
    chunks = _split_frames(pending, devices)
    ctx = multiprocessing.get_context("spawn")
    cancel_event = ctx.Event()
    progress_q = ctx.Queue()
    args = [
        ChunkArgs(
            backend_cls=backend_cls,
            model_path=str(cfg.model_path),
            frames_in=str(frames_in),
            frames_out=str(frames_out),
            items=tuple(chunk),
            device=device,
            precision=cfg.precision,
            tile_size=cfg.tile_size,
            export_size=frame_size,
            ordinal=ordinal,
        )
        for ordinal, (device, chunk) in enumerate(zip(devices, chunks, strict=True))
    ]
    # A stop requested before the pool started still has to stop this one.
    if cancel.is_set():
        cancel_event.set()

    failures: list[BaseException] = []
    pool = ProcessPoolExecutor(
        max_workers=len(devices),
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(cancel_event, progress_q),
    )
    try:
        futures = [pool.submit(_run_one, arg) for arg in args]
        deadline_count = 0
        while any(not future.done() for future in futures):
            _drain_progress(progress_q, emit)
            if cancel.is_set() and not cancel_event.is_set():
                cancel_event.set()
            time.sleep(PROGRESS_POLL_SECONDS)
            deadline_count += 1
            if deadline_count * PROGRESS_POLL_SECONDS > TEARDOWN_GRACE_SECONDS * 30:
                # Fifteen minutes with no completion: the workers are wedged,
                # not slow. Stop waiting rather than hang the UI forever.
                cancel_event.set()
                deadline_count = 0
        _drain_progress(progress_q, emit)
        for future in futures:
            failure = future.exception()
            if failure is not None:
                failures.append(failure)
    finally:
        _teardown(pool)
    if cancel.is_set():
        unwritten = [name for name in pending if not (frames_out / name).is_file()]
        raise UpscalerError(
            f"Stopped with {len(unwritten)} of {len(pending)} frames still to write. "
            "Re-run to resume."
        )
    if failures:
        raise UpscalerError(str(failures[0]))


def _run_one(args: ChunkArgs) -> int:
    """Indirection so the pool pickles a module-level function, not a closure."""
    from .worker import upscale_chunk

    return upscale_chunk(args)


def work_dir_size(path: Path) -> int:
    """Bytes a work directory currently occupies, for the clean-up confirmation."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total
