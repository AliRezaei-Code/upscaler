#!/usr/bin/env python3
"""The maintainer's GPU gate: does this machine really run the model on its GPU?

Exit codes: 0 the gate passed, 1 it did not, 2 the invocation was wrong.

Why this script exists
----------------------
ONNX Runtime does not fail when it cannot use the provider it was asked for. It
logs a warning, runs the graph on the CPU, and returns a perfectly correct
answer — indistinguishable from success unless somebody looks at
``session.get_providers()``, and looking is exactly what nobody does during a
smoke test. So the gate asserts the provider it asked for is the provider the
session is using, and treats a silent fallback as a failure.

A synthetic ``conv2d`` is not enough, and this is the reason the whole script
exists. A synthetic graph passes in precisely the situation the gate is meant to
catch: the CUDA execution provider fell back to the CPU — it always does when
``libcublas``/``libcudnn`` cannot be loaded — while PyTorch, which is a
different library with its own CUDA loader, still works. A green synthetic test
and a 3x slower job are indistinguishable from the outside. So the graph under
test is the **real** one: ``RealESRGAN_x4plus.pth``, 67 MB of RRDBNet, exported
to ONNX by the app's own exporter, run through the app's own ``session_for`` and
``TorchBackend``.

Why a CPU reference is the only honest check
--------------------------------------------
There are three ways to judge a GPU result and two of them are worthless. An
expected value: nobody has one for a 4x super-resolution model, and nobody will
write a 165,000-frame golden file. A visual check: a GPU that silently computes
a subtly wrong convolution produces a video that looks *fine*. The third is a
comparison against the same computation on the CPU, in the same library, on the
same bytes — which is the definition of correct for an accelerator, because a
driver or runtime that returns a *different* answer is broken even when the
number is plausible. Hence ``--tolerance``, ``2e-3`` by default.

The comparison is on the **float** tensor, before the cast to ``uint8``. One
least-significant bit of a pixel is 1/255 = 3.9e-3, so a tolerance of 2e-3
applied to pixels would be measuring quantisation rather than the engine. The
pixel-level difference is printed as well, as a number of least-significant
bits, because that is the number a user would eventually see.

Both tensors are clamped to ``[0, 1]`` before they are differenced, because
that is the range in which a difference can reach a pixel at all — the app
clips before it writes — and because spandrel clamps the model's output while
the exported ONNX graph does not, a clamp not being an op the exporter emits.
On the real checkpoint at 384x288 that distinction is the whole measurement:
1.4e-6 clamped against 7.0e-2 unclamped, every large difference being
overshoot where ONNX Runtime says 1.0705 and torch says 1.0000.

A gate that cannot fail is worse than no gate, so three ways to pass without
testing anything are refused rather than tolerated:

* no accelerator at all — an all-CPU run is a failure, not a pass;
* an accelerator the probe already declared unreachable, which is reported by
  name instead of being skipped;
* a checkpoint that is missing, truncated, or not the one the catalogue
  describes. Any of those stops the run, and the message carries the URL and
  the SHA-256 the catalogue claims, so the claim is checked here rather than
  assumed. This script is where the catalogue's hashes earn their keep.

The whole run is bounded by a total timeout (``--timeout``, 600 s by default,
``0`` disables it) enforced by a parent process, because a wedged driver hangs
inside a C call that no Python-level deadline can interrupt; when it expires the
stage that was running is reported. Raise it if the first run has to pay for the
one-off export of the real checkpoint, which takes minutes.

    python scripts/verify_gpu.py                       # every accelerator
    python scripts/verify_gpu.py --model x4.safetensors # a different checkpoint
    python scripts/verify_gpu.py --timeout 0            # no watchdog at all
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # The gate is run by hand, on maintainers' machines, where the editable
    # install is frequently stale or was made for another interpreter. Failing
    # with a bare ImportError for that reason would waste more time than these
    # four lines ever could.
    sys.path.insert(0, str(_REPO_ROOT))

from core.backends.export import export_cache_path, export_to_onnx
from core.config import Device, format_memory
from core.devices import cpu_device, probe_all
from core.errors import UpscalerError
from core.models import cached_hash, entry_for, verify_model
from core.paths import models_dir

# `core.backends.onnx_backend` and `core.backends.torch_backend` are imported
# inside the functions that use them: they pull in onnxruntime and torch, which
# between them are ten seconds of import time, and `--help` must not pay it.

_SELF = Path(__file__).resolve()

#: The checkpoint the catalogue calls the default choice, and the one whose
#: architecture exercises the ops the CoreML and DirectML tables have to support.
DEFAULT_MODEL_NAME = "RealESRGAN_x4plus.pth"
#: FP32 max-absolute-difference tolerance against the CPU reference. 2e-3 on a
#: [0, 1] tensor is about half a pixel level; FP32 kernels on different hardware
#: reorder their reductions, so the exact value is not zero, and anything above
#: this means the two runs are not computing the same function.
DEFAULT_TOLERANCE = 2e-3
DEFAULT_TIMEOUT = 600.0
#: 288x384 — the frame size the export is tested at, and small enough that the
#: 67 MB model still fits on a 4 GB card.
DEFAULT_WIDTH = 384
DEFAULT_HEIGHT = 288


class Stages:
    """The name of the stage in progress, published for the parent process.

    The timeout is enforced by a parent because a wedged driver hangs inside a C
    call that no Python-level deadline can interrupt, and the parent has to be
    able to say *where* the child stopped. The name is replaced atomically so a
    half-written name is never read back.
    """

    def __init__(self, path: Path | None) -> None:
        """`path` is the file the parent polls, or `None` in-process."""
        self._path = path
        self.current = "starting"

    def begin(self, name: str) -> None:
        """Announce a stage on stdout and publish it to the parent."""
        self.current = name
        print(f"--- stage: {name}", flush=True)
        if self._path is None:
            return
        staging = self._path.with_suffix(".part")
        staging.write_text(name, encoding="utf-8")
        os.replace(staging, self._path)


def _read_stage(path: Path) -> str:
    """The child's current stage, or a truthful unknown."""
    try:
        return path.read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "starting up"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    # Plain HelpFormatter, not ArgumentDefaultsHelpFormatter: the defaults are
    # written into each help string instead, because one of them ("the model in
    # the data directory") is only resolvable at run time and the formatter
    # would print a bare "(default: None)" for it.
    parser = argparse.ArgumentParser(
        prog="verify_gpu.py",
        description=(
            "Run the real exported model on every accelerator and compare it "
            "against a CPU reference. Exits non-zero unless every one of them "
            "is reached and agrees."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "the checkpoint to verify; a .pth is exported and run through both "
            "engines, a .onnx is run through ONNX Runtime only. Default: "
            f"<data dir>/models/{DEFAULT_MODEL_NAME}"
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=(
            "max absolute difference from the CPU reference, FP32, on a [0, 1] "
            f"tensor. Default: {DEFAULT_TOLERANCE:g}"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=(
            "total seconds for the whole gate, enforced by a parent process; 0 "
            f"runs it in-process, unwatched. Default: {DEFAULT_TIMEOUT:g}"
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"frame width to export for. Default: {DEFAULT_WIDTH}",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"frame height to export for. Default: {DEFAULT_HEIGHT}",
    )
    parser.add_argument(
        "--stage-file",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,  # set by the parent, never by a person
    )
    args = parser.parse_args(argv)
    if args.width < 1 or args.height < 1:
        parser.error(
            f"--width and --height must be positive, got {args.width}x{args.height}"
        )
    if args.tolerance < 0:
        parser.error(f"--tolerance must not be negative, got {args.tolerance}")
    if args.timeout < 0:
        parser.error(f"--timeout must not be negative, got {args.timeout}")
    return args


def _frame(height: int, width: int) -> np.ndarray:
    """A deterministic BGR uint8 frame, identical on every run and machine.

    Fixed-seed noise over a gradient: a smooth frame would let a broken
    convolution pass by keeping nearly every pixel in range, and a random frame
    without structure would make the comparison noisier than the tolerance.
    """
    rng = np.random.default_rng(0)
    rows = np.linspace(0.0, 255.0, height, dtype=np.float32)[:, None]
    columns = np.linspace(0.0, 255.0, width, dtype=np.float32)[None, :]
    grey = np.clip(
        (rows + columns) / 2.0 + rng.normal(0.0, 12.0, (height, width)), 0, 255
    )
    frame = np.stack(
        [grey, np.roll(grey, 7, axis=1), np.roll(grey, 13, axis=0)], axis=-1
    )
    return np.ascontiguousarray(frame.astype(np.uint8))


def _tensor_for(frame: np.ndarray) -> np.ndarray:
    """BGR uint8 frame to the float NCHW tensor both engines are fed.

    The same conversion ``OnnxBackend._infer_full`` and ``TorchBackend._infer_full``
    perform. It has to be identical, or the comparison would be measuring the
    preprocessing rather than the accelerator.
    """
    rgb = np.ascontiguousarray(frame[:, :, ::-1].transpose(2, 0, 1)[None])
    return np.ascontiguousarray(rgb, dtype=np.float32) / np.float32(255.0)


def _onnx_inference(
    model_path: Path, device: Device, frame: np.ndarray
) -> tuple[np.ndarray, list[str], list[str]]:
    """One real ONNX Runtime inference; the output tensor, and both provider lists.

    `model_path` is the app's own export, and the session is the app's own
    `session_for`, so the DML session options, the CUDA `preload_dlls` and the
    CoreML format retry are all in play — the gate verifies the code that ships,
    not a hand-built session.
    """
    from core.backends.onnx_backend import providers_for, session_for

    session = session_for(model_path, device)
    in_use = list(session.get_providers())
    requested = providers_for(device)
    output = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: _tensor_for(frame)},
    )[0]
    return np.asarray(output, dtype=np.float32), in_use, requested


def _torch_inference(
    model_path: Path, device: Device, frame: np.ndarray
) -> tuple[np.ndarray, str, str, str]:
    """One inference through the real `TorchBackend`; the float tensor it produced.

    `TorchBackend` publishes an image API — uint8 in, uint8 out — and one least
    significant bit is 3.9e-3, so an image-level comparison would measure
    quantisation instead of the engine. The model is therefore run through
    `TorchBackend.load`, which owns the device string, the FP16 gate, the cuDNN
    settings and the error messages, and the float tensor is read from the
    placed descriptor's output. Reaching for the loaded descriptor is the one
    private attribute this script touches; spandrel's own `device` and `dtype`
    properties do the rest, and they are public by design.
    """
    import torch

    from core.backends.torch_backend import TorchBackend

    backend = TorchBackend()
    backend.load(model_path, "fp32", device)
    descriptor = backend._descriptor
    if descriptor is None:  # pragma: no cover - load() either sets it or raises
        raise UpscalerError("TorchBackend.load() returned without a model")
    placed_at, placed_as = descriptor.device, descriptor.dtype
    tensor = torch.from_numpy(_tensor_for(frame)).to(device=placed_at, dtype=placed_as)
    with torch.inference_mode():
        produced = descriptor(tensor)
    return (
        np.asarray(produced.detach().float().cpu().numpy(), dtype=np.float32),
        str(placed_at),
        str(placed_as).removeprefix("torch."),
        backend.precision,
    )


def _to_uint8(tensor: np.ndarray) -> np.ndarray:
    """The same clamp-and-scale the app applies before writing a PNG."""
    return np.clip(tensor * np.float32(255.0), 0, 255).astype(np.uint8)


def _compare(
    label: str, produced: np.ndarray, reference: np.ndarray, tolerance: float
) -> str | None:
    """Return a failure message, or `None` when `produced` agrees with `reference`.

    Both tensors are clamped to `[0, 1]` before they are differenced, and that
    is not a convenience. The app writes `np.clip(out * 255, 0, 255)`, so a
    difference outside `[0, 1]` is discarded before it can reach a pixel; and
    spandrel clamps the model's output while the exported ONNX graph does not,
    because a clamp is not an op the exporter emits. Measured on the real
    checkpoint at 384x288, the two engines differ by 1.4e-6 across 5.3 million
    clamped values and by 7.0e-2 unclamped — every one of the large
    differences being overshoot, where ONNX Runtime says 1.0705 and torch says
    1.0000, and both become 255. Differencing the raw tensors would measure that
    clamp and fail a GPU that is computing the right thing.
    """
    if produced.shape != reference.shape:
        return (
            f"{label}: produced {tuple(produced.shape)}, the CPU reference is "
            f"{tuple(reference.shape)}"
        )
    difference = np.abs(np.clip(produced, 0.0, 1.0) - np.clip(reference, 0.0, 1.0))
    worst = float(difference.max()) if difference.size else 0.0
    mean = float(difference.mean()) if difference.size else 0.0
    levels = int(
        np.abs(
            _to_uint8(produced).astype(np.int16) - _to_uint8(reference).astype(np.int16)
        ).max()
    )
    verdict = "PASS" if worst <= tolerance else "FAIL"
    print(
        f"verify_gpu:   {label}: {verdict} max|d|={worst:.3e} mean|d|={mean:.3e} "
        f"(tolerance {tolerance:.1e}); in pixels that is at most {levels} "
        f"least-significant bit(s)"
    )
    if verdict == "PASS":
        return None
    return (
        f"{label}: max|d| {worst:.3e} exceeds the tolerance {tolerance:.1e} — "
        f"this device does not compute the same function as the CPU reference"
    )


def _describe(devices: list[Device]) -> None:
    """Print every device the probe found, the way the UI shows it."""
    print(f"verify_gpu: {len(devices)} device(s) found by core.devices.probe_all()")
    for device in devices:
        capability = (
            "-"
            if device.compute_capability is None
            else f"sm_{device.compute_capability[0]}{device.compute_capability[1]}"
        )
        state = "usable" if device.usable else f"UNUSABLE — {device.unusable_reason}"
        print(
            f"verify_gpu:   [{device.index}] {device.name}: vendor={device.vendor} "
            f"backend={device.backend} compute_capability={capability} "
            f"{format_memory(device.total_memory_bytes)} — {state}"
        )


def _report_model_failure(model: Path) -> None:
    """Name what the model should have been, so the catalogue can be checked."""
    entry = entry_for(model)
    print(f"verify_gpu: FAILED — no usable model at {model}", file=sys.stderr)
    if entry is None:
        print(
            f"verify_gpu: {model.name} is not in models/catalogue.json, so this "
            "gate cannot state the URL or hash it should have. Pass --model with "
            "a catalogue checkpoint, or add the file to the models directory.",
            file=sys.stderr,
        )
        return
    print("verify_gpu: the catalogue says:", file=sys.stderr)
    print(f"verify_gpu:   id     {entry.id}", file=sys.stderr)
    print(f"verify_gpu:   url     {entry.url}", file=sys.stderr)
    print(f"verify_gpu:   sha256  {entry.sha256}", file=sys.stderr)
    print(f"verify_gpu:   size    {entry.size_bytes} bytes", file=sys.stderr)
    print(
        f"verify_gpu:   install it with the Models tab, or:\n"
        f"verify_gpu:     mkdir -p {models_dir()} && curl -fL -o "
        f"{models_dir() / entry.filename} '{entry.url}'",
        file=sys.stderr,
    )


def run_gate(args: argparse.Namespace, stages: Stages) -> int:
    """The gate itself. Returns the process exit code."""
    failures: list[str] = []

    stages.begin("probe devices")
    devices = probe_all()
    _describe(devices)
    accelerators = [device for device in devices if device.vendor != "cpu"]
    if not accelerators:
        print(
            "verify_gpu: FAILED — no accelerator was found. A gate that ran on "
            "the CPU only would pass without testing anything, which is worse "
            "than not having a gate.",
            file=sys.stderr,
        )
        return 1
    for device in accelerators:
        if not device.usable:
            failures.append(
                f"GPU {device.index} ({device.name}, {device.backend}): "
                f"{device.unusable_reason}"
            )

    stages.begin("model")
    model: Path = args.model or (models_dir() / DEFAULT_MODEL_NAME)
    try:
        verify_model(model, entry_for(model))
    except UpscalerError as exc:
        _report_model_failure(model)
        print(f"verify_gpu:   {exc}", file=sys.stderr)
        return 1
    digest = cached_hash(model) or "not in the catalogue, so not hashed"
    print(f"verify_gpu: model {model} ({model.stat().st_size} bytes, sha256 {digest})")

    stages.begin("runtimes")
    import onnxruntime as ort
    import torch

    print(f"verify_gpu: onnxruntime {ort.__version__} from {ort.__file__}")
    print(
        f"verify_gpu: available providers: {', '.join(ort.get_available_providers())}"
    )
    print(f"verify_gpu: torch {torch.__version__} from {torch.__file__}")

    stages.begin("provider pre-flight")
    from core.backends.onnx_backend import providers_for

    testable: list[Device] = []
    for device in accelerators:
        if not device.usable:
            continue
        try:
            providers = providers_for(device)
        except UpscalerError as exc:
            print(
                f"verify_gpu:   GPU {device.index} {device.name}: {exc}",
                file=sys.stderr,
            )
            failures.append(f"GPU {device.index} ({device.name}): {exc}")
            continue
        print(
            f"verify_gpu:   GPU {device.index} {device.name}: will request {providers}"
        )
        testable.append(device)
    if not testable:
        # Nothing can be tested, so the minutes an export costs would buy
        # nothing. Say so and stop.
        print(
            f"verify_gpu: GATE FAILED — none of the {len(accelerators)} "
            "accelerator(s) can be reached, so nothing was tested.",
            file=sys.stderr,
        )
        _print_failures(failures)
        return 1

    stages.begin("frame")
    frame = _frame(args.height, args.width)
    print(f"verify_gpu: frame {args.width}x{args.height} BGR uint8, deterministic")

    stages.begin(f"export {model.name} to ONNX")
    onnx_path = export_cache_path(model, args.width, args.height)
    if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
        print(
            f"verify_gpu: exporting to {onnx_path}. This is the app's own "
            "exporter on the real 67 MB checkpoint and takes minutes; it is "
            "cached afterwards, and --timeout has to cover it this once.",
            flush=True,
        )
    export_to_onnx(model, onnx_path, args.height, args.width)
    print(f"verify_gpu: graph {onnx_path} ({onnx_path.stat().st_size} bytes)")

    stages.begin("CPU reference (onnxruntime)")
    reference, reference_providers, _ = _onnx_inference(onnx_path, cpu_device(), frame)
    print(f"verify_gpu: reference tensor {reference.shape} via {reference_providers}")

    for device in testable:
        stages.begin(f"onnxruntime on GPU {device.index} ({device.name})")
        try:
            produced, in_use, requested = _onnx_inference(onnx_path, device, frame)
        except Exception as exc:  # ONNX Runtime raises its own exception types
            message = (
                f"GPU {device.index} ({device.name}, {device.backend}): onnxruntime "
                f"could not run the graph: {type(exc).__name__}: {exc}"
            )
            print(f"verify_gpu:   {message}", file=sys.stderr)
            failures.append(message)
            continue
        print(f"verify_gpu:   onnxruntime {ort.__version__} from {ort.__file__}")
        print(f"verify_gpu:   session.get_providers() = {in_use}")
        print(f"verify_gpu:   requested = {requested}")
        if requested[0] not in in_use:
            message = (
                f"GPU {device.index} ({device.name}): asked ONNX Runtime for "
                f"{requested[0]}, the session is using {in_use}. This is the "
                "silent fallback: the run below would be a CPU run wearing a "
                "GPU's name, and it is the failure this gate exists to catch."
            )
            print(f"verify_gpu:   FAILED — {message}", file=sys.stderr)
            failures.append(message)
            continue
        failure = _compare(
            f"onnxruntime on GPU {device.index} ({device.name})",
            produced,
            reference,
            args.tolerance,
        )
        if failure is not None:
            failures.append(f"GPU {device.index} ({device.name}): {failure}")

    if model.suffix.lower() == ".onnx":
        stages.begin("torch")
        print(
            f"verify_gpu: {model.name} is an ONNX graph, so there is no PyTorch "
            "checkpoint to run the torch backend against; that half of the gate "
            "does not apply to this model and is reported as not run rather "
            "than passed.",
            file=sys.stderr,
        )
    else:
        for device in testable:
            stages.begin(f"torch on GPU {device.index} ({device.name})")
            try:
                produced, where, dtype, precision = _torch_inference(
                    model, device, frame
                )
            except Exception as exc:
                message = (
                    f"GPU {device.index} ({device.name}, {device.backend}): the torch "
                    f"backend could not run it: {type(exc).__name__}: {exc}"
                )
                print(f"verify_gpu:   {message}", file=sys.stderr)
                failures.append(message)
                continue
            print(f"verify_gpu:   torch {torch.__version__} from {torch.__file__}")
            print(
                f"verify_gpu:   model placed on {where} at {dtype}; "
                f"TorchBackend.precision={precision}"
            )
            if where.split(":")[0] == "cpu" and device.vendor != "cpu":
                message = (
                    f"GPU {device.index} ({device.name}): the torch backend placed "
                    f"the model on {where}, so the job would run on the CPU"
                )
                print(f"verify_gpu:   FAILED — {message}", file=sys.stderr)
                failures.append(message)
                continue
            failure = _compare(
                f"torch on GPU {device.index} ({device.name})",
                produced,
                reference,
                args.tolerance,
            )
            if failure is not None:
                failures.append(f"GPU {device.index} ({device.name}): {failure}")

    stages.begin("summary")
    print(
        f"verify_gpu: {len(testable)} of {len(accelerators)} accelerator(s) were "
        f"tested against a CPU reference of the real exported model at "
        f"{args.width}x{args.height}, tolerance {args.tolerance:g}"
    )
    if failures:
        print(
            f"verify_gpu: GATE FAILED — {len(failures)} problem(s). "
            "Do not ship a build for this machine.",
            file=sys.stderr,
        )
        _print_failures(failures)
        return 1
    print("verify_gpu: GATE PASSED — every accelerator agrees with the CPU reference.")
    return 0


def _print_failures(failures: list[str]) -> None:
    for failure in failures:
        print(f"verify_gpu:   - {failure}", file=sys.stderr)


def _run_watched(args: argparse.Namespace) -> int:
    """Run the gate in a child process and hold it to the total timeout."""
    with tempfile.TemporaryDirectory(prefix="verify-gpu-") as workspace:
        stage_path = Path(workspace) / "stage"
        command = [
            sys.executable,
            str(_SELF),
            "--stage-file",
            str(stage_path),
            "--tolerance",
            repr(args.tolerance),
            "--timeout",
            repr(args.timeout),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
        ]
        if args.model is not None:
            # Only forwarded when a person asked for it, so that omitting
            # --model keeps the child on exactly the same default.
            command += ["--model", str(args.model)]
        print(
            f"verify_gpu: total timeout {args.timeout:g} s, enforced by a parent "
            "process because a wedged driver hangs inside a C call",
            flush=True,
        )
        with subprocess.Popen(command, cwd=str(_REPO_ROOT)) as child:
            deadline = time.monotonic() + args.timeout
            while True:
                code = child.poll()
                if code is not None:
                    return code
                if time.monotonic() < deadline:
                    time.sleep(0.25)
                    continue
                print(
                    f"verify_gpu: TIMED OUT after {args.timeout:g} s, during stage "
                    f"'{_read_stage(stage_path)}'. Raise --timeout if the one-off "
                    "export of the real checkpoint is what is running.",
                    file=sys.stderr,
                )
                child.terminate()
                try:
                    child.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5.0)
                return 1


def main(argv: list[str] | None = None) -> int:
    """Entry point: the parent applies the timeout, the child does the work."""
    args = _parse_args(argv)
    if args.stage_file is not None:
        return run_gate(args, Stages(args.stage_file))
    if args.timeout > 0:
        return _run_watched(args)
    return run_gate(args, Stages(None))


if __name__ == "__main__":
    sys.exit(main())
