"""Which engine serves which model on which device.

The routing rule is deliberately dumb: the device's `backend` prefix decides
the engine, and the model file's extension decides whether the ONNX path has to
export first. Anything else — "try CUDA, then DML, then CPU" — is a decision
this module would have to repeat per call site, and the point of the plan's
provider matrix is that there is exactly one answer per (vendor, OS) pair.

`core.devices` is the only enumerator. Nothing here probes hardware.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Device
from ..devices import is_fp16_capable
from ..errors import BackendUnavailableError
from .base import Backend

#: Extensions spandrel can load and `export_to_onnx` can convert.
TORCH_MODEL_SUFFIXES = frozenset({".pth", ".ckpt", ".safetensors", ".pt"})
ONNX_MODEL_SUFFIXES = frozenset({".onnx"})


def resolve_precision(device: Device, precision: str) -> str:
    """The precision this device will actually run at.

    `"auto"` follows the hardware, not the model's ambitions: a device whose
    compute capability is unknown — because the driver never answered — gets
    FP32, which is both the conservative branch and, on the reference workload,
    the faster one.
    """
    if precision == "fp32":
        return "fp32"
    if precision == "fp16":
        return "fp16"
    return "fp16" if is_fp16_capable(device.compute_capability) else "fp32"


def backend_for_device(device: Device) -> type[Backend]:
    """The engine class that serves `device`, ignoring the model.

    Split out from `select_backend` because the Runtimes tab and the UI want to
    say "this device runs on DirectML" without naming a model file.
    """
    from .onnx_backend import OnnxBackend
    from .torch_backend import TorchBackend

    if device.backend.startswith("onnx:"):
        return OnnxBackend
    if device.backend.startswith("torch:") or device.backend == "cpu":
        return TorchBackend
    raise BackendUnavailableError(
        f"Unknown backend {device.backend!r} for {device.name}; "
        "core.devices should never produce one"
    )


def select_backend(devices: tuple[Device, ...], model_path: Path) -> type[Backend]:
    """The engine class for this model on this device.

    Raises `BackendUnavailableError` rather than falling back: a `.pth` that
    cannot be exported on an `onnx:` device is a real dead end, and silently
    running it on the CPU of the same machine would turn a 40-minute job into a
    6-hour one without saying so.
    """
    from .onnx_backend import OnnxBackend

    if not devices:
        raise BackendUnavailableError("No device selected")
    device = devices[0]
    suffix = model_path.suffix.lower()

    if suffix in ONNX_MODEL_SUFFIXES:
        # The CPU device runs ONNX Runtime too — `CPUExecutionProvider` — so an
        # existing .onnx must not be refused on a machine with no accelerator.
        # `torch:xpu` and `torch:mps` cannot, because there is no ONNX EP for
        # either: Intel Extension for PyTorch is archived, and CoreML is only
        # reachable through ONNX Runtime.
        if device.backend == "cpu" or device.backend.startswith("onnx:"):
            return OnnxBackend
        raise BackendUnavailableError(
            f"{model_path.name} is an ONNX model but {device.name} has no ONNX "
            f"execution provider ({device.backend}); use a .pth model instead"
        )

    if suffix in TORCH_MODEL_SUFFIXES:
        return backend_for_device(device)

    raise BackendUnavailableError(
        f"Unsupported model file {model_path.name}; expected one of "
        f"{', '.join(sorted(ONNX_MODEL_SUFFIXES | TORCH_MODEL_SUFFIXES))}"
    )
