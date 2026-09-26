"""Backend routing: which engine serves which model on which device.

Routing is a pure function of the model file's extension and the device's
`backend` prefix, so all of it is testable without an accelerator. The parts
that need a real device are the provider lists, and those are the agent's
tests' job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.backends.base import Backend
from core.backends.registry import (
    ONNX_MODEL_SUFFIXES,
    TORCH_MODEL_SUFFIXES,
    backend_for_device,
    resolve_precision,
    select_backend,
)
from core.config import Device
from core.errors import BackendUnavailableError


def make_device(backend: str, **overrides: object) -> Device:
    base: dict[str, object] = {
        "index": 0,
        "name": "test device",
        "vendor": "nvidia",
        "backend": backend,
        "total_memory_bytes": 24 * 1024**3,
        "pci_bus_id": "0:01:00.0",
        "compute_capability": (8, 6),
        "usable": True,
        "unusable_reason": None,
    }
    base.update(overrides)
    return Device(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "backend",
    [
        "onnx:cuda",
        "onnx:migraphx",
        "onnx:dml",
        "onnx:coreml",
        "torch:xpu",
        "torch:mps",
        "cpu",
    ],
)
def test_every_backend_prefix_routes_to_an_engine(backend: str) -> None:
    from core.backends.onnx_backend import OnnxBackend
    from core.backends.torch_backend import TorchBackend

    expected = OnnxBackend if backend.startswith("onnx:") else TorchBackend
    assert backend_for_device(make_device(backend)) is expected


@pytest.mark.parametrize(
    "backend", ["onnx:cuda", "onnx:migraphx", "onnx:dml", "onnx:coreml"]
)
def test_onnx_models_go_to_the_onnx_engine(backend: str) -> None:
    from core.backends.onnx_backend import OnnxBackend

    devices = (make_device(backend),)
    assert select_backend(devices, Path("/models/x.onnx")) is OnnxBackend


@pytest.mark.parametrize("backend", ["onnx:cuda", "torch:xpu", "torch:mps", "cpu"])
def test_torch_models_go_to_the_engine_the_device_has(backend: str) -> None:
    from core.backends.onnx_backend import OnnxBackend
    from core.backends.torch_backend import TorchBackend

    devices = (make_device(backend),)
    chosen = select_backend(devices, Path("/models/x.pth"))
    assert chosen is (OnnxBackend if backend.startswith("onnx:") else TorchBackend)


def test_an_intel_linux_device_never_routes_through_onnx() -> None:
    """There is no ONNX Runtime Intel EP, so `torch:xpu` must not reach ONNX."""
    from core.backends.torch_backend import TorchBackend

    device = make_device("torch:xpu", vendor="intel")
    assert select_backend((device,), Path("/models/x.pth")) is TorchBackend
    assert issubclass(select_backend((device,), Path("/models/x.pth")), Backend)


def test_an_onnx_model_on_a_torch_only_device_is_refused() -> None:
    device = make_device("torch:xpu", vendor="intel")
    with pytest.raises(BackendUnavailableError, match="no ONNX execution provider"):
        select_backend((device,), Path("/models/x.onnx"))


def test_an_unknown_extension_is_refused() -> None:
    with pytest.raises(BackendUnavailableError, match="Unsupported model file"):
        select_backend((make_device("cpu"),), Path("/models/x.txt"))


def test_no_device_is_refused() -> None:
    with pytest.raises(BackendUnavailableError, match="No device selected"):
        select_backend((), Path("/models/x.pth"))


def test_an_unknown_backend_prefix_is_refused() -> None:
    with pytest.raises(BackendUnavailableError, match="Unknown backend"):
        backend_for_device(make_device("quantum:annealer"))


@pytest.mark.parametrize("suffix", sorted(TORCH_MODEL_SUFFIXES | ONNX_MODEL_SUFFIXES))
def test_every_declared_suffix_is_accepted(suffix: str) -> None:
    chosen = select_backend((make_device("cpu"),), Path(f"/models/x{suffix}"))
    assert issubclass(chosen, Backend)


@pytest.mark.parametrize(
    ("capability", "requested", "expected"),
    [
        ((6, 1), "auto", "fp32"),
        ((7, 0), "auto", "fp16"),
        ((7, 5), "auto", "fp16"),
        ((8, 6), "auto", "fp16"),
        (None, "auto", "fp32"),
        ((8, 6), "fp32", "fp32"),
        ((6, 1), "fp16", "fp16"),
        (None, "fp16", "fp16"),
    ],
)
def test_precision_follows_the_hardware(
    capability: tuple[int, int] | None, requested: str, expected: str
) -> None:
    device = make_device("onnx:cuda", compute_capability=capability)
    assert resolve_precision(device, requested) == expected


def test_a_pascal_device_is_forced_to_fp32() -> None:
    """The reference hardware: 3x Tesla P40 at compute capability 6.1."""
    device = make_device("onnx:cuda", name="Tesla P40", compute_capability=(6, 1))
    assert resolve_precision(device, "auto") == "fp32"


def test_a_degraded_device_with_unknown_capability_is_forced_to_fp32() -> None:
    device = make_device("onnx:cuda", compute_capability=None, total_memory_bytes=0)
    assert resolve_precision(device, "auto") == "fp32"
