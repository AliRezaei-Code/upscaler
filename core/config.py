"""The job configuration: what to run, where, and on which devices.

Both front-ends build a `JobConfig` and hand it to `UpscalerApp`. Nothing
else in the app makes policy decisions, which is why this module imports only
the standard library and `core.errors`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .errors import ConfigError

Vendor = Literal["nvidia", "amd", "intel", "apple", "cpu"]
Precision = Literal["auto", "fp16", "fp32"]


def format_memory(total_memory_bytes: int) -> str:
    """Render a VRAM size, or say plainly that the driver did not report one."""
    if total_memory_bytes <= 0:
        return "VRAM unknown"
    mib = total_memory_bytes / (1024**2)
    if mib < 1024:
        return f"{mib:.0f} MiB VRAM"
    return f"{mib / 1024:.0f} GiB VRAM"


@dataclass(frozen=True)
class Device:
    """One accelerator, as the probe classified it.

    Attributes:
        index: Ordinal within this backend's own device space, so it can be
            passed straight to CUDA, DirectML or MIGraphX.
        name: Human-readable device name, e.g. `"Tesla P40"`.
        vendor: Which vendor's driver owns it.
        backend: Which execution backend serves it — `"onnx:cuda"`,
            `"onnx:migraphx"`, `"onnx:dml"`, `"onnx:coreml"`, `"torch:xpu"`,
            `"torch:mps"` or `"cpu"`.
        total_memory_bytes: VRAM, or `0` when the driver would not say.
        pci_bus_id: PCI bus address, used to order devices the way CUDA does.
        compute_capability: `(major, minor)`, or `None` when unknown.
        usable: Whether `run_job` will accept this device.
        unusable_reason: Why not, when `usable` is false. Shown as the row
            tooltip in both front-ends.
    """

    index: int
    name: str
    vendor: Vendor
    backend: str
    total_memory_bytes: int
    pci_bus_id: str | None
    compute_capability: tuple[int, int] | None
    usable: bool
    unusable_reason: str | None

    @property
    def label(self) -> str:
        """The one-line row label both front-ends display.

        Lives here rather than in either UI so the PySide6 and Toga device
        lists cannot drift apart.
        """
        kind = "CPU" if self.vendor == "cpu" else f"GPU {self.index}"
        memory = format_memory(self.total_memory_bytes)
        return f"{kind} — {self.name} — {memory} — {self.backend}"


@dataclass(frozen=True)
class JobConfig:
    """Everything one run needs.

    `scale` is a field rather than something threaded through from the model
    catalogue because four separate places consume it — the preflight, the
    tiling threshold, the completeness check and the encode's output
    dimensions — and a `.pth` picked through "Browse local…" has no catalogue
    entry to read it from.

    Attributes:
        input_path: The source video.
        output_path: Where the encoded result goes.
        model_path: The checkpoint, `.onnx` or `.safetensors`.
        work_dir: Scratch directory for `frames_in` and `frames_out`.
        devices: The accelerators to spread frames across, in selection order.
        scale: Output multiplier; `0` means "read it from the model".
        precision: `"auto"` resolves per device from its FP16 capability.
        tile_size: `0` lets `resolve_tile_size` pick from the frame size.
        fps: `None` inherits the source frame rate.
        crf: x264 quality, 0 (lossless) to 51 (worst).
        delete_frames_after_encode: Removes `frames_out` once the video is
            written. It is 820 GB on a 165,303-frame run, and off by default.
    """

    input_path: Path
    output_path: Path
    model_path: Path
    work_dir: Path
    devices: tuple[Device, ...]
    scale: int = 0
    precision: Precision = "auto"
    tile_size: int = 0
    fps: float | None = None
    crf: int = 18
    delete_frames_after_encode: bool = False

    def validate(self) -> None:
        """Raise `ConfigError` if this configuration cannot be run.

        Checks run in a fixed order — input, devices, device health, CRF — so
        the first message a user sees is the first thing that is actually
        wrong, not an arbitrary one.
        """
        if not self.input_path.is_file():
            raise ConfigError(f"Input video not found: {self.input_path}")
        if not self.devices:
            raise ConfigError("Select at least one GPU")
        broken = [device for device in self.devices if not device.usable]
        if broken:
            raise ConfigError(
                "; ".join(f"{d.name}: {d.unusable_reason}" for d in broken)
            )
        if not 0 <= self.crf <= 51:
            raise ConfigError(f"CRF must be 0-51, got {self.crf}")
