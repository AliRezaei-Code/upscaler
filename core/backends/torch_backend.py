"""PyTorch + spandrel: the engine for `.pth` checkpoints and the `torch:*` devices.

ONNX Runtime has no Intel execution provider, so an Intel GPU on Linux is served
by `torch.xpu`; macOS falls back to torch `mps` when CoreML is not an option.
Everything else about this module is the same shape as `onnx_backend`: images
in, images out, one tiling rule shared from `core.backends.base`.

Two decisions worth stating.

**The conversion is here, not in spandrel.** spandrel 0.4 dropped
`ModelDescriptor.preprocess`/`postprocess`; `ImageModelDescriptor.__call__` takes
a `(1, C, H, W)` tensor already in `[0, 1]` and returns one in the same range,
and it pads to the architecture's `size_requirements` itself. So BGR→RGB, the
`[0, 1]` scaling and the `uint8` conversion are done here, and the colour
reversal on the way out is the same one the ONNX backend performs — the two
engines must return identical bytes for the same frame or a model swap changes
the video.

**FP16 is a hardware gate, not a preference.** `auto` follows
`core.devices.is_fp16_capable`, and `None` compute capability — a driver that
never answered — is False, because FP16 measured *slower* than FP32 on the
Pascal cards this was written against. Asking for FP16 on an architecture
spandrel marks as half-unsupported raises rather than quietly degrading, because
a subtly wrong video is worse than a refusal.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import spandrel
import torch

from ..config import Device
from ..devices import is_fp16_capable
from ..errors import BackendUnavailableError, ModelLoadError
from .base import Backend, needs_tiling, resolve_tile_size, tiled_infer

#: `Device.backend` -> the torch device string. A CUDA device keeps the
#: `cuda:{index}` form because the pipeline hands out per-device workers and
#: the index is what distinguishes them.
_TORCH_DEVICE_BY_BACKEND: dict[str, Callable[[Device], str]] = {
    "torch:xpu": lambda device: f"xpu:{device.index}",
    "torch:mps": lambda _device: "mps",
    "onnx:cuda": lambda device: f"cuda:{device.index}",
    "cpu": lambda _device: "cpu",
}


def _noop(message: str) -> None:
    """Swallow a log line; the default for a backend nobody is watching."""


def _device_string(device: Device) -> str:
    """The torch device string for a probed `Device`."""
    try:
        build = _TORCH_DEVICE_BY_BACKEND[device.backend]
    except KeyError:
        raise ModelLoadError(
            f"{device.name} has unknown backend {device.backend!r}; "
            "core.devices should never produce one"
        ) from None
    return build(device)


def _uses_cuda(device: Device) -> bool:
    """Whether this run owns a CUDA context that has to be released."""
    return device.backend == "onnx:cuda"


class TorchBackend(Backend):
    """One spandrel model, ready to turn BGR frames into bigger BGR frames.

    Attributes:
        name: Shown in the log.
        precision: `"fp16"` or `"fp32"`, resolved in `load()` from the
            request, the compute capability and what the model supports.
        log: Where load-time warnings go. Replace it before `load()`.
    """

    name = "torch"

    def __init__(self, *, tile_size: int = 0) -> None:
        """`tile_size` is 0 to let `resolve_tile_size` decide from the frame."""
        super().__init__()
        self._tile_size = tile_size
        self._configured_tile_size = tile_size
        self.precision = "fp32"
        self.log: Callable[[str], None] = _noop
        self._descriptor: spandrel.ImageModelDescriptor | None = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype = torch.float32

    def load(self, model_path: Path, precision: str, device: Device) -> None:
        """Load `model_path` onto `device` and set `self.scale`."""
        descriptor = self._load_descriptor(model_path)
        self.scale = int(descriptor.scale)
        self._device = torch.device(_device_string(device))

        # cuDNN autotuning picks the fastest convolution algorithm per shape,
        # which is worth it for a job that runs one shape tens of thousands of
        # times; determinism would forbid it.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        use_half = precision == "fp16" or (
            precision == "auto" and is_fp16_capable(device.compute_capability)
        )
        if use_half and not descriptor.supports_half:
            raise ModelLoadError(
                f"{model_path.name} ({descriptor.architecture.id}) has no FP16 "
                "implementation; choose FP32 or Auto"
            )
        self.precision = "fp16" if use_half else "fp32"
        self._dtype = torch.float16 if use_half else torch.float32

        if _uses_cuda(device):
            # `torch.version.cuda` is a property of the wheel, so this is an
            # instant check that never touches the driver — `torch.cuda.set_device`
            # on a CPU build raises a bare AttributeError, and `is_available()`
            # would hang on a wedged one, which is `core.devices`' job not ours.
            if torch.version.cuda is None:
                raise BackendUnavailableError(
                    f"{device.name} needs a CUDA build of PyTorch; the bundled "
                    f"torch {torch.__version__} has no CUDA. Install the CUDA "
                    "runtime from the Runtimes tab, or select the CPU device."
                )
            # Each worker is pinned to its own card before the first allocation,
            # so every later `cuda:{index}` is this card.
            torch.cuda.set_device(device.index)

        self._descriptor = self._place(descriptor, model_path)

    def _load_descriptor(self, model_path: Path) -> spandrel.ImageModelDescriptor:
        """spandrel's descriptor, or a `ModelLoadError` naming what came back.

        `spandrel.ModelDescriptor` is a generic alias, and `isinstance` against
        a subscripted generic raises `TypeError` rather than answering, so the
        only check made is against `ImageModelDescriptor` — which is the one
        that matters, and which catches the masked descriptors spandrel also
        returns.
        """
        try:
            descriptor = spandrel.ModelLoader().load_from_file(model_path)
        except Exception as exc:
            raise ModelLoadError(
                f"spandrel could not load {model_path.name}: {exc}"
            ) from exc
        if not isinstance(descriptor, spandrel.ImageModelDescriptor):
            raise ModelLoadError(
                f"{model_path.name} loaded as {type(descriptor).__name__}, which is "
                "not an image model; the app only upscales images"
            )
        return descriptor

    def _place(
        self, descriptor: spandrel.ImageModelDescriptor, model_path: Path
    ) -> spandrel.ImageModelDescriptor:
        """Move and cast the model, turning spandrel's and torch's refusals into
        ours.

        Nothing runs the model here, so a failure is always about the device or
        the dtype rather than about the checkpoint.
        """
        assert self._device is not None
        try:
            return descriptor.to(device=self._device, dtype=self._dtype)
        except spandrel.UnsupportedDtypeError as exc:
            raise ModelLoadError(
                f"{model_path.name} ({descriptor.architecture.id}) cannot run at "
                f"{self.precision}: {exc}"
            ) from exc
        except (AssertionError, RuntimeError, ValueError) as exc:
            raise BackendUnavailableError(
                f"torch could not place {model_path.name} on {self._device}: {exc}"
            ) from exc

    def _infer_full(self, image_bgr: np.ndarray) -> np.ndarray:
        """One frame, no tiling. The contract is BGR uint8 in and out."""
        descriptor = self._descriptor
        assert descriptor is not None
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1].transpose(2, 0, 1)[None])
        tensor = torch.from_numpy(rgb).to(device=self._device, dtype=self._dtype)
        tensor = tensor / 255.0
        with torch.inference_mode():
            upscaled = descriptor(tensor)
        frame = (
            upscaled[0]
            .permute(1, 2, 0)
            .clamp(0, 1)
            .mul(255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        return np.ascontiguousarray(frame[:, :, ::-1])

    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        """Upscale one BGR uint8 frame, honouring the tiling rule."""
        height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])
        tile = resolve_tile_size(height, width, self._configured_tile_size)
        if needs_tiling(height, width):
            return tiled_infer(self._infer_full, image_bgr, tile, self.scale)
        return self._infer_full(image_bgr)

    def close(self) -> None:
        """Drop the model and hand the device memory back to the driver.

        A worker is reused across chunks, so a model left behind would pin its
        weights for the rest of the process.
        """
        self._descriptor = None
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.empty_cache()
