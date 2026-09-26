"""ONNX Runtime: one engine for the NVIDIA, AMD, Intel, Apple and CPU paths.

The execution provider is decided by `Device.backend`, which `core.devices` is
the only thing allowed to produce, and it is **never** negotiated. Asking for
`DmlExecutionProvider` on a build that has no DirectML raises
`BackendUnavailableError` naming the missing provider; falling through to
`CPUExecutionProvider` instead would turn a 40-minute job into a six-hour one
and never say so.

Three vendor requirements are encoded here, each of which fails at *session
creation* rather than at inference if it is ignored:

* **DirectML** "does not support the use of memory pattern optimizations or
  parallel execution ... these options must be disabled or an error will be
  returned" (onnxruntime.ai, DirectML EP). So `enable_mem_pattern = False` and
  `ExecutionMode.ORT_SEQUENTIAL`, read off the enum at call time rather than
  written as a number, because a non-English ORT build labels it differently.
* **CoreML** `MLProgram` "requires macOS 12 or newer"; a refusal surfaces as a
  `Fail`/`InvalidArgument` naming the format, so there is exactly one retry
  with `NeuralNetwork`.
* **CUDA** needs `libcublas`/`libcudart`/`libcudnn` on the dynamic linker's
  search path. `sys.path` cannot do that; the two supported mechanisms are
  `onnxruntime.preload_dlls()` and importing torch first, so the first is
  called before the session exists.

`self.scale` is learned, never guessed: `core.backends.export` reads it off the
graph's input and output shapes. Four consumers — the disk preflight, the
tiling threshold, the completeness check and the encode's output dimensions —
size a 165,000-frame job from it, so a wrong value does not fail loudly, it
produces a video that is silently the wrong size.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from ..config import Device
from ..errors import (
    BackendUnavailableError,
    ModelLoadError,
    UnsupportedScaleError,
)
from ..paths import coreml_cache_dir, runtimes_dir
from .base import Backend, needs_tiling, resolve_tile_size, tiled_infer
from .export import (
    export_cache_path,
    export_to_onnx,
    graph_spatial_shape,
    scale_from_graph,
    scale_from_metadata,
)

_LOG = logging.getLogger(__name__)

#: `Device.backend` -> the provider order ORT is asked for, best first.
_PROVIDER_BY_BACKEND: dict[str, tuple[str, ...]] = {
    "onnx:cuda": ("CUDAExecutionProvider", "CPUExecutionProvider"),
    "onnx:migraphx": ("MIGraphXExecutionProvider", "CPUExecutionProvider"),
    "onnx:dml": ("DmlExecutionProvider", "CPUExecutionProvider"),
    "onnx:coreml": ("CoreMLExecutionProvider", "CPUExecutionProvider"),
    "cpu": ("CPUExecutionProvider",),
}

#: The MIGraphX EP reads this at session creation; without it every load
#: recompiles the graph, which on a 60 MB model is minutes per run.
MIGRAPHX_CACHE_ENV = "ORT_MIGRAPHX_MODEL_CACHE_PATH"

#: Where the Runtimes tab unpacks a CUDA runtime, and therefore what
#: `preload_dlls` is pointed at when one has been installed.
CUDA_RUNTIME_DIRNAME = "cuda"


def _log_to_module(message: str) -> None:
    """Report a backend-level warning when the caller supplied no `log`."""
    _LOG.warning("%s", message)


def _noop(message: str) -> None:
    """Swallow a log line; the default for a backend nobody is watching."""


def providers_for(device: Device) -> list[str]:
    """The provider order to request for `device`.

    Raises rather than degrading: the first entry is the one the user selected
    a GPU for, and ORT will happily run on the CPU underneath a CUDA request
    while reporting success.
    """
    try:
        wanted = _PROVIDER_BY_BACKEND[device.backend]
    except KeyError:
        raise BackendUnavailableError(
            f"{device.name} has unknown backend {device.backend!r}; "
            "core.devices should never produce one"
        ) from None
    available = ort.get_available_providers()
    missing = [name for name in wanted if name not in available]
    if missing:
        raise BackendUnavailableError(
            f"{device.name} needs the {', '.join(missing)} execution provider, which "
            f"this ONNX Runtime build does not have (it offers "
            f"{', '.join(available)}). Install the matching runtime from the "
            "Runtimes tab, or pick the CPU device."
        )
    return list(wanted)


def _session_options(provider: str) -> ort.SessionOptions:
    """Session options for `provider`, including the DirectML requirement.

    `ExecutionMode.ORT_SEQUENTIAL` is read as an attribute, not as a literal:
    the enum's value is an implementation detail of the ORT build in use.
    """
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "DmlExecutionProvider":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def _preload_cuda(log: Callable[[str], None]) -> None:
    """Load the CUDA/cuDNN shared libraries before any session is built.

    A failure here is reported and then tolerated, because the CUDA runtime
    may already be resident (importing torch first loads the same libraries) —
    but it is never silent, since this is precisely how a CUDA request ends up
    running on the CPU.
    """
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        log(
            "This ONNX Runtime build has no preload_dlls(); it will search for "
            "libcudnn itself and may fall back to the CPU"
        )
        return
    root = runtimes_dir() / CUDA_RUNTIME_DIRNAME
    try:
        if root.is_dir():
            preload(directory=str(root))
        else:
            preload()
    except Exception as exc:  # reported below, then tolerated
        log(f"preload_dlls failed ({exc}); the CUDA provider may fall back to the CPU")


def _run_session(
    model_path: Path,
    options: ort.SessionOptions,
    providers: list[str],
    provider_options: list[dict[str, Any]] | None,
) -> ort.InferenceSession:
    """Build one session, keeping ORT's own message intact when it fails."""
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=providers,
        provider_options=provider_options,
    )


def _coreml_provider_options(model_format: str) -> list[dict[str, Any]]:
    """The documented CoreML EP options for one `ModelFormat`."""
    return [
        {
            "ModelFormat": model_format,
            "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "0",
            "EnableOnSubgraphs": "0",
            "ModelCacheDirectory": str(coreml_cache_dir()),
        }
    ]


def _mentions_mlprogram(error: Exception) -> bool:
    """Whether an ORT failure is the documented `MLProgram` refusal.

    The match is deliberately loose. ORT surfaces this as `Fail(...)` and
    `InvalidArgument(...)` with the format named inside, and the only cost of
    a false positive is one retry with the other format before a clear error.
    """
    text = str(error).lower()
    return "mlprogram" in text or "fail" in text or "invalidargument" in text


def _coreml_session(
    model_path: Path,
    options: ort.SessionOptions,
    providers: list[str],
    log: Callable[[str], None],
) -> ort.InferenceSession:
    """Build a CoreML session, retrying once with `NeuralNetwork`.

    `MLProgram` is refused below macOS 12 and there is no way to ask which
    formats are available, so the retry is the documented detection.
    """
    try:
        return _run_session(
            model_path, options, providers, _coreml_provider_options("MLProgram")
        )
    except Exception as exc:
        if not _mentions_mlprogram(exc):
            raise
    log("CoreML MLProgram unavailable (macOS < 12); using NeuralNetwork format")
    try:
        return _run_session(
            model_path, options, providers, _coreml_provider_options("NeuralNetwork")
        )
    except Exception as exc:
        raise BackendUnavailableError(
            f"CoreML could not load {model_path.name} as either MLProgram or "
            f"NeuralNetwork: {exc}"
        ) from exc


def session_for(
    model_path: Path,
    device: Device,
    *,
    log: Callable[[str], None] | None = None,
) -> ort.InferenceSession:
    """An inference session for `model_path` on `device`.

    `log` receives provider-level warnings; it defaults to this module's
    logger so a direct caller is never left in the dark.
    """
    report: Callable[[str], None] = _log_to_module if log is None else log
    providers = providers_for(device)
    provider = providers[0]

    if provider == "CUDAExecutionProvider":
        _preload_cuda(report)

    if provider == "MIGraphXExecutionProvider":
        cache = runtimes_dir() / "migraphx-cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[MIGRAPHX_CACHE_ENV] = str(cache)
        return _run_session(
            model_path,
            _session_options(provider),
            providers,
            [{"device_id": str(device.index)}],
        )

    if provider == "CoreMLExecutionProvider":
        return _coreml_session(
            model_path, _session_options(provider), providers, report
        )

    return _run_session(model_path, _session_options(provider), providers, None)


def _graph_input_dtype(session: ort.InferenceSession) -> str:
    """The element type the graph was built with, e.g. `"float16"`.

    Precision is a property of the exported graph, not of the session: a
    float32 model stays float32 on a CUDA device no matter what was asked for,
    and pretending otherwise would misreport what the job is actually running.
    """
    declared: str = session.get_inputs()[0].type
    return declared.rsplit("(", 1)[-1].rstrip(")")


class OnnxBackend(Backend):
    """One ONNX model, ready to turn BGR frames into bigger BGR frames.

    `self.scale` is the contract everything downstream sizes a job from: the
    disk preflight, the tiling threshold, the completeness check and the
    encode's output dimensions all assume the output is exactly `scale` times
    the input. It is read off the graph, and when the graph cannot say —
    a dynamic-shape model with no `scale` metadata — `infer` checks the ratio
    it actually received and raises `UnsupportedScaleError` rather than writing
    a video of the wrong size.

    Attributes:
        name: Shown in the log.
        precision: The precision the session will actually run at, which is
            the graph's own dtype rather than the request.
        log: Where provider-level warnings go. Replace it before `load()`.
    """

    name = "onnxruntime"

    def __init__(
        self, *, export_size: tuple[int, int] | None = None, tile_size: int = 0
    ) -> None:
        """`export_size` is `(height, width)` of the frames being upscaled.

        A `.pth` has to be exported, and an export is specific to one frame
        size because the static spatial dims are what make the CoreML rewrite
        in `core.backends.export` possible: it builds a constant shape vector,
        which a symbolic spatial dim cannot give it. The ABC's `load()` has
        nowhere to put a frame size, so it arrives here.
        """
        super().__init__()
        self.export_size = export_size
        self._tile_size = tile_size
        self._configured_tile_size = tile_size
        self.precision = "fp32"
        self.log: Callable[[str], None] = _noop
        self._session: ort.InferenceSession | None = None
        self._input_hw: tuple[int, int] | None = None
        self._input_name = ""
        self._output_name = ""
        self._scale_is_assumed = False

    def load(self, model_path: Path, precision: str, device: Device) -> None:
        """Session the model (exporting first if it is a checkpoint)."""
        session_path = self._model_to_session(model_path)
        self._session = session_for(session_path, device, log=self.log)
        self._input_hw = graph_spatial_shape(session_path)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._check_channels(session_path)
        self.scale = self._learn_scale(session_path)
        self.precision = self._resolve_precision(device, precision)
        self._configured_tile_size = self._tile_size

    def _model_to_session(self, model_path: Path) -> Path:
        """The `.onnx` to session: the file itself, or a cached export of it."""
        if model_path.suffix.lower() == ".onnx":
            return model_path
        if self.export_size is None:
            raise BackendUnavailableError(
                f"Exporting {model_path.name} needs the frame size; construct the "
                "backend with export_size=(height, width)"
            )
        height, width = self.export_size
        # `export_to_onnx` is already cached and already asserts the op set, so
        # neither is repeated here.
        return export_to_onnx(
            model_path,
            export_cache_path(model_path, width, height),
            height,
            width,
        )

    def _check_channels(self, session_path: Path) -> None:
        """Refuse a graph the app cannot feed.

        The worker reads every frame with `cv2.imread`, so it only ever has a
        3-channel BGR image. A 1-channel model would return a 1-channel result
        that `cv2.imwrite` then writes as a greyscale file — a silent change of
        the user's video, not an error.
        """
        session = self._session
        assert session is not None
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        in_channels = inputs[0].shape[1] if len(inputs[0].shape) > 1 else None
        out_channels = outputs[0].shape[1] if len(outputs[0].shape) > 1 else None
        if in_channels not in (None, 3) or out_channels not in (None, 3):
            raise ModelLoadError(
                f"{session_path.name} takes {in_channels} channels and produces "
                f"{out_channels}; the app only feeds 3-channel BGR frames"
            )

    def _learn_scale(self, session_path: Path) -> int:
        """The output multiplier, and whether it had to be assumed."""
        from_graph = scale_from_graph(session_path)
        if from_graph is not None:
            self._scale_is_assumed = False
            return from_graph
        from_metadata = scale_from_metadata(session_path)
        if from_metadata is not None:
            self._scale_is_assumed = False
            return from_metadata
        self._scale_is_assumed = True
        return 1

    def _resolve_precision(self, device: Device, requested: str) -> str:
        """The precision the session will run at, and why."""
        session = self._session
        assert session is not None
        if device.backend in ("cpu", "onnx:coreml"):
            # Neither provider has a documented half path for these graphs.
            return "fp32"
        graph_dtype = _graph_input_dtype(session)
        if graph_dtype == "float16":
            return "fp16"
        if requested == "fp16":
            self.log(
                f"FP16 was requested but the graph's input is {graph_dtype}; running "
                f"{graph_dtype} anyway — a session cannot change an export's dtype"
            )
        return "fp32"

    def _fit_to_input(self, tensor: np.ndarray) -> np.ndarray:
        """Pad a `(1, C, H, W)` tensor up to the graph's declared input shape.

        This only ever bites a graph with concrete spatial dims, which is what
        our own exporter produces and what a user-supplied static-shape .onnx
        looks like; ORT refuses any other size once the shape is fixed. Tiling
        reaches it on the last tile of every row and column, and so does a frame
        whose size is not a multiple of the tile.

        Edges are replicated rather than zeroed so a partial tile does not
        infer against a black border, which the feathering in `tiled_infer`
        would then blend across the frame.
        """
        if self._input_hw is None:
            return tensor
        height, width = self._input_hw
        have = (int(tensor.shape[2]), int(tensor.shape[3]))
        if have == (height, width):
            return tensor
        if have[0] > height or have[1] > width:
            raise ModelLoadError(
                f"a {have[0]}x{have[1]} frame does not fit the graph's "
                f"{height}x{width} input; construct the backend with a matching "
                "export_size or tile_size"
            )
        padded = np.pad(
            tensor,
            ((0, 0), (0, 0), (0, height - have[0]), (0, width - have[1])),
            mode="edge",
        )
        if int(padded.shape[2]) != height or int(padded.shape[3]) != width:
            raise ModelLoadError(
                f"could not pad a {have[0]}x{have[1]} frame to the graph's "
                f"{height}x{width} input"
            )
        return padded

    def _crop_output(self, upscaled: np.ndarray, height: int, width: int) -> np.ndarray:
        """Undo `_fit_to_input`: cut the padded border back off."""
        if self._scale_is_assumed and height and width:
            produced = int(upscaled.shape[2]) // height
            if produced != self.scale:
                raise UnsupportedScaleError(
                    f"the graph produced {produced}x for a {height}x{width} frame "
                    f"but this backend reports scale {self.scale}; the disk "
                    "estimate, the output canvas and the completeness check are "
                    "all sized from that number"
                )
        return upscaled[:, :, : height * self.scale, : width * self.scale]

    def _infer_full(self, image_bgr: np.ndarray) -> np.ndarray:
        """One frame, no tiling. The contract is BGR uint8 in and out."""
        session = self._session
        assert session is not None
        height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])
        rgb = image_bgr[:, :, ::-1]
        tensor = np.ascontiguousarray(
            rgb.transpose(2, 0, 1)[None], dtype=np.float32
        ) / np.float32(255.0)
        fitted = self._fit_to_input(tensor)
        upscaled = session.run([self._output_name], {self._input_name: fitted})[0]
        cropped = self._crop_output(upscaled, height, width)
        rgb_out = np.clip(
            cropped[0].transpose(1, 2, 0) * np.float32(255.0), 0, 255
        ).astype(np.uint8)
        return np.ascontiguousarray(rgb_out[:, :, ::-1])

    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        """Upscale one BGR uint8 frame, honouring the tiling rule."""
        height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])
        tile = resolve_tile_size(height, width, self._configured_tile_size)
        if needs_tiling(height, width):
            return tiled_infer(self._infer_full, image_bgr, tile, self.scale)
        return self._infer_full(image_bgr)

    def session_providers(self) -> list[str]:
        """The providers the session is really using, not the ones requested.

        ORT accepts a provider it cannot create and runs the graph on the CPU
        instead, reporting success; `verify_gpu.py` and the Runtimes tab need
        the truth to tell that apart.
        """
        if self._session is None:
            return []
        return list(self._session.get_providers())

    def close(self) -> None:
        """Drop the session and let ORT free the device memory."""
        self._session = None
        self._input_hw = None
