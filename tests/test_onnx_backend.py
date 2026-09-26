"""What `OnnxBackend` guarantees to the pipeline and to the log.

Every test that infers runs the real `CPUExecutionProvider` over a real graph
exported by `torch.onnx`, so a passing run means the numbers came out of ONNX
Runtime and not out of a mock. The provider-specific branches — DirectML's
mandatory session options, the CUDA DLL preload, the CoreML format retry, the
MIGraphX compiler cache — cannot run on a machine with no GPU, so each is
exercised through the same seam the production code uses: the attributes it
reads off the `onnxruntime` module at call time.

No test is marked `gpu`: this suite has to pass on a CI runner with no
accelerator at all.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pytest

from core.backends import onnx_backend
from core.backends.export import exported_ops
from core.backends.onnx_backend import MIGRAPHX_CACHE_ENV, OnnxBackend, providers_for
from core.config import Device
from core.errors import BackendUnavailableError, UnsupportedScaleError

COREML_RETRY_LOG = (
    "CoreML MLProgram unavailable (macOS < 12); using NeuralNetwork format"
)

#: The providers this suite lets a fake build claim to have.
#: `ort.get_available_providers()` reports only what the installed wheel was
#: compiled with, which on a CPU runner is Azure and CPU, so the branches under
#: test have to declare their own.
ALL_PROVIDERS = [
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]

#: Two exported values can differ by one level because `astype(uint8)`
#: truncates a float32 product of nines over nines.
TRUNCATION_SLACK = 1


def make_device(**overrides: object) -> Device:
    """A CUDA device, with the given fields replaced."""
    base: dict[str, object] = {
        "index": 0,
        "name": "Test Device",
        "vendor": "nvidia",
        "backend": "onnx:cuda",
        "total_memory_bytes": 24 * 1024**3,
        "pci_bus_id": "00000000:07:00.0",
        "compute_capability": (8, 6),
        "usable": True,
        "unusable_reason": None,
    }
    base.update(overrides)
    return Device(**base)  # type: ignore[arg-type]


def cpu_device(**overrides: object) -> Device:
    """The CPU fallback device that the real inference tests run on."""
    base: dict[str, object] = {
        "name": "CPU",
        "vendor": "cpu",
        "backend": "cpu",
        "total_memory_bytes": 0,
        "pci_bus_id": None,
        "compute_capability": None,
    }
    base.update(overrides)
    return make_device(**base)


def export_model(
    path: Path,
    *,
    scale: int = 1,
    height: int = 4,
    width: int = 4,
    dynamic: bool = False,
    padding_free: bool = False,
) -> Path:
    """Export a real graph: `Conv -> Relu`, plus `Resize` when `scale > 1`.

    The convolution is depthwise with weights of 1/9 and no bias, so it is a 3x3
    box mean *per colour channel* and nothing mixes the channels together.
    That makes a BGR/RGB swap visible in the channel test, and it makes a
    constant frame come back constant, so the tiling test can look for a seam
    without a second implementation of the maths to compare against.

    `padding_free` uses a 1x1 kernel instead, which leaves the spatial size
    alone. A padded kernel bakes zero padding into the graph, and a graph
    that pads puts its own dark band on the border of *every* tile — a
    property of the fixture, not of the tiling, so the tiling test uses a
    graph that does not have it.
    """
    import torch
    from torch import nn

    kernel = 1 if padding_free else 3
    conv_padding = 0 if padding_free else 1

    class BoxFilter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(
                3, 3, kernel, padding=conv_padding, bias=False, groups=3
            )
            with torch.no_grad():
                self.conv.weight.fill_(1.0 / (kernel * kernel))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = torch.relu(self.conv(x))
            if scale > 1:
                y = torch.nn.functional.interpolate(
                    y, scale_factor=scale, mode="nearest"
                )
            return y

    axes = None
    if dynamic:
        axes = {"input": {0: "b", 2: "h", 3: "w"}, "output": {0: "b", 2: "oh", 3: "ow"}}
    with torch.no_grad():
        torch.onnx.export(
            BoxFilter().eval(),
            torch.zeros(1, 3, height, width),
            str(path),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            dynamo=False,
            dynamic_axes=axes,
        )
    return path


#: What the `models` fixture returns: a path to a graph with the given options.
ModelFactory = Callable[..., Path]


@pytest.fixture(scope="session")
def models(tmp_path_factory: pytest.TempPathFactory) -> ModelFactory:
    """One export per distinct graph, shared by every test that asks for it.

    `torch.onnx.export` costs a second and a half apiece and most of the
    provider tests want the very same 4x4 graph, so the export is done once
    per distinct set of options rather than once per test. The files are
    written to one session directory and only ever read.
    """
    root = tmp_path_factory.mktemp("onnx-models")
    built: dict[tuple[tuple[str, Any], ...], Path] = {}

    def build(**options: Any) -> Path:
        key = tuple(sorted(options.items()))
        if key not in built:
            built[key] = export_model(root / f"model{len(built)}.onnx", **options)
        return built[key]

    return build


def box_filter(frame_bgr: np.ndarray) -> np.ndarray:
    """The graph's own maths in numpy: a zero-padded 3x3 mean per channel.

    ONNX `Conv` pads with zeros, not by replicating the edge, and a constant
    frame therefore comes back darker in a one-pixel border. Written out by
    hand rather than with torch so the assertion cannot be satisfied by the
    same code that produced the number.
    """
    values = frame_bgr.astype(np.float64) / 255.0
    padded = np.pad(values, ((1, 1), (1, 1), (0, 0)), mode="constant")
    total = np.zeros(values.shape, dtype=np.float64)
    for row in range(3):
        for col in range(3):
            total += padded[row : row + values.shape[0], col : col + values.shape[1], :]
    return total / 9.0 * 255.0


def loaded(
    model_path: Path,
    device: Device,
    *,
    precision: str = "fp32",
    tile_size: int = 0,
) -> OnnxBackend:
    """A backend with a real CPU session behind it."""
    backend = OnnxBackend(tile_size=tile_size)
    backend.load(model_path, precision, device)
    return backend


def test_fixture_is_a_real_three_op_graph(
    models: ModelFactory,
) -> None:
    model = models(scale=2, height=4, width=4)

    assert exported_ops(model) == {"Conv", "Relu", "Resize", "Constant"}


def test_infer_matches_a_numpy_reference(
    models: ModelFactory,
) -> None:
    model = models(height=6, width=8)
    backend = loaded(model, cpu_device())
    frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)

    out = backend.infer(frame)

    assert out.dtype == np.uint8
    assert out.shape == frame.shape
    np.testing.assert_allclose(
        out, box_filter(frame).astype(np.uint8), atol=TRUNCATION_SLACK
    )


def test_colour_channels_survive_the_bgr_round_trip(
    models: ModelFactory,
) -> None:
    model = models(height=5, width=5)
    backend = loaded(model, cpu_device())
    frame = np.zeros((5, 5, 3), dtype=np.uint8)
    frame[:, :, 0] = 200
    frame[:, :, 1] = 1
    frame[:, :, 2] = 40

    out = backend.infer(frame)

    # The centre pixel's 3x3 window is entirely inside the frame, so each
    # channel comes back at its own level. A BGR/RGB swap would put the 200 on
    # the blue plane instead.
    np.testing.assert_allclose(out[2, 2], [200, 1, 40], atol=TRUNCATION_SLACK)
    # The corner is where the graph's own zero padding shows, and it is darker
    # there for every channel alike.
    assert int(out[0, 0, 0]) < int(out[2, 2, 0])


def test_scale_is_learned_from_the_graph(
    models: ModelFactory,
) -> None:
    model = models(scale=2, height=4, width=4)
    backend = loaded(model, cpu_device())

    assert backend.scale == 2
    assert backend.infer(np.full((4, 4, 3), 90, dtype=np.uint8)).shape == (8, 8, 3)


def test_dynamic_graph_without_metadata_is_checked_at_inference(
    models: ModelFactory,
) -> None:
    model = models(scale=2, dynamic=True)
    backend = loaded(model, cpu_device())

    # Nothing in the graph says 2, so the backend assumes 1 and then owns the
    # discrepancy rather than writing a half-size video.
    assert backend.scale == 1
    with pytest.raises(UnsupportedScaleError) as excinfo:
        backend.infer(np.full((4, 4, 3), 90, dtype=np.uint8))
    assert "produced 2x" in str(excinfo.value)


def test_short_frame_is_padded_to_the_graph_and_cropped_back(
    models: ModelFactory,
) -> None:
    # A 1x1 kernel keeps the graph's own padding out of the way, so what is
    # measured is the backend padding up to the declared shape and cropping
    # back, and nothing else.
    model = models(height=8, width=8, padding_free=True)
    backend = loaded(model, cpu_device())

    out = backend.infer(np.full((5, 7, 3), 128, dtype=np.uint8))

    assert out.shape == (5, 7, 3)
    # Every row of the padded frame, including the ones the padding invented,
    # has to carry 128; a crop taken at the wrong offset would show a band.
    assert abs(int(out[0, 0, 0]) - 128) <= TRUNCATION_SLACK
    assert int(out.max()) - int(out.min()) <= TRUNCATION_SLACK


def test_tiled_infer_has_no_seam(
    models: ModelFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The graph is bigger than the tile, so all sixteen tiles are shorter than
    # the static input and each has to be padded up and cropped back — the path
    # a real 4x model takes on the last tile of every row and column.
    model = models(height=96, width=96, padding_free=True)
    monkeypatch.setattr(onnx_backend, "needs_tiling", lambda _h, _w: True)
    backend = loaded(model, cpu_device(), tile_size=64)

    out = backend.infer(np.full((192, 192, 3), 77, dtype=np.uint8))

    assert out.shape == (192, 192, 3)
    # Sixteen overlapping, feathered tiles of one constant value: a broken
    # weight normalisation or a mis-sized crop shows up as a jump far larger
    # than the one-level truncation slack.
    assert int(out.max()) - int(out.min()) <= TRUNCATION_SLACK
    assert abs(int(out[0, 0, 0]) - 77) <= TRUNCATION_SLACK
    assert abs(int(out[191, 191, 0]) - 77) <= TRUNCATION_SLACK


def test_cpu_reports_fp32_even_when_fp16_is_requested(
    models: ModelFactory,
) -> None:
    model = models()
    backend = loaded(model, cpu_device(), precision="fp16")

    assert backend.precision == "fp32"


class FakeSession:
    """Just enough of `InferenceSession` for a code path that never infers."""

    def __init__(self, providers: list[str], dtype: str = "tensor(float)") -> None:
        self._providers = providers
        self._dtype = dtype

    def get_inputs(self) -> list[_FakeNodeArg]:
        return [_FakeNodeArg(self._dtype, [1, 3, 4, 4])]

    def get_outputs(self) -> list[_FakeNodeArg]:
        return [_FakeNodeArg(self._dtype, [1, 3, 4, 4])]

    def get_providers(self) -> list[str]:
        return list(self._providers)


class _FakeNodeArg:
    def __init__(self, type_name: str, shape: list[int]) -> None:
        self.name = "input"
        self.type = type_name
        self.shape = shape


class RecordingOptions:
    """A `SessionOptions` that remembers which fields were assigned.

    The DirectML requirement is a *pair of documented fields*, so the contract
    being tested is which fields the backend touches, not merely what a session
    would end up holding.
    """

    def __init__(self) -> None:
        # The defaults are set with `object.__setattr__` so the recorder only
        # ever holds what the *backend* assigned, not what the constructor did.
        for name, value in (
            ("assigned", set()),
            ("graph_optimization_level", None),
            ("enable_mem_pattern", True),
            ("execution_mode", None),
        ):
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        if name != "assigned":
            object.__setattr__(self, "assigned", set(self.assigned) | {name})
        object.__setattr__(self, name, value)


class Fakes:
    """What the swapped-out `onnxruntime` attributes recorded."""

    def __init__(self) -> None:
        self.options: list[RecordingOptions] = []
        self.sessions: list[tuple[str, dict[str, Any]]] = []


def install_fake_ort(
    monkeypatch: pytest.MonkeyPatch,
    *,
    providers: list[str] | None = None,
    session_factory: Callable[..., Any] | None = None,
) -> Fakes:
    """Swap the `onnxruntime` attributes the backend reads at call time."""
    fakes = Fakes()

    def make_options() -> RecordingOptions:
        options = RecordingOptions()
        fakes.options.append(options)
        return options

    def make_session(path: str, **kwargs: Any) -> Any:
        fakes.sessions.append((path, kwargs))
        if session_factory is not None:
            return session_factory(path, **kwargs)
        return FakeSession(list(kwargs.get("providers") or []))

    monkeypatch.setattr(
        ort, "get_available_providers", lambda: providers or ALL_PROVIDERS
    )
    monkeypatch.setattr(ort, "SessionOptions", make_options)
    monkeypatch.setattr(ort, "InferenceSession", make_session)
    return fakes


def test_cpu_leaves_the_directml_options_alone(
    models: ModelFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = models()
    fakes = install_fake_ort(monkeypatch, providers=["CPUExecutionProvider"])

    loaded(model, cpu_device())

    assert fakes.options[0].assigned == {"graph_optimization_level"}


def test_directml_disables_mem_pattern_and_parallel_execution(
    models: ModelFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = models()
    fakes = install_fake_ort(
        monkeypatch, providers=["DmlExecutionProvider", "CPUExecutionProvider"]
    )

    loaded(model, make_device(vendor="intel", backend="onnx:dml"))

    options = fakes.options[0]
    assert options.enable_mem_pattern is False
    assert options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    assert options.assigned == {
        "graph_optimization_level",
        "enable_mem_pattern",
        "execution_mode",
    }


def test_providers_follow_the_device_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ort, "get_available_providers", lambda: ALL_PROVIDERS)

    assert providers_for(make_device(backend="onnx:cuda")) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert providers_for(make_device(backend="onnx:migraphx")) == [
        "MIGraphXExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert providers_for(make_device(backend="onnx:dml")) == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert providers_for(make_device(backend="onnx:coreml")) == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert providers_for(cpu_device()) == ["CPUExecutionProvider"]


def test_missing_provider_is_named_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["AzureExecutionProvider", "CPUExecutionProvider"],
    )

    with pytest.raises(BackendUnavailableError) as excinfo:
        providers_for(make_device(name="Tesla P40"))

    message = str(excinfo.value)
    assert "CUDAExecutionProvider" in message
    assert "AzureExecutionProvider" in message
    assert "Tesla P40" in message


def test_cuda_preloads_dlls_before_the_session_exists(
    tmp_path: Path,
    models: ModelFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = models()
    order: list[str] = []

    def factory(_path: str, **_kwargs: Any) -> FakeSession:
        order.append("session")
        return FakeSession(["CPUExecutionProvider"])

    install_fake_ort(
        monkeypatch,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        session_factory=factory,
    )
    monkeypatch.setattr(ort, "preload_dlls", lambda **_kw: order.append("preload_dlls"))
    monkeypatch.setattr(onnx_backend, "runtimes_dir", lambda: tmp_path / "runtimes")

    loaded(model, make_device())

    assert order == ["preload_dlls", "session"]


def test_cpu_never_preloads_dlls(
    models: ModelFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = models()
    fakes = install_fake_ort(monkeypatch, providers=["CPUExecutionProvider"])
    monkeypatch.setattr(
        ort, "preload_dlls", lambda **_kw: pytest.fail("CPU must not preload CUDA")
    )

    loaded(model, cpu_device())

    assert len(fakes.sessions) == 1


def test_migraphx_points_the_compiler_cache_at_the_runtime_dir(
    tmp_path: Path,
    models: ModelFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = models()
    runtime_root = tmp_path / "runtimes"
    monkeypatch.setattr(onnx_backend, "runtimes_dir", lambda: runtime_root)
    fakes = install_fake_ort(
        monkeypatch, providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"]
    )
    monkeypatch.delenv(MIGRAPHX_CACHE_ENV, raising=False)

    loaded(model, make_device(vendor="amd", backend="onnx:migraphx", index=2))

    cache = runtime_root / "migraphx-cache"
    assert os.environ[MIGRAPHX_CACHE_ENV] == str(cache)
    assert cache.is_dir()
    assert fakes.sessions[0][1]["provider_options"] == [{"device_id": "2"}]


def test_coreml_retries_with_neuralnetwork_when_mlprogram_is_refused(
    tmp_path: Path,
    models: ModelFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = models()
    monkeypatch.setattr(onnx_backend, "coreml_cache_dir", lambda: tmp_path / "mlcache")
    attempts: list[dict[str, Any]] = []

    def factory(_path: str, **kwargs: Any) -> FakeSession:
        attempts.append(dict(kwargs["provider_options"][0]))
        if len(attempts) == 1:
            raise RuntimeError(
                "Fail: Node(Conv) : MLProgram format is not supported on this macOS"
            )
        return FakeSession(["CoreMLExecutionProvider"])

    install_fake_ort(
        monkeypatch,
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        session_factory=factory,
    )
    logged: list[str] = []
    backend = OnnxBackend()
    backend.log = logged.append

    backend.load(model, "fp32", make_device(vendor="apple", backend="onnx:coreml"))

    assert COREML_RETRY_LOG in logged
    assert [attempt["ModelFormat"] for attempt in attempts] == [
        "MLProgram",
        "NeuralNetwork",
    ]
    assert attempts[1]["MLComputeUnits"] == "ALL"
    assert attempts[1]["RequireStaticInputShapes"] == "0"
    assert attempts[1]["EnableOnSubgraphs"] == "0"
    assert attempts[1]["ModelCacheDirectory"] == str(tmp_path / "mlcache")
    assert backend.session_providers() == ["CoreMLExecutionProvider"]
    assert backend.precision == "fp32"


def test_coreml_failure_beyond_the_retry_is_a_backend_error(
    tmp_path: Path,
    models: ModelFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = models()
    monkeypatch.setattr(onnx_backend, "coreml_cache_dir", lambda: tmp_path / "mlcache")

    def factory(_path: str, **_kwargs: Any) -> FakeSession:
        raise RuntimeError("InvalidArgument: MLProgram is unavailable")

    install_fake_ort(
        monkeypatch,
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        session_factory=factory,
    )
    backend = OnnxBackend()

    with pytest.raises(BackendUnavailableError) as excinfo:
        backend.load(model, "fp32", make_device(vendor="apple", backend="onnx:coreml"))

    assert model.name in str(excinfo.value)


def test_fp16_request_on_a_float32_graph_is_reported_not_claimed(
    models: ModelFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = models()
    install_fake_ort(
        monkeypatch, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    logged: list[str] = []
    backend = OnnxBackend()
    backend.log = logged.append

    backend.load(model, "fp16", make_device())

    assert backend.precision == "fp32"
    assert any("FP16 was requested" in line for line in logged)


def test_session_providers_reports_what_ort_actually_used(
    models: ModelFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = models()
    install_fake_ort(
        monkeypatch,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        session_factory=lambda *_a, **_k: FakeSession(["CPUExecutionProvider"]),
    )

    # ORT accepts a provider it cannot create and runs the graph on the CPU
    # while reporting success, so the honest report is the CPU one.
    assert loaded(model, make_device()).session_providers() == ["CPUExecutionProvider"]


def test_export_needs_the_frame_size(tmp_path: Path) -> None:
    backend = OnnxBackend()

    with pytest.raises(BackendUnavailableError) as excinfo:
        backend.load(tmp_path / "RealESRGAN_x4plus.pth", "fp32", make_device())

    assert "export_size=(height, width)" in str(excinfo.value)
    assert "RealESRGAN_x4plus.pth" in str(excinfo.value)


def test_checkpoint_is_exported_at_the_frame_size(
    tmp_path: Path,
    models: ModelFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = models(scale=2, height=8, width=8)
    checkpoint = tmp_path / "RealESRGAN_x4plus.pth"
    checkpoint.write_bytes(b"pretend state dict")
    seen: dict[str, Any] = {}

    def fake_export(
        pth: Path, out: Path, height: int, width: int, opset: int = 17
    ) -> Path:
        seen.update(pth=pth, out=out, height=height, width=width, opset=opset)
        return exported

    monkeypatch.setattr(onnx_backend, "export_to_onnx", fake_export)
    monkeypatch.setattr(
        onnx_backend,
        "export_cache_path",
        lambda pth, width, height, opset=17: (
            tmp_path / f"{pth.stem}-{width}x{height}.onnx"
        ),
    )

    backend = OnnxBackend(export_size=(8, 8))
    backend.load(checkpoint, "fp32", cpu_device())

    assert seen["height"] == 8
    assert seen["width"] == 8
    # The session is the export's output, so the scale is read from that file.
    assert backend.scale == 2
    assert seen["out"] == tmp_path / "RealESRGAN_x4plus-8x8.onnx"


def test_closer_forgets_the_session(
    models: ModelFactory,
) -> None:
    backend = loaded(models(), cpu_device())
    assert backend.session_providers()

    backend.close()

    assert backend.session_providers() == []


class ConcurrencyRecorder(OnnxBackend):
    """Reports whether two `infer` calls were ever inside at once."""

    def __init__(self) -> None:
        super().__init__()
        self.inside = 0
        self.overlapped = False
        self._guard = threading.Lock()

    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        with self._guard:
            self.inside += 1
            if self.inside > 1:
                self.overlapped = True
        time.sleep(0.02)
        with self._guard:
            self.inside -= 1
        return image_bgr


def test_infer_guarded_serialises_concurrent_calls() -> None:
    backend = ConcurrencyRecorder()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    threads = [
        threading.Thread(target=backend.infer_guarded, args=(frame,)) for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not backend.overlapped
    assert all(not thread.is_alive() for thread in threads)
