"""The ONNX exporter, and the CoreML rewrite it depends on.

Two halves:

* a **synthetic** model built in memory, so CI can prove the exporter's
  contract — concrete spatial dims, dynamic batch, ops inside the allow-list,
  and the `DepthToSpace` rewrite — without a 67 MB download;
* the **real** `RealESRGAN_x4plus.pth`, marked `gpu`, which is the check that
  settles whether the plan's `DepthToSpace` reasoning was right. It was not:
  torch 2.7.1 emits a CRD `DepthToSpace`, and the CoreML MLProgram path
  supports DCR only, so `export.py` rewrites the node. The marker is the plan's
  and it is also the truth about CI — the checkpoint is not in the repository.

The checkpoint is located with `UPSCALER_TEST_MODEL`, then a short list of the
places it has lived on the machine this was built on. It skips with that reason
rather than failing when it is absent.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn

from core.backends.export import (
    EMITTED_OPS_ALLOWLIST,
    EXPORT_REVISION,
    ONNX_OPSET,
    assert_ops_supported,
    export_cache_path,
    export_model_to_onnx,
    export_to_onnx,
    exported_ops,
    fold_constant_nodes,
    graph_spatial_shape,
    rewrite_depth_to_space,
    scale_from_graph,
    scale_from_metadata,
)
from core.errors import BackendUnavailableError, ModelLoadError

HEIGHT, WIDTH = 288, 384
SMALL = 64
# A graph built by hand defaults to onnx 1.23's IR version, which onnxruntime 1.22
# refuses to load. torch stamps its own (older) value, so only these need pinning.
ORT_MAX_IR_VERSION = 10
# Tracing the real 23-block model at 288x384 costs minutes, not seconds.
EXPORT_TIMEOUT = 1800

CANDIDATE_MODELS = (
    "/media/ali0rez/ext4-Linux/mosaferan-mahtab/RealESRGAN_x4plus.pth",
    "/media/ali0rez/ext4-Linux/upscaler/models/RealESRGAN_x4plus.pth",
)


def _real_checkpoint() -> Path:
    from_env = os.environ.get("UPSCALER_TEST_MODEL")
    candidates = ([Path(from_env)] if from_env else []) + [
        Path(path) for path in CANDIDATE_MODELS
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip(
        "no real checkpoint: set UPSCALER_TEST_MODEL or place RealESRGAN_x4plus.pth "
        f"in one of {CANDIDATE_MODELS}"
    )


class _PixelShuffleNet(nn.Module):
    """The smallest thing that exercises the op the exporter cares about.

    `pixel_shuffle` is what Real-ESRGAN ends in, and it is the node whose
    exported form the whole CoreML question turns on.
    """

    def __init__(self, scale: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3 * scale * scale, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pixel_shuffle(self.conv(x))


def _raw_export(model: nn.Module, out: Path, size: int = SMALL) -> Path:
    """`torch.onnx.export` with the same arguments `export.py` uses."""
    torch.onnx.export(
        model,
        # torch annotates `args` as a tuple but takes the example tensor.
        cast("Any", torch.zeros(1, 3, size, size)),
        str(out),
        opset_version=ONNX_OPSET,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "b"}, "output": {0: "b"}},
        do_constant_folding=True,
    )
    return out


@pytest.fixture(scope="session")
def synthetic_onnx(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One 4x synthetic export, shared by every test that inspects the graph."""
    out = tmp_path_factory.mktemp("synthetic") / "tiny4x.onnx"
    yield export_model_to_onnx(_PixelShuffleNet(scale=4), out, SMALL, SMALL)


@pytest.fixture(scope="session")
def synthetic_onnx_2x(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same model at 2x, for the scale a graph implies."""
    out = tmp_path_factory.mktemp("synthetic2x") / "tiny2x.onnx"
    return export_model_to_onnx(_PixelShuffleNet(scale=2), out, SMALL, SMALL)


@pytest.fixture(scope="session")
def real_onnx() -> Iterator[Path]:
    """The real checkpoint, exported once for the whole session.

    The production cache, so a local re-run of this 14-minute test is a cache
    hit. Bump `EXPORT_REVISION` when the export changes, or this test would keep
    asserting against a file an older revision produced.
    """
    checkpoint = _real_checkpoint()
    yield export_to_onnx(
        checkpoint, export_cache_path(checkpoint, WIDTH, HEIGHT), HEIGHT, WIDTH
    )


# --- the synthetic half (runs in CI) ------------------------------------------


def test_an_export_has_concrete_spatial_dims(synthetic_onnx: Path) -> None:
    assert graph_spatial_shape(synthetic_onnx) == (SMALL, SMALL)


def test_the_batch_axis_stays_dynamic(synthetic_onnx: Path) -> None:
    import onnx

    graph = onnx.load(str(synthetic_onnx)).graph
    assert graph.input[0].type.tensor_type.shape.dim[0].dim_param == "b"
    assert graph.output[0].type.tensor_type.shape.dim[0].dim_param == "b"


def test_the_export_uses_the_documented_opset(synthetic_onnx: Path) -> None:
    import onnx

    opsets = {
        (entry.domain or "ai.onnx"): entry.version
        for entry in onnx.load(str(synthetic_onnx)).opset_import
    }
    assert opsets["ai.onnx"] == ONNX_OPSET
    assert ONNX_OPSET <= 20, "DirectML's ceiling is opset 20"


def test_every_emitted_op_is_inside_the_allow_list(synthetic_onnx: Path) -> None:
    assert exported_ops(synthetic_onnx) <= EMITTED_OPS_ALLOWLIST


def test_a_crd_depth_to_space_is_rewritten_for_coreml(synthetic_onnx: Path) -> None:
    """PyTorch's `pixel_shuffle` is CRD; the CoreML MLProgram path is DCR only.

    So an exported graph must contain no `DepthToSpace` at all, and must contain
    the equivalent `Reshape → Transpose(perm=[0,1,4,2,5,3]) → Reshape`.
    """
    import onnx

    assert "DepthToSpace" not in exported_ops(synthetic_onnx)
    graph = onnx.load(str(synthetic_onnx)).graph
    perms = [
        list(node.attribute[0].ints)
        for node in graph.node
        if node.op_type == "Transpose" and len(node.attribute) == 1
    ]
    assert [0, 1, 4, 2, 5, 3] in perms, "the pixel-shuffle permutation is missing"


def test_the_rewrite_uses_initializers_not_constant_nodes(synthetic_onnx: Path) -> None:
    """`Constant` is absent from the CoreML EP's op table; initializers are free."""
    assert "Constant" not in exported_ops(synthetic_onnx)


def test_the_rewrite_is_numerically_identical(tmp_path: Path) -> None:
    """The whole point of the rewrite: same numbers, ops the table accepts."""
    import onnxruntime as ort

    torch.manual_seed(0)
    model = _PixelShuffleNet(scale=4)
    raw = _raw_export(model, tmp_path / "raw.onnx")
    reference = _raw_export(model, tmp_path / "reference.onnx")
    assert "DepthToSpace" in exported_ops(raw), "torch no longer emits DepthToSpace"
    assert rewrite_depth_to_space(raw) == 1
    assert "DepthToSpace" not in exported_ops(raw)

    frame = np.random.default_rng(7).random((1, 3, SMALL, SMALL)).astype(np.float32)
    before = ort.InferenceSession(
        str(reference), providers=["CPUExecutionProvider"]
    ).run(None, {"input": frame})[0]
    after = ort.InferenceSession(str(raw), providers=["CPUExecutionProvider"]).run(
        None, {"input": frame}
    )[0]
    assert before.shape == (1, 3, SMALL * 4, SMALL * 4)
    assert np.array_equal(before, after), "the rewrite changed the numbers"


def test_a_constant_node_is_folded_into_an_initializer(tmp_path: Path) -> None:
    """`Constant` is a node the CoreML table does not list; an initializer is free."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    data = numpy_helper.from_array(
        np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2), name="x"
    )
    shape = numpy_helper.from_array(np.array([1, 4], dtype=np.int64), name="shape")
    constant = helper.make_node(
        "Constant",
        [],
        ["shape"],
        name="const",
        value=shape,
    )
    graph = helper.make_graph(
        [constant, helper.make_node("Reshape", ["x", "shape"], ["y"], name="r")],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        [data],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", ONNX_OPSET)]
    )
    model.ir_version = ORT_MAX_IR_VERSION
    onnx.checker.check_model(model)
    path = tmp_path / "constant.onnx"
    onnx.save(model, str(path))

    assert fold_constant_nodes(path) == 1
    assert "Constant" not in exported_ops(path)
    assert [init.name for init in onnx.load(str(path)).graph.initializer] == [
        "x",
        "shape",
    ]
    assert_ops_supported(path)

    import onnxruntime as ort

    result = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        None, {"x": np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)}
    )[0]
    assert result.shape == (1, 4)


def test_a_graph_without_constants_is_left_alone(synthetic_onnx: Path) -> None:
    before = exported_ops(synthetic_onnx)
    assert fold_constant_nodes(synthetic_onnx) == 0
    assert exported_ops(synthetic_onnx) == before


def test_a_dcr_depth_to_space_is_left_alone(tmp_path: Path) -> None:
    """DCR is what the CoreML table supports, so it must survive untouched."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    data = numpy_helper.from_array(
        np.arange(12 * 4, dtype=np.float32).reshape(1, 12, 2, 2), name="x"
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "DepthToSpace", ["x"], ["y"], blocksize=2, mode="DCR", name="dcr"
            )
        ],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 12, 2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])],
        [data],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", ONNX_OPSET)]
    )
    model.ir_version = ORT_MAX_IR_VERSION
    onnx.checker.check_model(model)
    path = tmp_path / "dcr.onnx"
    onnx.save(model, str(path))
    assert rewrite_depth_to_space(path) == 0
    assert "DepthToSpace" in exported_ops(path)


def test_a_session_runs_the_exported_graph(synthetic_onnx: Path) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(synthetic_onnx), providers=["CPUExecutionProvider"]
    )
    assert session.get_inputs()[0].shape == ["b", 3, SMALL, SMALL]
    for batch in (1, 2):
        frame = (
            np.random.default_rng(batch)
            .random((batch, 3, SMALL, SMALL))
            .astype(np.float32)
        )
        result = session.run(None, {session.get_inputs()[0].name: frame})[0]
        assert result.shape == (batch, 3, SMALL * 4, SMALL * 4)


def test_assert_ops_supported_accepts_an_allowed_graph(synthetic_onnx: Path) -> None:
    assert_ops_supported(synthetic_onnx)


def test_assert_ops_supported_rejects_an_unknown_op(tmp_path: Path) -> None:
    import onnx
    from onnx import helper

    graph = helper.make_graph(
        [helper.make_node("Sin", ["x"], ["y"], name="unsupported")],
        "g",
        [helper.make_tensor_value_info("x", 1, [1])],
        [helper.make_tensor_value_info("y", 1, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    path = tmp_path / "sin.onnx"
    onnx.save(model, str(path))
    with pytest.raises(BackendUnavailableError, match="Sin"):
        assert_ops_supported(path)


def test_a_second_export_of_the_same_file_is_a_cache_hit(tmp_path: Path) -> None:
    out = tmp_path / "tiny.onnx"
    first = export_model_to_onnx(_PixelShuffleNet(), out, SMALL, SMALL)
    stamp = first.stat().st_mtime_ns
    second = export_to_onnx(tmp_path / "unused.pth", out, SMALL, SMALL)
    assert second == first
    assert second.stat().st_mtime_ns == stamp, "a cache hit rewrote the file"


def test_the_cache_key_depends_on_checkpoint_size_and_opset(tmp_path: Path) -> None:
    a = tmp_path / "a.pth"
    a.write_bytes(b"a" * 4096)
    b = tmp_path / "b.pth"
    b.write_bytes(b"b" * 4096)
    assert export_cache_path(a, WIDTH, HEIGHT) != export_cache_path(b, WIDTH, HEIGHT)
    assert export_cache_path(a, WIDTH, HEIGHT) != export_cache_path(
        a, WIDTH, HEIGHT + 8
    )
    assert export_cache_path(a, WIDTH, HEIGHT).name.endswith(
        f"-{WIDTH}x{HEIGHT}-op{ONNX_OPSET}-r{EXPORT_REVISION}.onnx"
    )
    assert export_cache_path(a, WIDTH, HEIGHT, opset=18).name.endswith(
        f"-op18-r{EXPORT_REVISION}.onnx"
    )


def test_a_failed_export_leaves_no_partial_file_behind(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError):
        export_to_onnx(tmp_path / "absent.pth", tmp_path / "out.onnx", SMALL, SMALL)
    assert not list(tmp_path.glob("*.part"))


def test_exporting_refuses_a_zero_dimension(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="dimensions must be positive"):
        export_model_to_onnx(_PixelShuffleNet(), tmp_path / "tiny.onnx", 0, SMALL)


def test_exporting_a_checkpoint_spandrel_cannot_read_says_so(tmp_path: Path) -> None:
    broken = tmp_path / "not-a-model.pth"
    broken.write_bytes(b"\x00" * 4096)
    with pytest.raises(ModelLoadError, match="Could not load"):
        export_to_onnx(broken, tmp_path / "out.onnx", SMALL, SMALL)


def test_scale_from_metadata_is_none_without_the_key(synthetic_onnx: Path) -> None:
    assert scale_from_metadata(synthetic_onnx) is None


# --- the real checkpoint (deselected in CI) -----------------------------------


@pytest.mark.gpu
@pytest.mark.timeout(EXPORT_TIMEOUT)
def test_the_real_model_exports_at_a_known_shape(real_onnx: Path) -> None:
    assert graph_spatial_shape(real_onnx) == (HEIGHT, WIDTH)


@pytest.mark.gpu
@pytest.mark.timeout(EXPORT_TIMEOUT)
def test_the_real_model_is_a_4x_model(real_onnx: Path) -> None:
    assert scale_from_graph(real_onnx) == 4


@pytest.mark.gpu
@pytest.mark.timeout(EXPORT_TIMEOUT)
def test_every_op_the_real_model_emits_is_inside_the_allow_list(
    real_onnx: Path,
) -> None:
    ops = exported_ops(real_onnx)
    assert ops, "the graph has no nodes at all"
    assert ops <= EMITTED_OPS_ALLOWLIST, (
        f"outside the allow-list: {sorted(ops - EMITTED_OPS_ALLOWLIST)}"
    )
    assert "Conv" in ops
    assert "DepthToSpace" not in ops, "a CRD DepthToSpace survived the rewrite"


@pytest.mark.gpu
@pytest.mark.timeout(EXPORT_TIMEOUT)
def test_the_real_exported_model_runs_and_quadruples_the_frame(real_onnx: Path) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(real_onnx), providers=["CPUExecutionProvider"])
    frame = np.random.default_rng(0).random((1, 3, HEIGHT, WIDTH)).astype(np.float32)
    result = session.run(None, {session.get_inputs()[0].name: frame})[0]
    assert result.shape == (1, 3, HEIGHT * 4, WIDTH * 4)
    assert np.isfinite(result).all()


def test_prelu_is_allowed_because_the_coreml_table_allows_it(
    synthetic_onnx: Path,
) -> None:
    """The compact Real-ESRGAN models export `PRelu`, and the table lists it.

    Found by running the catalogue's own recommended model for a wedged P40:
    `realesr-general-x4v3` refused to export with "ops no supported execution
    provider accepts: PRelu", because PRelu was missing from the allow-list. The
    CoreML EP's table carries it with one condition — "Input slope should be
    constant. Input slope should either have shape [C, 1, 1] or have 1 element"
    — which is exactly the form the compact models use.
    """
    assert "PRelu" in EMITTED_OPS_ALLOWLIST


def test_a_computed_prelu_slope_is_refused(tmp_path: Path) -> None:
    """Being in the table is not the same as being runnable: the slope does."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    data = numpy_helper.from_array(np.zeros((1, 3, 4, 4), dtype=np.float32), name="x")
    ones = numpy_helper.from_array(np.ones((3, 1, 1), dtype=np.float32), name="ones")
    twos = numpy_helper.from_array(
        np.full((3, 1, 1), 2.0, dtype=np.float32), name="twos"
    )
    graph = helper.make_graph(
        [
            # Every op here is in the allow-list, so the refusal can only be
            # about the slope being computed rather than constant.
            helper.make_node("Mul", ["ones", "twos"], ["slope"], name="mul"),
            helper.make_node("PRelu", ["x", "slope"], ["y"], name="pr"),
        ],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])],
        [data, ones, twos],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", ONNX_OPSET)]
    )
    model.ir_version = ORT_MAX_IR_VERSION
    onnx.checker.check_model(model)
    path = tmp_path / "computed_prelu.onnx"
    onnx.save(model, str(path))
    with pytest.raises(BackendUnavailableError, match="PRelu whose slope is computed"):
        assert_ops_supported(path)


def test_a_constant_prelu_slope_is_accepted(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    data = numpy_helper.from_array(np.zeros((1, 3, 4, 4), dtype=np.float32), name="x")
    slope = numpy_helper.from_array(np.ones((3, 1, 1), dtype=np.float32), name="slope")
    graph = helper.make_graph(
        [helper.make_node("PRelu", ["x", "slope"], ["y"], name="pr")],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])],
        [data, slope],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", ONNX_OPSET)]
    )
    model.ir_version = ORT_MAX_IR_VERSION
    onnx.checker.check_model(model)
    path = tmp_path / "constant_prelu.onnx"
    onnx.save(model, str(path))
    assert_ops_supported(path)
