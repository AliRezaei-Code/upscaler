"""Export a PyTorch checkpoint to ONNX, and refuse an unusable graph.

Three measured facts drive this module; all three were read from the pinned
toolchain rather than assumed, and each replaced an assumption that turned out
to be wrong.

**`pixel_shuffle` becomes `DepthToSpace`, not a Reshape chain.** On torch
2.7.1 the exporter emits `DepthToSpace(mode="CRD", blocksize=r)` for
`aten::pixel_shuffle` at every opset from 11 to 20, with static or dynamic
spatial dims alike — measured here, not read from a version of the symbolic
file that predates it. The CoreML EP's MLProgram table supports "Only DCR mode
DepthToSpace"; its NeuralNetwork column accepts CRD but only for a fixed input
shape. So a CRD node either fails on the fast path or forces every macOS export
onto the slower fallback. `rewrite_depth_to_space` therefore replaces each CRD
node with the equivalent `Reshape → Transpose → Reshape` — the same
permutation, using only ops the CoreML table lists unconditionally, with the
shape vectors as initializers rather than `Constant` nodes, which the table
does not list at all.

**Static spatial dims.** They are concrete because CoreML and DirectML both
document that known shapes at session creation are materially faster, and
because they are what the rewrite above needs to build a constant shape vector.
Only the batch axis is dynamic, so an export is specific to one frame size —
which is why the cache key carries the size.

**`assert_ops_supported` is a refusal, not a warning.** A graph outside
`EMITTED_OPS_ALLOWLIST` is rejected at export time, because the alternative is
a CoreML provider that quietly falls back to the CPU and a job that runs for
hours at a small fraction of the speed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # torch is imported lazily; this is for the type checker only
    from torch import nn

from ..errors import BackendUnavailableError, ModelLoadError
from ..paths import models_dir

# 17, not the newest opset: DirectML supports opset 20 and nothing above it.
ONNX_OPSET = 17

# Bumped whenever the export or either graph rewrite changes. The cached file's
# *content* depends on this code, not only on the checkpoint, so a cache key
# without it would let a pre-rewrite file outlive the rewrite that was meant to
# replace it — and on a 14-minute export nobody would notice for a long time.
EXPORT_REVISION = 2

# Every op a graph in this app may contain, which is every op the CoreML EP's
# table lists that these architectures need. `DepthToSpace` is in the set
# because the table supports it in DCR mode; the CRD nodes PyTorch emits are
# rewritten away before this check runs, so a graph that reaches the check
# intact is one the table can run.
EMITTED_OPS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "Add",
        "Clip",
        "Concat",
        "Conv",
        "DepthToSpace",
        "Div",
        "Erf",
        "Gelu",
        "LayerNormalization",
        "LeakyRelu",
        "Mul",
        "Pad",
        "Resize",
        "Reshape",
        "Squeeze",
        "Transpose",
        "Unsqueeze",
    }
)

_INPUT_NAME = "input"
_OUTPUT_NAME = "output"


def _sha256_prefix(path: Path, length: int = 16) -> str:
    """The first `length` hex characters of a file's SHA-256, for a cache key."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def export_cache_path(
    pth_path: Path, width: int, height: int, opset: int = ONNX_OPSET
) -> Path:
    """Where the export for this checkpoint at this size is cached.

    The key carries the checkpoint's hash, the spatial size, the opset and
    `EXPORT_REVISION`, because all four change the graph and a stale hit would
    silently run a model built by older code or for a different frame.
    """
    name = (
        f"{_sha256_prefix(pth_path)}-{width}x{height}-op{opset}-r{EXPORT_REVISION}.onnx"
    )
    return models_dir() / ".onnx-cache" / name


def _load_torch_module(pth_path: Path) -> nn.Module:
    """Load a checkpoint with spandrel, which knows the state-dict layouts."""
    import spandrel

    try:
        descriptor = spandrel.ModelLoader().load_from_file(pth_path)
    except Exception as exc:
        raise ModelLoadError(f"Could not load {pth_path.name}: {exc}") from exc
    model = getattr(descriptor, "model", None)
    if model is None:  # pragma: no cover - spandrel always sets it or raises
        raise ModelLoadError(f"{pth_path.name} produced no model")
    return cast("nn.Module", model)


def export_to_onnx(
    pth_path: Path,
    out_path: Path,
    height: int,
    width: int,
    opset: int = ONNX_OPSET,
) -> Path:
    """Export `pth_path` to `out_path` for `height` x `width` frames.

    Cached: if `out_path` exists and is non-empty it is returned untouched,
    because a second export of a 67 MB checkpoint costs minutes.
    """
    if height < 1 or width < 1:
        raise ModelLoadError(
            f"Cannot export at {width}x{height}: dimensions must be positive"
        )
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path
    return export_model_to_onnx(
        _load_torch_module(pth_path), out_path, height, width, opset
    )


def export_model_to_onnx(
    model: nn.Module,
    out_path: Path,
    height: int,
    width: int,
    opset: int = ONNX_OPSET,
) -> Path:
    """Export an already-loaded torch module.

    Split from `export_to_onnx` so the exporter's contract can be tested against
    a model built in memory. spandrel — correctly — refuses a checkpoint that
    matches none of its 42 architectures, and waiting for it to say so costs
    seconds per attempt, which is a poor way to test an ONNX exporter.
    """
    import torch

    if height < 1 or width < 1:
        raise ModelLoadError(
            f"Cannot export at {width}x{height}: dimensions must be positive"
        )
    module = model
    module.eval()
    dummy = torch.zeros(1, 3, height, width, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staging = out_path.with_suffix(".onnx.part")
    torch.onnx.export(
        module,
        # torch's annotation for `args` is a tuple, but the documented and
        # implemented call passes the example tensor directly.
        cast("Any", dummy),
        str(staging),
        opset_version=opset,
        input_names=[_INPUT_NAME],
        output_names=[_OUTPUT_NAME],
        dynamic_axes={_INPUT_NAME: {0: "b"}, _OUTPUT_NAME: {0: "b"}},
        do_constant_folding=True,
    )
    staging.replace(out_path)
    rewrite_depth_to_space(out_path)
    fold_constant_nodes(out_path)
    assert_ops_supported(out_path)
    return out_path


def fold_constant_nodes(model_path: Path) -> int:
    """Move `Constant` nodes' values into the graph's initializers.

    A `Constant` is a node; an initializer is not. Every execution provider
    handles initializers, and the CoreML EP's supported-op table does not list
    `Constant` at all — torch emits one inside the real `RealESRGAN_x4plus`
    graph, and without this step the allow-list check refuses a model the
    machine could in fact run.

    Folding is pure relabelling: the value moves to `graph.initializer` and
    takes the *node's output* name, because a `Constant` node's value usually
    carries no name of its own while every consumer already refers to the
    output. An initializer filed under the empty name loads, and then fails
    with "node input is not a graph input, initializer, or output of a previous
    node" — a message that points at the graph, not at the fold.
    Returns the number of nodes folded; a `Constant` with no tensor `value`
    attribute is left alone, and the allow-list check then names it.
    """
    import onnx

    model = onnx.load(str(model_path))
    graph = model.graph
    folded = 0
    keep: list[Any] = []
    for node in graph.node:
        attribute = _attribute(node, "value")
        if node.op_type != "Constant" or attribute is None or not node.output:
            keep.append(node)
            continue
        # A sparse value has no dense TensorProto to move, and the allow-list
        # check then names the node instead of a graph nobody can run.
        if _attribute(node, "sparse_value") is not None or not attribute.HasField("t"):
            keep.append(node)
            continue
        existing = {initializer.name for initializer in graph.initializer}
        if node.output[0] in existing:
            keep.append(node)
            continue
        tensor = onnx.TensorProto()
        tensor.CopyFrom(attribute.t)
        tensor.name = node.output[0]
        graph.initializer.append(tensor)
        folded += 1
    if folded:
        del graph.node[:]
        graph.node.extend(keep)
        onnx.save(model, str(model_path))
    return folded


def _dims(value_info: object) -> tuple[int | None, ...]:
    """A value's dims, with `None` wherever the shape is symbolic.

    The batch axis of an export is symbolic on purpose, so a helper that
    demanded every dim be concrete would refuse exactly the graphs this rewrite
    exists to fix.
    """
    shape = value_info.type.tensor_type.shape  # type: ignore[attr-defined]
    return tuple(
        int(dim.dim_value) if dim.HasField("dim_value") else None for dim in shape.dim
    )


def _attribute(node: object, name: str) -> Any:
    for attribute in node.attribute:  # type: ignore[attr-defined]
        if attribute.name == name:
            return attribute
    return None


def rewrite_depth_to_space(model_path: Path) -> int:
    """Replace CRD `DepthToSpace` nodes with the equivalent Reshape chain.

    Returns the number of nodes rewritten, so a caller — and a test — can tell
    the difference between "nothing to do" and "the rewrite silently did not
    apply". A node is left alone when it is DCR (which the CoreML MLProgram
    table supports), when its blocksize is missing, or when its input shape is
    not fully concrete: the vector a `Reshape` needs cannot be built from a
    symbolic shape, and `assert_ops_supported` will then refuse the graph with
    a message that says which op is the problem.
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(model_path))
    graph = model.graph
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    shapes = {value.name: _dims(value) for value in inferred.graph.value_info}

    rewritten = 0
    for node in list(graph.node):
        if node.op_type != "DepthToSpace":
            continue
        mode_attribute = _attribute(node, "mode")
        mode = mode_attribute.s.decode() if mode_attribute is not None else "DCR"
        blocksize_attribute = _attribute(node, "blocksize")
        if mode != "CRD" or blocksize_attribute is None:
            continue
        r = int(blocksize_attribute.i)
        dims = shapes.get(node.input[0])
        if dims is None or len(dims) != 4 or r < 1:
            continue
        _batch, channels, height, width = dims
        if channels is None or height is None or width is None or channels % (r * r):
            continue
        out_channels = channels // (r * r)
        reshape_in = f"{node.name}_reshape_in"
        transpose_out = f"{node.name}_transpose"
        replacements = [
            helper.make_node(
                "Reshape",
                [node.input[0], reshape_in],
                [transpose_out],
                name=f"{node.name}_reshape_1",
            ),
            helper.make_node(
                "Transpose",
                [transpose_out],
                ["__perm__" + node.name],
                name=f"{node.name}_transpose_node",
                perm=[0, 1, 4, 2, 5, 3],
            ),
            helper.make_node(
                "Reshape",
                ["__perm__" + node.name, f"{node.name}_reshape_out"],
                list(node.output),
                name=f"{node.name}_reshape_2",
            ),
        ]
        index = list(graph.node).index(node)
        del graph.node[index]
        for offset, replacement in enumerate(replacements):
            graph.node.insert(index + offset, replacement)
        for name, values in (
            (reshape_in, [0, out_channels, r, r, height, width]),
            (f"{node.name}_reshape_out", [0, out_channels, height * r, width * r]),
        ):
            tensor = numpy_helper.from_array(
                np.array(values, dtype=np.int64), name=name
            )
            graph.initializer.append(tensor)
        rewritten += 1
    if rewritten:
        onnx.save(model, str(model_path))
    return rewritten


def _op_types(model_path: Path) -> set[str]:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    return {node.op_type for node in model.graph.node}


def assert_ops_supported(model_path: Path) -> None:
    """Refuse a graph that uses an op outside `EMITTED_OPS_ALLOWLIST`.

    The message names the offending ops, because "the CoreML provider silently
    fell back to CPU" is the failure this check exists to prevent.
    """
    unknown = sorted(_op_types(model_path) - EMITTED_OPS_ALLOWLIST)
    if unknown:
        raise BackendUnavailableError(
            f"{model_path.name} uses ops no supported execution provider accepts: "
            f"{', '.join(unknown)}"
        )


def graph_spatial_shape(model_path: Path) -> tuple[int, int] | None:
    """The concrete `(height, width)` of a graph input, or `None` if dynamic."""
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    graph_input = model.graph.input[0]
    dims = graph_input.type.tensor_type.shape.dim
    if len(dims) < 4:
        return None
    height, width = dims[2], dims[3]
    if not height.HasField("dim_value") or not width.HasField("dim_value"):
        return None
    return int(height.dim_value), int(width.dim_value)


def scale_from_graph(model_path: Path) -> int | None:
    """The output scale implied by a graph's input and output shapes.

    `None` when either side is dynamic or the graph has no spatial rank — the
    caller then has to learn the scale another way rather than guess it.
    """
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    if not model.graph.input or not model.graph.output:
        return None
    in_dims = model.graph.input[0].type.tensor_type.shape.dim
    out_dims = model.graph.output[0].type.tensor_type.shape.dim
    if len(in_dims) < 4 or len(out_dims) < 4:
        return None
    if not all(
        dim.HasField("dim_value")
        for dim in (in_dims[2], in_dims[3], out_dims[2], out_dims[3])
    ):
        return None
    in_height, in_width = int(in_dims[2].dim_value), int(in_dims[3].dim_value)
    out_height, out_width = int(out_dims[2].dim_value), int(out_dims[3].dim_value)
    if in_height < 1 or in_width < 1:
        return None
    if out_height % in_height or out_width % in_width:
        return None
    vertical = out_height // in_height
    horizontal = out_width // in_width
    if vertical != horizontal:
        return None
    return vertical


def scale_from_metadata(model_path: Path) -> int | None:
    """The `scale` key from the ONNX metadata props, if the exporter wrote one."""
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    for entry in model.metadata_props:
        if entry.key == "scale":
            try:
                value = int(entry.value)
            except ValueError:
                return None
            return value if value >= 1 else None
    return None


def exported_ops(model_path: Path) -> set[str]:
    """The op types a graph uses. Exposed for tests and diagnostics."""
    return _op_types(model_path)
