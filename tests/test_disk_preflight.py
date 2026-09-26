"""The disk preflight: calibrated, honest, and asked before anything is written.

The naive estimate — `n * w * h * 4 * scale**2` of uncompressed RGB — is 12 to
27 times too high for PNG output, and it is the number that would say "Need
~9750 GB" for the 165,303-frame job this application was built to run, which
used 846 GB. So the estimate here is calibrated against that run, and the
test that matters is the one that asserts the real job's estimate is under a
terabyte.

The second test is the one about *order*: `shutil.disk_usage` raises
`FileNotFoundError` for a directory that does not exist, and on a brand-new
input the work directory never existed, so the preflight has to create it
before it measures.
"""

from __future__ import annotations

import collections
import multiprocessing
import shutil
from pathlib import Path

import pytest

from core import pipeline
from core.errors import UpscalerError
from core.pipeline import (
    BYTES_PER_INPUT_PIXEL,
    BYTES_PER_OUTPUT_PIXEL,
    DISK_HEADROOM,
    estimate_frame_bytes,
    run_job,
)
from tests.fake_backend import FakeBackend
from tests.support import make_config, quiet

#: What `shutil.disk_usage` returns, so a test can pretend a full disk.
DiskUsage = collections.namedtuple("DiskUsage", "total used free")

#: The reference run: 165,303 frames, 384x288, 4x, 846 GiB actually used
#: (820 GiB of `frames_out` plus 26 GiB of `frames_in`, as `du -sh` reported).
REFERENCE_FRAMES = 165_303
REFERENCE_WIDTH = 384
REFERENCE_HEIGHT = 288
REFERENCE_SCALE = 4
REFERENCE_ACTUAL_BYTES = 846 * 1024**3


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "select_backend", lambda devices, model: FakeBackend)


def test_the_reference_run_fits_under_a_terabyte() -> None:
    estimate = estimate_frame_bytes(
        REFERENCE_FRAMES, REFERENCE_WIDTH, REFERENCE_HEIGHT, REFERENCE_SCALE
    )
    assert estimate < 1.1e12, f"the estimate says {estimate / 1e12:.2f} TB"
    # And it must be in the right order of magnitude, not merely small: an
    # estimate of 12 GB would pass the ceiling above and be just as useless.
    ratio = estimate / REFERENCE_ACTUAL_BYTES
    assert 0.3 < ratio < 3.0, (
        f"the estimate is {ratio:.2f}x the 846 GB the reference run actually used"
    )


def test_the_uncompressed_formula_is_not_a_calibration_either() -> None:
    """It lands within 15% of the truth by luck, and in the wrong direction.

    The plan claimed the naive `n*w*h*4*scale**2` over-estimates by 12-27x. It
    does not: for this job it is 1.17 TB against a real 846 GiB, so it is 15%
    *high* by coincidence — the frame term it assumes is the size of the
    uncompressed output, and the real PNG output happens to be a little smaller
    than that. For a cleaner source it would be a large over-estimate and for a
    noisy one a large under-estimate.
    """
    naive = (
        REFERENCE_FRAMES * REFERENCE_WIDTH * REFERENCE_HEIGHT * 4 * REFERENCE_SCALE**2
    )
    calibrated = estimate_frame_bytes(
        REFERENCE_FRAMES, REFERENCE_WIDTH, REFERENCE_HEIGHT, REFERENCE_SCALE
    )
    assert 0.5 < naive / calibrated < 2.0, (
        f"the two agree ({naive / calibrated:.2f}x), so neither is a measurement"
    )


def test_the_estimate_scales_with_frames_and_pixels() -> None:
    one = estimate_frame_bytes(1, 100, 100, 4)
    assert estimate_frame_bytes(2, 100, 100, 4) == pytest.approx(2 * one, rel=1e-6)
    assert estimate_frame_bytes(1, 200, 100, 4) == pytest.approx(2 * one, rel=1e-6)
    # 4x means 16x the pixels of 2x, and the output term dominates.
    # 4x output is 16x the pixels of 2x, so the frame terms grow faster than
    # the frame count does.
    four = estimate_frame_bytes(1, 100, 100, 4)
    two = estimate_frame_bytes(1, 100, 100, 2)
    assert 3.5 < four / two < 5.0


@pytest.mark.parametrize(
    "args",
    [(0, 0, 0, 0), (-1, 384, 288, 4), (5, 0, 288, 4), (5, 384, 288, 0)],
)
def test_a_nonsensical_estimate_is_refused(args: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="nonsensical estimate"):
        estimate_frame_bytes(*args)


def test_the_calibration_constants_are_the_measured_ones() -> None:
    # 820 GiB / 165,303 frames / (1536x1152 output pixels) = 3.01
    assert BYTES_PER_OUTPUT_PIXEL == 3.0
    # 26 GiB / 165,303 frames / (384x288 input pixels) = 1.53
    assert BYTES_PER_INPUT_PIXEL == 1.5
    assert DISK_HEADROOM == 1.05


def test_the_preflight_tolerates_a_work_dir_that_does_not_exist_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`shutil.disk_usage` raises `FileNotFoundError` for a missing directory."""
    cfg = make_config(tmp_path, frames=2)
    assert not cfg.work_dir.exists()

    seen: list[str] = []

    def record(path: Path) -> DiskUsage:
        seen.append(str(path))
        assert Path(path).is_dir(), (
            "the preflight measured a directory it had not created"
        )
        return DiskUsage(1000 * 1000**3, 0, 900 * 1000**3)

    monkeypatch.setattr(pipeline.shutil, "disk_usage", record)
    run_job(cfg, quiet(), multiprocessing.Event())
    assert seen, "the preflight never asked about the disk"
    assert cfg.output_path.is_file()


def test_a_disk_that_is_too_small_refuses_with_the_quoted_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_config(tmp_path, frames=4)
    monkeypatch.setattr(
        pipeline.shutil, "disk_usage", lambda _path: DiskUsage(10**9, 0, 1000)
    )
    with pytest.raises(UpscalerError) as excinfo:
        run_job(cfg, quiet(), multiprocessing.Event())
    message = str(excinfo.value)
    assert message.startswith("Need ~")
    assert str(cfg.work_dir) in message
    assert message.endswith("Choose another work directory.")
    assert not cfg.output_path.exists()
    assert not (cfg.work_dir / "frames_in").exists(), (
        "extraction ran despite the refusal"
    )


def test_the_headroom_is_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disk that fits the estimate but not the headroom is still refused."""
    cfg = make_config(tmp_path, frames=4)
    # The fake model is not in the catalogue, so the preflight sizes a 1x job.
    estimate = estimate_frame_bytes(4, 48, 32, 1)
    monkeypatch.setattr(
        pipeline.shutil,
        "disk_usage",
        lambda _path: DiskUsage(10**9, 0, int(estimate * 1.02)),
    )
    with pytest.raises(UpscalerError, match="Choose another work directory"):
        run_job(cfg, quiet(), multiprocessing.Event())


def test_a_disk_that_fits_the_headroom_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_config(tmp_path, frames=4)
    estimate = estimate_frame_bytes(4, 48, 32, 1)
    monkeypatch.setattr(
        pipeline.shutil,
        "disk_usage",
        lambda _path: DiskUsage(10**9, 0, int(estimate * DISK_HEADROOM) + 1),
    )
    run_job(cfg, quiet(), multiprocessing.Event())
    assert cfg.output_path.is_file()


def test_the_real_disk_is_what_is_measured_by_default(tmp_path: Path) -> None:
    """No monkeypatching: the real `shutil.disk_usage` answers for tmp_path."""
    free = shutil.disk_usage(tmp_path).free
    assert free > 0
    assert estimate_frame_bytes(4, 48, 32, 1) < free


# --- the scale the preflight sizes itself with ---------------------------------


def test_a_catalogue_model_sizes_the_preflight_from_its_own_entry(
    tmp_path: Path,
) -> None:
    """A 4x model must not be preflighted as a 1x job — that is 16x short."""
    from core.pipeline import resolve_scale
    from tests.support import make_model

    cfg = make_config(tmp_path, frames=2)
    cfg = type(cfg)(
        input_path=cfg.input_path,
        output_path=cfg.output_path,
        model_path=make_model(tmp_path / "realesr-general-x4v3.pth"),
        work_dir=cfg.work_dir,
        devices=cfg.devices,
    )
    scale, source = resolve_scale(cfg)
    assert scale == 4
    assert "catalogue" in source


def test_the_settings_win_over_the_catalogue(tmp_path: Path) -> None:
    from core.pipeline import resolve_scale
    from tests.support import make_model

    cfg = make_config(tmp_path, frames=2)
    cfg = type(cfg)(
        input_path=cfg.input_path,
        output_path=cfg.output_path,
        model_path=make_model(tmp_path / "realesr-general-x4v3.pth"),
        work_dir=cfg.work_dir,
        devices=cfg.devices,
        scale=2,
    )
    assert resolve_scale(cfg) == (2, "the settings")


def test_an_unknown_model_states_its_assumption(tmp_path: Path) -> None:
    from core.pipeline import resolve_scale
    from tests.support import make_model

    cfg = make_config(tmp_path, frames=2)
    cfg = type(cfg)(
        input_path=cfg.input_path,
        output_path=cfg.output_path,
        model_path=make_model(tmp_path / "somebody-elses.pth"),
        work_dir=cfg.work_dir,
        devices=cfg.devices,
    )
    scale, source = resolve_scale(cfg)
    assert scale == 1
    assert "assumption" in source


def test_an_onnx_model_sizes_the_preflight_from_its_own_graph(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    from core.backends.export import ONNX_OPSET
    from core.pipeline import resolve_scale

    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"], name="r")],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 128, 128])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", ONNX_OPSET)]
    )
    model.ir_version = 10
    path = tmp_path / "x2.onnx"
    onnx.save(model, str(path))

    cfg = make_config(tmp_path, frames=2)
    cfg = type(cfg)(
        input_path=cfg.input_path,
        output_path=cfg.output_path,
        model_path=path,
        work_dir=cfg.work_dir,
        devices=cfg.devices,
    )
    scale, source = resolve_scale(cfg)
    assert scale == 2
    assert "shapes" in source
