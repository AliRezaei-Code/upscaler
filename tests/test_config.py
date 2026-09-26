from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Device, JobConfig, format_memory
from core.errors import ConfigError
from core.paths import default_work_dir

INPUT = Path("/tmp/source.mp4")
OUTPUT = Path("/tmp/out.mp4")
MODEL = Path("/tmp/model.pth")
WORK = Path("/tmp/work")


def make_device(**overrides: object) -> Device:
    """A usable CUDA device, with the given fields replaced."""
    base: dict[str, object] = {
        "index": 0,
        "name": "Tesla P40",
        "vendor": "nvidia",
        "backend": "onnx:cuda",
        # nvidia-smi reports memory.total in MiB (24576 for a 24 GB card);
        # the probe converts to bytes before it reaches a Device.
        "total_memory_bytes": 24576 * 1024 * 1024,
        "pci_bus_id": "00000000:07:00.0",
        "compute_capability": (6, 1),
        "usable": True,
        "unusable_reason": None,
    }
    base.update(overrides)

    return Device(**base)  # type: ignore[arg-type]


def make_config(tmp_path: Path, **overrides: object) -> JobConfig:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"not really a video")
    base: dict[str, object] = {
        "input_path": source,
        "output_path": tmp_path / "out.mp4",
        "model_path": MODEL,
        "work_dir": tmp_path / "work",
        "devices": (make_device(),),
    }
    base.update(overrides)
    return JobConfig(**base)  # type: ignore[arg-type]


def test_defaults(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    assert cfg.scale == 0
    assert cfg.precision == "auto"
    assert cfg.tile_size == 0
    assert cfg.fps is None
    assert cfg.crf == 18
    assert cfg.delete_frames_after_encode is False
    cfg.validate()


def test_missing_input_message(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, input_path=tmp_path / "absent.mp4")
    with pytest.raises(ConfigError) as excinfo:
        cfg.validate()
    assert str(excinfo.value) == f"Input video not found: {tmp_path / 'absent.mp4'}"


def test_no_device_message(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, devices=())
    with pytest.raises(ConfigError) as excinfo:
        cfg.validate()
    assert str(excinfo.value) == "Select at least one GPU"


def test_unusable_device_message(tmp_path: Path) -> None:
    reason = "driver did not answer within 10 s; VRAM and compute capability unknown"
    cfg = make_config(
        tmp_path,
        devices=(make_device(name="Tesla P40", usable=False, unusable_reason=reason),),
    )
    with pytest.raises(ConfigError) as excinfo:
        cfg.validate()
    assert str(excinfo.value) == f"Tesla P40: {reason}"


def test_every_unusable_device_is_named(tmp_path: Path) -> None:
    cfg = make_config(
        tmp_path,
        devices=(
            make_device(index=0, name="Tesla P40", usable=False, unusable_reason="a"),
            make_device(index=1, name="Tesla P40", usable=False, unusable_reason="b"),
            make_device(index=2, name="Tesla P40", usable=True),
        ),
    )
    with pytest.raises(ConfigError) as excinfo:
        cfg.validate()
    message = str(excinfo.value)
    assert "Tesla P40: a" in message
    assert "Tesla P40: b" in message


@pytest.mark.parametrize("crf", [-1, 52, 100])
def test_crf_out_of_range(tmp_path: Path, crf: int) -> None:
    cfg = make_config(tmp_path, crf=crf)
    with pytest.raises(ConfigError) as excinfo:
        cfg.validate()
    assert str(excinfo.value) == f"CRF must be 0-51, got {crf}"


@pytest.mark.parametrize("crf", [0, 18, 51])
def test_crf_in_range(tmp_path: Path, crf: int) -> None:
    make_config(tmp_path, crf=crf).validate()


def test_missing_input_is_reported_before_the_crf(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, input_path=tmp_path / "absent.mp4", crf=99)
    with pytest.raises(ConfigError, match="Input video not found"):
        cfg.validate()


def test_devices_checked_before_crf(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, devices=(), crf=99)
    with pytest.raises(ConfigError, match="Select at least one GPU"):
        cfg.validate()


def test_default_work_dir_creates_nothing(tmp_path: Path) -> None:
    source = tmp_path / "Mosaferan.Mahtab.mkv"
    source.write_bytes(b"x")
    work = default_work_dir(source)
    assert work == tmp_path / ".upscaler-work-Mosaferan.Mahtab"
    assert not work.exists()
    assert list(tmp_path.iterdir()) == [source]


def test_label_states_unknown_vram() -> None:
    device = make_device(total_memory_bytes=0, compute_capability=None)
    assert device.label == "GPU 0 — Tesla P40 — VRAM unknown — onnx:cuda"


def test_label_states_known_vram() -> None:
    assert format_memory(24576 * 1024 * 1024) == "24 GiB VRAM"
    assert format_memory(512 * 1024**2) == "512 MiB VRAM"
    assert format_memory(0) == "VRAM unknown"
    assert "CPU" in make_device(vendor="cpu", backend="cpu").label


def test_config_is_frozen(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    with pytest.raises(Exception, match="cannot assign to field"):
        cfg.crf = 40  # type: ignore[misc]


def test_input_path_and_output_path_are_independent(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    assert cfg.input_path != cfg.output_path
    assert cfg.input_path.suffix == ".mp4"
    assert cfg.output_path.suffix == ".mp4"
    assert OUTPUT.name not in {INPUT.name, MODEL.name, WORK.name}
