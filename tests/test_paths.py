from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from core import paths
from core.paths import (
    APP_NAME,
    catalogue_path,
    config_dir,
    coreml_cache_dir,
    data_dir,
    default_work_dir,
    log_file,
    models_dir,
    runtime_dir,
    runtimes_dir,
)


def _use_linux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    return tmp_path / "xdg-data" / APP_NAME


def test_linux_uses_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _use_linux(monkeypatch, tmp_path)
    assert data_dir() == root
    assert root.is_dir()
    assert models_dir() == root / "models"
    assert models_dir().is_dir()
    assert runtimes_dir() == root / "runtimes"
    assert runtimes_dir().is_dir()
    assert runtime_dir("cuda-12.8") == root / "runtimes" / "cuda-12.8"
    assert (root / "runtimes" / "cuda-12.8").is_dir()
    assert coreml_cache_dir() == root / "coreml-cache"
    assert coreml_cache_dir().is_dir()
    assert log_file() == root / "upscaler.log"
    assert log_file().parent == root


def test_linux_config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_linux(monkeypatch, tmp_path)
    expected = tmp_path / "xdg-config" / APP_NAME
    assert config_dir() == expected
    assert expected.is_dir()


def test_linux_falls_back_to_dot_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert data_dir() == tmp_path / ".local" / "share" / APP_NAME
    assert data_dir().is_dir()


def test_windows_uses_appdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    assert data_dir() == tmp_path / "AppData" / "Roaming" / APP_NAME
    assert config_dir() == tmp_path / "AppData" / "Roaming" / APP_NAME
    assert data_dir().is_dir()


def test_macos_uses_application_support(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    expected = tmp_path / "Library" / "Application Support" / APP_NAME
    assert data_dir() == expected
    assert config_dir() == expected
    assert data_dir().is_dir()


def test_default_work_dir_is_next_to_the_source(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    work = default_work_dir(source)
    assert work.parent == tmp_path
    assert work.name == ".upscaler-work-clip"
    assert not work.exists()


def test_default_work_dir_creates_nothing_for_a_dotfile_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / ".hidden.mkv"
    assert default_work_dir(source) == tmp_path / ".upscaler-work-.hidden"
    assert list(tmp_path.iterdir()) == []


def test_catalogue_path_points_into_the_models_directory() -> None:
    found = catalogue_path()
    assert found.name == "catalogue.json"
    assert found.parent.name == "models"


def test_core_imports_no_heavy_module() -> None:
    """A front-end must be able to import the core without a heavy import."""
    code = (
        "import sys\n"
        "import core.config, core.paths, core.events, core.errors\n"
        "heavy = [m for m in ('torch', 'cv2', 'onnxruntime', 'PySide6', 'toga')"
        " if m in sys.modules]\n"
        "print(','.join(heavy))\n"
        "sys.exit(1 if heavy else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, f"heavy modules imported: {result.stdout}"
