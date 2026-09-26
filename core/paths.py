"""Filesystem locations.

Nothing here imports torch, cv2 or onnxruntime: every path is cheap to
resolve, so a front-end can compute all of them at import time and during a
repaint without blocking on a heavy import.

Every accessor creates its directory on first call. `default_work_dir` is the
one exception: see its docstring.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "upscaler"


def _ensure(path: Path) -> Path:
    """Create `path` and its parents, and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    """Where models, runtimes, logs and caches live.

    Windows: `%APPDATA%/upscaler`. macOS: `~/Library/Application
    Support/upscaler`. Elsewhere: `$XDG_DATA_HOME/upscaler`, falling back to
    `~/.local/share/upscaler`.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return _ensure(base / APP_NAME)


def config_dir() -> Path:
    """Where the app's own settings live.

    Windows: `%APPDATA%/upscaler`. macOS: `~/Library/Application
    Support/upscaler`. Elsewhere: `$XDG_CONFIG_HOME/upscaler`, falling back to
    `~/.config/upscaler`.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return _ensure(base / APP_NAME)


def models_dir() -> Path:
    """Downloaded model weights, plus the export and hash caches."""
    return _ensure(data_dir() / "models")


def runtimes_dir() -> Path:
    """Runtimes unpacked by the Runtimes tab, one site-packages tree each."""
    return _ensure(data_dir() / "runtimes")


def runtime_dir(name: str) -> Path:
    """The unpacked runtime called `name`; created on first call."""
    return _ensure(runtimes_dir() / name)


def coreml_cache_dir() -> Path:
    """Where the CoreML execution provider caches compiled `.mlmodelc` files."""
    return _ensure(data_dir() / "coreml-cache")


def log_file() -> Path:
    """The rolling log both front-ends append to."""
    return data_dir() / "upscaler.log"


def default_work_dir(src: Path) -> Path:
    """The work directory a source video gets by default.

    Returns a path and **does not create it**. The directory is created by the
    pipeline once the disk preflight has passed, and nothing else: a UI repaint
    that materialises an empty directory is free, but a repaint that
    materialises one next to an 846 GB scratch tree is not, and this function
    is called on every Source-field change.
    """
    return src.parent / f".upscaler-work-{src.stem}"


def catalogue_path() -> Path:
    """Locate the shipped model catalogue.

    Three shipping modes have to work: a checkout (the file is at the repo
    root), a PyInstaller bundle (`sys._MEIPASS`, set only inside the bundle),
    and an installed copy next to the executable. The first hit wins, so the
    caller gets an existing file in every mode and an `UpscalerError` in none.
    """
    roots = [Path.cwd(), Path(__file__).resolve().parent.parent]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.insert(0, Path(meipass))
    if getattr(sys, "frozen", False):
        roots.insert(0, Path(sys.executable).resolve().parent)
    for root in roots:
        candidate = root / "models" / "catalogue.json"
        if candidate.is_file():
            return candidate
    return roots[0] / "models" / "catalogue.json"
