"""The Runtimes tab: the accelerator stacks this build does not ship.

The Linux slim `.deb` is about 0.1 GB precisely because it carries neither
`onnxruntime` nor `torch`; this tab is where the user decides to pay for them.
It therefore has to say what that will cost *before* the click, and the number
cannot be hard-coded because the wheels move. So the size is resolved from
PyPI's JSON API on a background thread, the row says `calculating…` until the
answer arrives, and it says `size unavailable` when PyPI cannot be reached. A
made-up number would be worse than none: the user is deciding whether to
commit gigabytes to a download.

Two facts shape everything else here.

**A wheel is a zip.** It is unpacked with `zipfile` into a real site-packages
tree under `runtimes_dir()/<runtime>/lib/pythonX.Y/site-packages`, which this
module puts on `sys.path`. `pip` is never invoked: a frozen bundle has no
installer to invoke, and shelling out to one is exactly what the packaging
exists to avoid.

**A restart is required after every install.** Extending `sys.path` after
`torch` and `onnxruntime` are already in `sys.modules` cannot re-import them,
so a button that merely changed the path would appear to work and change
nothing. The marker file therefore survives the close, and the next start
consumes it after logging the resolved `onnxruntime.__file__` and
`torch.__file__` — those two lines are what make a bug report reproducible,
because a GPU execution provider that cannot load falls back to CPU silently
and the first question is always which file the interpreter picked.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import re
import shutil
import sys
import sysconfig
import threading
import zipfile
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import unquote, urlparse

import requests
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core import devices
from core.app import Emit, TaskFn, UpscalerApp
from core.download import USER_AGENT, download_file
from core.errors import DownloadError, UpscalerError
from core.events import PipelineEvent
from core.paths import runtimes_dir

log = logging.getLogger(__name__)

PYTHON_DIR = f"{sys.version_info.major}.{sys.version_info.minor}"
"""The `3.12` in a runtime's `lib/python3.12/site-packages` path."""

PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
PYPI_RELEASE_JSON_URL = "https://pypi.org/pypi/{name}/{version}/json"
HTTP_TIMEOUT_SECONDS = 10.0
DOWNLOAD_TIMEOUT_SECONDS = 60.0

#: Every task this tab submits starts with this, so the window can dispatch to
#: it without knowing which tab is showing. A size probe appends `SIZE_SUFFIX`.
TASK_PREFIX = "runtime:"
SIZE_SUFFIX = ":size"

#: The staged tree the wheels are unpacked into, renamed into place only after
#: every one of them has unpacked cleanly.
STAGING_DIR = "lib.tmp"
RUNTIME_LIB_DIR = "lib"
WHEELS_DIR = "wheels"
MARKER_NAME = "restart-required"

CALCULATING = "calculating…"
SIZE_UNAVAILABLE = "size unavailable"
STATE_INSTALLED = "installed"
STATE_MISSING = "not installed"
STATE_UNAVAILABLE = "unavailable"
RESTART_MESSAGE = "Restart required to apply"
MIGRAPHX = "migraphx"

#: The progress bar's full-scale value. A wheel is gigabytes and a Qt range is
#: a 32-bit integer, so the bar carries a proportion of this rather than bytes.
_BAR_STEPS = 1000

#: Word for word what `core.devices.probe_amd` tells the user, so the device
#: list and this tab cannot disagree about why an AMD GPU is unusable.
MIGRAPHX_UNAVAILABLE = (
    "AMD GPU acceleration on Linux requires ROCm 7.x; see the Runtimes tab"
)
MIGRAPHX_SOURCE_URL = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/"

_COLUMN_SUMMARY = 0
_COLUMN_REQUIRES = 1
_COLUMN_SIZE = 2
_COLUMN_STATE = 3
_COLUMN_ACTION = 4
_HEADERS = ("Runtime", "Requires", "Size", "State", "")


@dataclass(frozen=True)
class RuntimeSpec:
    """One row of the Runtimes tab.

    Attributes:
        name: The stable id used by `RuntimesTab.install` and `resolved`, and
            the middle of this runtime's task id.
        summary: The wording shown as the row's name, verbatim.
        requirements: What must already be on the machine for the runtime to
            work, shown in the row so the user reads it before committing.
        size_bytes: The download size when it is known without asking anyone —
            `0` for the bundled CPU runtime and for the ones still being
            resolved, which read `calculating…` until the answer arrives.
        source_url: Where the packages come from, shown in the detail panel.
        packages: What to install, in the three forms this tab can resolve:
            `name==version`, `name[extras]==version`, a bare `name` for the
            newest release, or `name @ <wheel url>` for a wheel that is not on
            PyPI at all. An empty tuple means the runtime is already there.
        directory: The subdirectory of `runtimes_dir()` this runtime unpacks
            into, so a newer line can be installed beside an older one.
    """

    name: str
    summary: str
    requirements: tuple[str, ...]
    size_bytes: int
    source_url: str
    packages: tuple[str, ...]
    directory: str

    @property
    def bundled(self) -> bool:
        """True when the interpreter already has this runtime."""
        return not self.packages


RUNTIMES: tuple[RuntimeSpec, ...] = (
    RuntimeSpec(
        name="cpu",
        summary="CPU (bundled)",
        requirements=("nothing — this is what the app already runs on",),
        size_bytes=0,
        source_url="",
        packages=(),
        directory="cpu",
    ),
    RuntimeSpec(
        name="cuda-12.8",
        summary="CUDA 12.8 (ONNX Runtime 1.26.0 + torch 2.7.1)",
        requirements=("a CUDA 12.8 driver, 525 or newer",),
        size_bytes=0,
        source_url="https://pypi.org/project/onnxruntime-gpu/",
        packages=("onnxruntime-gpu[cuda,cudnn]==1.26.0", "torch==2.7.1"),
        directory="cuda-12.8",
    ),
    RuntimeSpec(
        name="cuda-13.0",
        summary="CUDA 13.0 (ONNX Runtime 1.30.0 + torch latest)",
        requirements=("a CUDA 13.0 driver, 580 or newer",),
        size_bytes=0,
        source_url="https://pypi.org/project/onnxruntime-gpu/",
        packages=("onnxruntime-gpu[cuda,cudnn]==1.30.0", "torch"),
        directory="cuda-13.0",
    ),
    RuntimeSpec(
        # The MIGraphX wheels are on repo.radeon.com, not PyPI, and are built
        # against a host ROCm install, so the row is only offerable on a
        # machine that has one.
        name=MIGRAPHX,
        summary="MIGraphX (ROCm 7.2)",
        requirements=("a host ROCm 7.x install at /opt/rocm",),
        size_bytes=0,
        source_url=MIGRAPHX_SOURCE_URL,
        packages=(
            "onnxruntime_migraphx @ https://repo.radeon.com/rocm/manylinux/"
            "rocm-rel-7.2/onnxruntime_migraphx-1.23.2-cp312-cp312-"
            "manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "torch @ https://download.pytorch.org/whl/rocm7.2/"
            "torch-2.12.1%2Brocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl",
        ),
        directory=MIGRAPHX,
    ),
)


def specs_for(rocm: str | None) -> tuple[RuntimeSpec, ...]:
    """`RUNTIMES` with the MIGraphX row told what this host actually offers.

    Args:
        rocm: `core.devices.rocm_version()`, or `None` where ROCm is absent.
            Without a host install there is no MIGraphX wheel to unpack — the
            execution provider lives in the host ROCm stack — so the row is
            left at its static form and the tab marks it unavailable itself.

    Returns:
        One spec per runtime, in the order they are listed.
    """
    resolved: list[RuntimeSpec] = []
    for spec in RUNTIMES:
        if spec.name == MIGRAPHX and rocm is not None:
            resolved.append(
                replace(
                    spec,
                    requirements=(f"ROCm {rocm} at /opt/rocm",),
                    source_url=MIGRAPHX_SOURCE_URL,
                )
            )
        else:
            resolved.append(spec)
    return tuple(resolved)


#: The four forms this tab can resolve, in one pattern. The extras are part of
#: the requirement and are not otherwise used: the wheel is the same file
#: whether or not the extra was asked for, and `onnxruntime-gpu` carries the
#: CUDA libraries inside the wheel rather than as separate downloads.
_REQUIREMENT_RE = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[[^\]]*\])?"
    r"(?:\s*==\s*(?P<version>[A-Za-z0-9.*+!_-]+)|\s*@\s*(?P<url>\S+))?"
)

_ARCH_ALIASES = {
    "amd64": ("x86_64", "amd64", "x86"),
    "x86_64": ("x86_64", "amd64", "x86"),
    "x86-64": ("x86_64", "amd64", "x86"),
    "arm64": ("aarch64", "arm64"),
    "aarch64": ("aarch64", "arm64"),
}


@dataclass(frozen=True)
class _Requirement:
    """One package specifier, taken apart."""

    raw: str
    name: str
    version: str | None
    url: str | None


def parse_requirement(spec: str) -> _Requirement:
    """Read `name`, `name[extras]==version`, `name` or `name @ url`.

    Raises:
        DownloadError: For anything else, naming the specifier. A requirement
            this function cannot read is one it cannot install either, and
            saying so here beats unpacking the wrong thing later.
    """
    text = spec.strip()
    match = _REQUIREMENT_RE.fullmatch(text)
    if match is None:
        raise DownloadError(
            f"{spec!r} is not a requirement this tab can install; expected "
            f"'name', 'name[extras]==version' or 'name @ <wheel url>'"
        )
    return _Requirement(
        raw=text,
        name=match["name"],
        version=match["version"],
        url=match["url"],
    )


def _python_tag() -> str:
    """This interpreter's CPython tag, `cp312`."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _arch_spellings() -> tuple[str, ...]:
    """The ways a wheel tag spells this machine's architecture.

    `platform.machine()` and the wheel tags disagree by name on every desktop
    platform — `AMD64` against `win_amd64`, `arm64` against `aarch64` — so
    every spelling has to be accepted or nothing matches anywhere.
    """
    name = platform.machine().lower()
    return _ARCH_ALIASES.get(name, (name,))


def _platform_matches(tag: str) -> bool:
    """True when one of a wheel's dot-separated platform tags is ours."""
    arches = _arch_spellings()
    for part in tag.split("."):
        if not any(part.endswith(arch) for arch in arches):
            continue
        if part.startswith("manylinux") and sys.platform.startswith("linux"):
            return True
        if part.startswith("linux") and sys.platform.startswith("linux"):
            return True
        if part.startswith("win") and sys.platform == "win32":
            return True
        if part.startswith("macosx") and sys.platform == "darwin":
            return True
    return False


def _wheel_matches(filename: str) -> bool:
    """True when `filename` names a wheel this interpreter can install.

    The Python tag and the platform tag are what decide it. The ABI tag is
    deliberately not checked: a `cp37-abi3` wheel loads into 3.12 as happily
    as into 3.7, and rejecting it would be wrong.
    """
    if not filename.endswith(".whl"):
        return False
    parts = filename[: -len(".whl")].split("-")
    if len(parts) < 5:
        return False
    python_tag, _abi_tag, platform_tag = parts[-3:]
    return python_tag == _python_tag() and _platform_matches(platform_tag)


def _platform_rank(filename: str) -> int:
    """Lower is the better match: a manylinux wheel over a bare `linux` one."""
    return 0 if "manylinux" in filename[: -len(".whl")] else 1


def _http_json(url: str) -> dict[str, Any]:
    """Fetch and parse one JSON document, or raise `DownloadError`."""
    try:
        response = requests.get(
            url,
            timeout=HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        if response.status_code != 200:
            raise DownloadError(f"{url} answered HTTP {response.status_code}")
        payload: Any = response.json()
    except requests.RequestException as exc:
        raise DownloadError(f"Could not reach {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DownloadError(f"{url} did not answer with a JSON object")
    return cast("dict[str, Any]", payload)


def _content_length(url: str) -> int:
    """The size of a wheel that is not on PyPI, from a HEAD request."""
    try:
        response = requests.head(
            url,
            timeout=HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        raise DownloadError(f"Could not reach {url}: {exc}") from exc
    if response.status_code != 200:
        raise DownloadError(f"{url} answered HTTP {response.status_code}")
    raw = response.headers.get("Content-Length", "")
    if not raw.isdigit():
        raise DownloadError(f"{url} did not report how big it is")
    return int(raw)


def _pypi_files(requirement: _Requirement) -> list[dict[str, Any]]:
    """Every file PyPI lists for the release `requirement` names.

    Both endpoints put that release's files under `urls`: the versioned one for
    a pinned requirement, the bare one for the newest release.
    """
    if requirement.version:
        url = PYPI_RELEASE_JSON_URL.format(
            name=requirement.name, version=requirement.version
        )
    else:
        url = PYPI_JSON_URL.format(name=requirement.name)
    files = _http_json(url).get("urls")
    if not isinstance(files, list) or not files:
        raise DownloadError(f"PyPI lists no files for {requirement.raw}")
    return [entry for entry in files if isinstance(entry, dict)]


def _select_wheel(requirement: _Requirement) -> tuple[str, str, int]:
    """Resolve one requirement to the `(url, filename, size)` to install.

    Raises:
        DownloadError: When nothing this machine can install is on offer. The
            message names the requirement, because the alternative — unpacking
            a wheel that cannot be imported — fails much later and much less
            clearly.
    """
    here = f"Python {_python_tag()} on {sysconfig.get_platform()}"
    if requirement.url is not None:
        # The wheel is named by the URL, and the host percent-encodes the
        # local version (`2.12.1%2Brocm7.2`), which is not a filename.
        filename = unquote(PurePosixPath(urlparse(requirement.url).path).name)
        if not _wheel_matches(filename):
            raise DownloadError(f"{requirement.raw} is not a wheel for {here}")
        return requirement.url, filename, _content_length(requirement.url)

    candidates = [
        entry
        for entry in _pypi_files(requirement)
        if entry.get("packagetype") == "bdist_wheel"
        and not entry.get("yanked")
        and isinstance(entry.get("filename"), str)
        and _wheel_matches(cast("str", entry["filename"]))
    ]
    if not candidates:
        raise DownloadError(f"{requirement.raw} has no wheel for {here}")
    # Oldest manylinux first: an older manylinux tag is the one that still
    # installs on an older glibc, and among the tags that qualify here the
    # smallest wheel is the conservative choice.
    best = min(
        candidates,
        key=lambda entry: (
            _platform_rank(cast("str", entry["filename"])),
            int(entry.get("size") or 0),
        ),
    )
    filename = cast("str", best["filename"])
    url = best.get("url")
    if not isinstance(url, str) or not url:
        raise DownloadError(f"PyPI listed {filename} without a download URL")
    return url, filename, int(best.get("size") or 0)


def runtime_size(spec: RuntimeSpec) -> int:
    """Exactly how many bytes installing `spec` will download."""
    if spec.size_bytes:
        return spec.size_bytes
    total = 0
    for package in spec.packages:
        total += _select_wheel(parse_requirement(package))[2]
    return total


def human_bytes(count: int) -> str:
    """A count of bytes as a rounded size with the exact figure beside it.

    The exact number is always there. A user deciding whether to spend an
    evening downloading cannot be given a rounded figure and nothing else.
    """
    if count < 1024:
        return f"{count:,} bytes"
    units = ("KB", "MB", "GB", "TB")
    value = float(count)
    for unit in units:
        value /= 1024.0
        if value < 1024.0 or unit == "TB":
            break
    return f"{value:.2f} {unit} ({count:,} bytes)"


def pending_restart_marker() -> Path:
    """The file that tells the next start a runtime was just installed."""
    return runtimes_dir() / MARKER_NAME


def site_packages_dir(directory: str) -> Path | None:
    """An unpacked runtime's site-packages tree, or `None` if it has none."""
    tree = (
        runtimes_dir()
        / directory
        / RUNTIME_LIB_DIR
        / f"python{PYTHON_DIR}"
        / "site-packages"
    )
    return tree if tree.is_dir() else None


def extend_search_path() -> list[Path]:
    """Put every installed runtime on `sys.path`, first-listed runtime first.

    Returns:
        The trees that were added, so a caller that cares can log them. A tree
        already on the path is left where it is rather than moved, because
        `sys.path` is the app's own decision and this only tops it up.
    """
    added: list[Path] = []
    for spec in reversed(RUNTIMES):
        tree = site_packages_dir(spec.directory)
        if tree is None:
            continue
        if str(tree) not in sys.path:
            sys.path.insert(0, str(tree))
            added.append(tree)
    return added


def resolved_origin(module: str) -> str | None:
    """The file `module` resolves to on this `sys.path`, without importing it.

    `importlib.util.find_spec` answers exactly the question the startup log
    asks, from the same search path an import would use, and costs a few
    directory entries. Importing torch to read the same answer costs seconds
    and this runs while the window is opening.
    """
    already = sys.modules.get(module)
    if already is not None:
        spec = getattr(already, "__spec__", None)
        if spec is not None:
            return _origin_of(spec)
    try:
        found = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    return None if found is None else _origin_of(found)


def _origin_of(spec: Any) -> str | None:
    """A module spec's `origin`, when it is a real file."""
    origin = getattr(spec, "origin", None)
    return origin if isinstance(origin, str) else None


def log_resolved_runtimes() -> None:
    """Log the resolved `onnxruntime` and `torch`, then clear a restart marker.

    Those two lines are what make a bug report reproducible: the ONNX Runtime
    GPU providers fall back to CPU without saying so when their runtime cannot
    load, so the first question about a "it ran on the CPU" report is which
    file the interpreter picked. A restart that has just been consumed is
    acknowledged once, after those paths are known.

    Neither module is imported to do this — the slim `.deb` ships neither, and
    an import would put seconds on the GUI thread for a log line.
    """
    for module in ("onnxruntime", "torch"):
        log.info("resolved %s: %s", module, resolved_origin(module) or "not installed")
    marker = pending_restart_marker()
    if marker.exists():
        log.info("a runtime was installed; this start has it on the search path")
        marker.unlink()


def _member_path(member: zipfile.ZipInfo) -> Path | None:
    """Where one wheel member belongs inside site-packages, or None to skip it.

    `name-version.data/purelib` and `.data/platlib` are the wheel's own
    spelling of "this goes in site-packages", so their prefix is dropped. The
    rest of `.data` — scripts, headers, prefixes — has nowhere to go in a
    runtime directory and is skipped, as are symbolic links and any name that
    could escape the tree. A wheel is a remote archive, and `pip` applies the
    same rule for the same reason.
    """
    if member.is_dir() or _is_symlink(member):
        return None
    parts = PurePosixPath(member.filename).parts
    if not parts or ".." in parts or PurePosixPath(member.filename).is_absolute():
        return None
    if len(parts) > 2 and parts[0].endswith(".data"):
        if parts[1] not in ("purelib", "platlib"):
            return None
        parts = parts[2:]
    if not parts:
        return None
    return Path(*parts)


def _is_symlink(member: zipfile.ZipInfo) -> bool:
    """True for a wheel member that is a link rather than a file."""
    return member.create_system == 3 and (member.external_attr >> 16) & 0o170000 == (
        0o120000
    )


def unpack_wheel(wheel: Path, destination: Path) -> int:
    """Unpack one wheel into a site-packages tree; return the files skipped.

    `pip` is not used: a frozen bundle has no installer to run, and the whole
    point of the runtime directory is a tree the app itself puts on
    `sys.path`. A wheel is a zip, and `zipfile` is enough.
    """
    skipped = 0
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            if member.is_dir():
                # Every file's parent directory is created anyway; a wheel's
                # directory entries are not files a user could notice.
                continue
            name = _member_path(member)
            if name is None:
                skipped += 1
                continue
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
    return skipped


def _retagged(emit: Emit, task_id: str) -> Emit:
    """Re-label `download_file`'s events, which it stamps `model:<stem>`.

    The downloader is shared with the Models tab and hard-codes that prefix.
    Without this the byte progress of a runtime install would be dispatched to
    the model rows instead of to this tab's bar.
    """

    def send(event: PipelineEvent) -> None:
        emit(replace(event, task_id=task_id))

    return send


def install_runtime(spec: RuntimeSpec, emit: Emit, stop: threading.Event) -> Path:
    """Download and unpack every package of `spec` into a fresh runtime.

    The wheels land in `runtimes_dir()/<directory>/wheels/` and are unpacked
    into a staging tree that is renamed into place only once all of them have
    unpacked cleanly. A failure therefore cannot leave a half-unpacked runtime
    that the next start finds and trusts — the staging tree is removed and
    nothing about the installed runtime has changed.

    Returns:
        The site-packages tree that was put in place.
    """
    task_id = f"{TASK_PREFIX}{spec.name}"
    root = runtimes_dir() / spec.directory
    wheels = root / WHEELS_DIR
    staging = root / STAGING_DIR
    final = root / RUNTIME_LIB_DIR
    wheels.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(staging, ignore_errors=True)

    emit(
        PipelineEvent(
            kind="stage",
            task_id=task_id,
            stage="resolve",
            message=f"Choosing wheels for {spec.summary}",
        )
    )
    plan: list[tuple[str, str, int]] = []
    for package in spec.packages:
        url, filename, size = _select_wheel(parse_requirement(package))
        plan.append((url, filename, size))
    total = sum(size for _url, _filename, size in plan)
    emit(
        PipelineEvent(
            kind="stage",
            task_id=task_id,
            stage="resolve",
            message=f"{len(plan)} wheels, {human_bytes(total)}",
        )
    )

    site = staging / f"python{PYTHON_DIR}" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    try:
        for url, filename, size in plan:
            if stop.is_set():
                raise UpscalerError(
                    "The install was stopped before it finished; nothing was unpacked"
                )
            wheel = wheels / filename
            download_file(
                url,
                wheel,
                expected_size=size,
                emit=_retagged(emit, task_id),
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )
            emit(
                PipelineEvent(
                    kind="stage",
                    task_id=task_id,
                    stage="unpack",
                    message=f"Unpacking {filename}",
                )
            )
            skipped = unpack_wheel(wheel, site)
            downloaded += size
            if skipped:
                emit(
                    PipelineEvent(
                        kind="log",
                        task_id=task_id,
                        message=(
                            f"Ignored {skipped} files of {filename} that a "
                            f"runtime directory has nowhere to put: scripts, "
                            f"headers and prefixes"
                        ),
                    )
                )
            emit(
                PipelineEvent(
                    kind="device_progress",
                    task_id=task_id,
                    processed=downloaded,
                    total=total,
                    message=filename,
                )
            )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Every wheel unpacked, so the tree can be committed. The old one goes
    # first: a rename onto an existing directory fails on both platforms.
    if final.exists():
        shutil.rmtree(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(final)
    installed = final / f"python{PYTHON_DIR}" / "site-packages"
    # Written here rather than when the event lands, so an app that is closed
    # the instant the install finishes still says so on the next start.
    pending_restart_marker().write_text(f"{spec.name}\n", encoding="utf-8")
    emit(PipelineEvent(kind="done", task_id=task_id, message=str(installed)))
    return installed


class RuntimesTab(QWidget):
    """Install the accelerator runtimes the running build does not ship.

    The window hands this tab an `UpscalerApp` and feeds it every pipeline
    event whose `task_id` starts with `runtime:`, through `handle_event`. It
    works without that: a tab that is constructed and shown with no event pump
    running is a supported state, because the main window's headless smoke
    builds one, and it is why every piece of state the tab needs — the
    installed trees, the restart marker — is read from the disk rather than
    from events that may never arrive.

    Attributes:
        specs: The runtimes as this host can offer them, after
            `core.devices.rocm_version()` has been consulted.
    """

    def __init__(self, app: UpscalerApp, parent: QWidget | None = None) -> None:
        """Build the tab, extend the search path and start the size probes.

        Nothing here blocks. The path extension and the two log lines are a
        handful of directory reads, and every network call is a task on
        `app`, so the window opens immediately whether or not PyPI answers.
        """
        super().__init__(parent)
        self._app = app
        self._rocm = devices.rocm_version()
        self._specs = specs_for(self._rocm)
        self._installing: set[str] = set()
        self._rows: dict[str, int] = {}
        self._names_by_row: dict[int, str] = {}
        self._requires_items: dict[str, QTableWidgetItem] = {}
        self._size_items: dict[str, QTableWidgetItem] = {}
        self._state_items: dict[str, QTableWidgetItem] = {}
        self._buttons: dict[str, QPushButton] = {}
        self._build_ui()
        extend_search_path()
        log_resolved_runtimes()
        self._show_resolved()
        self._refresh_restart()
        self._probe_sizes()

    @property
    def specs(self) -> tuple[RuntimeSpec, ...]:
        """The runtimes as this host can offer them."""
        return self._specs

    def install(self, name: str) -> None:
        """Start installing the runtime called `name`.

        A bundled runtime has nothing to install and a runtime the host cannot
        offer explains itself in its own row; both return without submitting
        anything, so a click on a disabled button is not an error.

        Raises:
            KeyError: If no runtime is called `name`.
        """
        spec = self._spec(name)
        if spec.bundled or name in self._installing:
            return
        if not self.is_offered(name):
            self._status.setText(self._unavailable_reason(spec))
            return
        self._installing.add(name)
        self._refresh_row(name)
        self._status.setText(f"Installing {spec.summary}…")
        self._bar.setRange(0, 0)
        self._app.submit(f"{TASK_PREFIX}{name}", partial(install_runtime, spec))

    def is_offered(self, name: str) -> bool:
        """True when this host can install the runtime called `name`."""
        spec = self._spec(name)
        return not spec.bundled and self._unavailable_reason(spec) == ""

    def resolved(self, name: str) -> Path | None:
        """The site-packages tree installed for `name`, or `None`.

        The bundled CPU runtime has none: the interpreter already has it, and
        there is no directory to point at.
        """
        try:
            spec = self._spec(name)
        except KeyError:
            return None
        return site_packages_dir(spec.directory)

    def is_restart_required(self) -> bool:
        """True when a runtime has been installed since this process started.

        The marker file is the state rather than a flag in memory, so the
        answer is the same in the process that installed the runtime and in the
        one that comes after it.
        """
        return self.pending_restart_marker().exists()

    def restart(self) -> None:
        """Replace this process with a fresh interpreter.

        `os.execv` rather than a second `QApplication`: the runtime tree is
        only read at import time, so the process that installed it cannot use
        it, and `execv` hands the new process this one's window instead of
        reopening another one beside it.
        """
        os.execv(sys.executable, [sys.executable, *sys.argv])

    @staticmethod
    def pending_restart_marker() -> Path:
        """The file that survives a close so the next start knows to say so."""
        return pending_restart_marker()

    def handle_event(self, event: PipelineEvent) -> None:
        """Fold one pipeline event into this tab.

        The main window wires this to `worker_bridge.EventBridge.event`, for
        every event whose `task_id` starts with `runtime:`. Events belonging to
        another producer — a job, a model download — are ignored rather than
        guessed at, because a model download's byte progress landing in this
        tab's bar would be worse than silence.
        """
        name = self._runtime_of(event.task_id)
        if name is None:
            return
        if event.task_id.endswith(SIZE_SUFFIX):
            if event.kind == "log":
                self._status.setText(event.message)
            elif event.kind == "done":
                self._size_items[name].setText(event.message or SIZE_UNAVAILABLE)
            return
        if event.kind in ("stage", "log"):
            self._status.setText(event.message)
        elif event.kind == "device_progress":
            self._show_progress(event.processed, event.total)
        elif event.kind == "error":
            self._installing.discard(name)
            self._status.setText(event.message)
            self._refresh_row(name)
        elif event.kind == "done":
            self._installing.discard(name)
            self._status.setText(event.message)
            self._bar.setRange(0, _BAR_STEPS)
            self._bar.setValue(_BAR_STEPS)
            self._refresh_row(name)
            self._refresh_restart()

    def _show_progress(self, processed: int, total: int) -> None:
        """Show byte progress as a proportion, with the bytes beside it.

        `QProgressBar` takes a 32-bit range and a CUDA wheel is several
        gigabytes, so the bar carries the proportion and the status line
        carries the exact counts. A `total` of zero — which is how the
        downloader says nobody knows the size — leaves the bar indeterminate.
        """
        if total <= 0:
            self._bar.setRange(0, 0)
            return
        self._bar.setRange(0, _BAR_STEPS)
        self._bar.setValue(min(_BAR_STEPS, _BAR_STEPS * processed // total))
        self._status.setText(f"{human_bytes(processed)} of {human_bytes(total)}")

    def row_of(self, name: str) -> int:
        """The table row showing `name`, or -1 when there is none."""
        return self._rows.get(name, -1)

    def _runtime_of(self, task_id: str) -> str | None:
        """The runtime a task id belongs to, or `None` when it is not ours."""
        if not task_id.startswith(TASK_PREFIX):
            return None
        name = task_id[len(TASK_PREFIX) :]
        if name.endswith(SIZE_SUFFIX):
            name = name[: -len(SIZE_SUFFIX)]
        return name if name in self._buttons else None

    def _spec(self, name: str) -> RuntimeSpec:
        """The spec called `name`.

        Raises:
            KeyError: If no runtime is called `name`, so a typo in a front-end
                fails at the call rather than silently doing nothing.
        """
        for spec in self._specs:
            if spec.name == name:
                return spec
        raise KeyError(f"no runtime called {name!r}")

    def _unavailable_reason(self, spec: RuntimeSpec) -> str:
        """Why this host cannot install `spec`, or `""` when it can.

        A bundled runtime has no reason: it is not missing, and saying it was
        unavailable would put a fault on the row that has none.
        """
        if spec.bundled:
            return ""
        if spec.name == MIGRAPHX and self._rocm is None:
            return MIGRAPHX_UNAVAILABLE
        return ""

    def _probe_sizes(self) -> None:
        """Ask the host how big each installable runtime is, one task each.

        Every runtime is resolved independently, so a package that PyPI cannot
        answer for leaves the rest of the row's figure intact. The answer is
        not cached across processes: the wheels move, and a figure from a
        previous run is exactly the fabricated number this tab exists to avoid.
        """
        for spec in self._specs:
            if not self.is_offered(spec.name):
                continue
            task_id = f"{TASK_PREFIX}{spec.name}{SIZE_SUFFIX}"
            self._app.submit(task_id, self._size_task(spec, task_id))

    def _size_task(self, spec: RuntimeSpec, task_id: str) -> TaskFn:
        """The body of one size probe: resolve, then report what came back."""

        def task(emit: Emit, stop: threading.Event) -> int | None:
            del stop
            try:
                total = runtime_size(spec)
            except DownloadError as exc:
                emit(PipelineEvent(kind="log", task_id=task_id, message=str(exc)))
                emit(
                    PipelineEvent(
                        kind="done", task_id=task_id, message=SIZE_UNAVAILABLE
                    )
                )
                return None
            emit(
                PipelineEvent(
                    kind="done",
                    task_id=task_id,
                    message=human_bytes(total),
                    processed=total,
                    total=total,
                )
            )
            return total

        return task

    def _build_ui(self) -> None:
        """Lay out the rows, the progress bar, the restart banner and the log."""
        layout = QVBoxLayout(self)
        intro = QLabel(
            "The slim build ships no GPU runtime. The download size is resolved "
            "from the host before anything is fetched, and a restart applies it."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._table = QTableWidget(0, len(_HEADERS))
        self._table.setHorizontalHeaderLabels(_HEADERS)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        for column in (_COLUMN_SUMMARY, _COLUMN_REQUIRES):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(
            _COLUMN_SIZE, QHeaderView.ResizeMode.ResizeToContents
        )
        header.setSectionResizeMode(
            _COLUMN_STATE, QHeaderView.ResizeMode.ResizeToContents
        )
        layout.addWidget(self._table)
        for spec in self._specs:
            self._add_row(spec)

        self._detail = QLabel()
        self._detail.setWordWrap(True)
        self._detail.setTextFormat(Qt.TextFormat.PlainText)
        self._detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._detail)
        self._table.itemSelectionChanged.connect(self._show_selected_detail)
        self._table.selectRow(0)

        activity = QHBoxLayout()
        self._bar = QProgressBar()
        self._bar.setRange(0, 1)
        self._status = QLabel("Idle")
        self._status.setWordWrap(True)
        activity.addWidget(self._bar, 2)
        activity.addWidget(self._status, 5)
        layout.addLayout(activity)

        banner_row = QHBoxLayout()
        self._restart_banner = QLabel(RESTART_MESSAGE)
        self._restart_banner.setStyleSheet("font-weight: 600; color: #b03030;")
        self._restart_button = QPushButton("Restart now")
        self._restart_button.clicked.connect(self.restart)
        banner_row.addWidget(self._restart_banner)
        banner_row.addStretch(1)
        banner_row.addWidget(self._restart_button)
        layout.addLayout(banner_row)

        self._resolved = QLabel()
        self._resolved.setTextFormat(Qt.TextFormat.PlainText)
        self._resolved.setStyleSheet("font-family: monospace;")
        layout.addWidget(self._resolved)

    def _add_row(self, spec: RuntimeSpec) -> None:
        """Add one runtime's row, then state it as the host sees it."""
        row = self._table.rowCount()
        self._table.insertRow(row)
        summary = QTableWidgetItem(spec.summary)
        summary.setData(Qt.ItemDataRole.UserRole, spec.name)
        requires = QTableWidgetItem("; ".join(spec.requirements))
        size = QTableWidgetItem(self._initial_size(spec))
        state = QTableWidgetItem(STATE_MISSING)
        button = QPushButton("Install")
        button.clicked.connect(
            lambda _checked=False, name=spec.name: self.install(name)
        )
        self._table.setItem(row, _COLUMN_SUMMARY, summary)
        self._table.setItem(row, _COLUMN_REQUIRES, requires)
        self._table.setItem(row, _COLUMN_SIZE, size)
        self._table.setItem(row, _COLUMN_STATE, state)
        self._table.setCellWidget(row, _COLUMN_ACTION, button)
        self._rows[spec.name] = row
        self._names_by_row[row] = spec.name
        self._requires_items[spec.name] = requires
        self._size_items[spec.name] = size
        self._state_items[spec.name] = state
        self._buttons[spec.name] = button
        self._refresh_row(spec.name)

    def _initial_size(self, spec: RuntimeSpec) -> str:
        """What the size cell says before anything has been resolved.

        A runtime the host cannot offer would read `calculating…` forever, and
        a row that looks like it is waiting for an answer that is never coming
        is worse than one that says so.
        """
        if spec.bundled:
            return "bundled"
        if self._unavailable_reason(spec):
            return "not available"
        return CALCULATING

    def _refresh_row(self, name: str) -> None:
        """Re-state one row from what is on disk and what the host offers."""
        spec = self._spec(name)
        reason = self._unavailable_reason(spec)
        installed = site_packages_dir(spec.directory) is not None
        if spec.bundled or installed:
            self._state_items[name].setText(STATE_INSTALLED)
        elif reason:
            self._state_items[name].setText(STATE_UNAVAILABLE)
        else:
            self._state_items[name].setText(STATE_MISSING)
        requires = reason or "; ".join(spec.requirements)
        self._requires_items[name].setText(requires)
        self._state_items[name].setToolTip(requires)
        self._buttons[name].setEnabled(
            not spec.bundled
            and not reason
            and not installed
            and name not in self._installing
        )

    def _show_selected_detail(self) -> None:
        """Write the full detail of the selected runtime, wheel names included.

        The table has room for a summary and a requirement; the exact wheel
        names, the source and what is on disk do not fit a cell, and they are
        the details a user needs before committing to a download.
        """
        selection = self._table.selectionModel()
        if selection is None:
            return
        rows = selection.selectedRows()
        if not rows:
            return
        name = self._names_by_row.get(rows[0].row())
        if name is None:
            return
        spec = self._spec(name)
        reason = self._unavailable_reason(spec)
        tree = site_packages_dir(spec.directory)
        lines = [spec.summary, "", f"Requires: {'; '.join(spec.requirements)}"]
        if reason:
            lines.append(f"Not available here: {reason}")
        if spec.packages:
            lines.append("")
            lines.append("Installs:")
            lines.extend(f"  {package}" for package in spec.packages)
        if spec.source_url:
            lines.extend(("", f"Source: {spec.source_url}"))
        if spec.bundled:
            lines.extend(("", "Already available: the interpreter's own copy."))
        else:
            lines.extend(("", f"Installed at: {tree if tree else 'not installed'}"))
        self._detail.setText("\n".join(lines))

    def _show_resolved(self) -> None:
        """Show the two resolved paths in the window as well as in the log."""
        self._resolved.setText(
            "\n".join(
                f"{module}: {resolved_origin(module) or 'not installed'}"
                for module in ("onnxruntime", "torch")
            )
        )

    def _refresh_restart(self) -> None:
        """Show the restart banner exactly when a restart is what is needed."""
        needed = self.is_restart_required()
        self._restart_banner.setVisible(needed)
        self._restart_button.setEnabled(needed)
