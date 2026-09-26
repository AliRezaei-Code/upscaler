"""The Runtimes tab, driven as a user drives it: rows, buttons and labels.

Every test builds the real widget offscreen and reads what it shows. The
network is off in all of them — the size probes are the only thing here that
talks to PyPI, and they are pointed at a stub that answers with a fixed JSON
document, because a test suite that silently depended on pypi.org would be a
test suite that fails on a Sunday.

There is no `pytest-qt` and no `qtbot` here, so the Qt application is a
session fixture and the events the main window would normally deliver through
`worker_bridge.EventPump` are drained by hand. That is deliberate: it is the
same path a headless smoke takes, and it proves the tab works with no pump at
all.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
import time
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
)

from core import devices, paths
from core.app import UpscalerApp
from core.download import DownloadResult
from core.errors import DownloadError
from core.events import PipelineEvent
from core.paths import runtimes_dir
from ui_pyside import runtimes_tab as rt
from ui_pyside.runtimes_tab import RUNTIMES, RuntimesTab

#: The `lib/python3.12` of this interpreter's runtime trees.
PYTHON_DIR = f"python{sys.version_info.major}.{sys.version_info.minor}"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qapp() -> Iterator[QApplication]:
    """The one Qt application for the session; Qt allows no more than one."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def data_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Send `runtimes_dir()` into the test's own tree.

    `core.paths.data_dir` re-reads its environment on every call, so setting it
    here rather than patching the function exercises the real path code. The
    branch is forced onto the XDG one so the redirect takes on every platform.
    """
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))


@pytest.fixture(autouse=True)
def host_without_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the host has no ROCm, which is true of every machine but one."""
    monkeypatch.setattr(devices, "rocm_version", lambda: None)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test reaches the network by accident.

    The default answer is a host that cannot be reached, so a widget test that
    does not care about sizes gets `size unavailable` — itself a state worth
    having exercised on every run.
    """

    def offline_json(url: str) -> dict[str, Any]:
        raise DownloadError(f"Could not reach {url}: the test is offline")

    def offline_head(url: str) -> int:
        raise DownloadError(f"Could not reach {url}: the test is offline")

    monkeypatch.setattr(rt, "_http_json", offline_json)
    monkeypatch.setattr(rt, "_content_length", offline_head)


@pytest.fixture
def make_tab(qapp: QApplication) -> Iterator[Callable[..., RuntimesTab]]:
    """Build tabs, deleting every one of them at the end of the test."""
    built: list[RuntimesTab] = []

    def build(app: UpscalerApp | None = None) -> RuntimesTab:
        widget = RuntimesTab(app or UpscalerApp())
        widget.show()
        qapp.processEvents()
        built.append(widget)
        return widget

    yield build
    # Every task has to be finished before its patches are undone, or a
    # thread that is still resolving a size would call the real PyPI.
    settle(qapp)
    for widget in built:
        widget.deleteLater()
    qapp.processEvents()


# --------------------------------------------------------------------------
# driving the tab the way the main window does
# --------------------------------------------------------------------------


def settle(qapp: QApplication, timeout: float = 20.0) -> None:
    """Wait until every task this app started has finished.

    `UpscalerApp.submit` runs each task on a named daemon thread and pushes a
    sentinel when it ends, so waiting for the threads is what makes the drain
    below safe: `events()` blocks forever on an empty queue.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not [t for t in threading.enumerate() if t.name.startswith("upscaler:")]:
            qapp.processEvents()
            return
        qapp.processEvents(QEventLoop.AllEvents, 20)
    raise AssertionError(f"a runtime task did not finish within {timeout:.0f} s")


def size_probes(tab: RuntimesTab) -> int:
    """How many size probes the tab started when it was built."""
    return sum(1 for spec in tab.specs if tab.is_offered(spec.name))


def drain(app: UpscalerApp, tab: RuntimesTab, tasks: int) -> None:
    """Hand the tab every event of `tasks` finished tasks, in order.

    This is what `worker_bridge.EventPump` does on a real run. The count is
    the caller's because `events()` returns at a task's sentinel and blocks on
    an empty queue, so a test has to know how many tasks it started.
    """
    for _ in range(tasks):
        for event in app.events():
            tab.handle_event(event)


# --------------------------------------------------------------------------
# reading the widget
# --------------------------------------------------------------------------


def table_of(tab: RuntimesTab) -> QTableWidget:
    """The tab's runtime table."""
    table = tab.findChild(QTableWidget)
    assert table is not None, "the tab has no runtime table"
    return table


def column_of(header: str) -> int:
    """The index of the column with that heading."""
    for column, expected in enumerate(rt._HEADERS):
        if expected == header:
            return column
    raise AssertionError(f"no column headed {header!r}")


def cell(tab: RuntimesTab, name: str, header: str) -> str:
    """The text of one cell, as the table shows it."""
    row = tab.row_of(name)
    assert row >= 0, f"no row for {name!r}"
    item = table_of(tab).item(row, column_of(header))
    assert item is not None
    return item.text()


def button_of(tab: RuntimesTab, name: str) -> QPushButton:
    """The button in one row — the only widget cell the table has."""
    row = tab.row_of(name)
    table = table_of(tab)
    for column in range(table.columnCount()):
        widget = table.cellWidget(row, column)
        if isinstance(widget, QPushButton):
            return widget
    raise AssertionError(f"row {row} has no button")


def restart_button(tab: RuntimesTab) -> QPushButton:
    """The Restart button, which lives outside the table."""
    for button in tab.findChildren(QPushButton):
        if button.text() == "Restart now":
            return button
    raise AssertionError("the tab has no Restart button")


def banner(tab: RuntimesTab) -> QLabel:
    """The label that says a restart is what is needed."""
    for label in tab.findChildren(QLabel):
        if label.text() == rt.RESTART_MESSAGE:
            return label
    raise AssertionError("the tab has no restart banner")


def texts(tab: RuntimesTab) -> list[str]:
    """Every non-empty label the tab is currently showing."""
    return [label.text() for label in tab.findChildren(QLabel) if label.text()]


def visible_text(tab: RuntimesTab) -> str:
    """Every label the tab is showing, as one string to look for text in."""
    return "\n".join(texts(tab))


def bar_of(tab: RuntimesTab) -> QProgressBar:
    """The tab's progress bar."""
    bar = tab.findChild(QProgressBar)
    assert bar is not None, "the tab has no progress bar"
    return bar


# --------------------------------------------------------------------------
# the stand-ins for the outside world
# --------------------------------------------------------------------------


def platform_tag() -> str:
    """A wheel platform tag this machine would accept, per the tab's own rule."""
    arches = rt._arch_spellings()
    if sys.platform == "win32":
        return f"win_{'amd64' if 'amd64' in arches else arches[0]}"
    if sys.platform == "darwin":
        return f"macosx_14_0_{'x86_64' if 'x86_64' in arches else arches[0]}"
    return f"manylinux_2_28_{arches[0]}.manylinux_2_28_{arches[0]}"


def wheel_name(project: str, version: str = "1.0.0") -> str:
    """A wheel filename matching this interpreter and machine."""
    return f"{project}-{version}-{rt._python_tag()}-abi3-{platform_tag()}.whl"


def pypi_stub(sizes: dict[str, int]) -> Callable[[str], dict[str, Any]]:
    """A PyPI that answers every release with one wheel of the given size."""

    def fetch(url: str) -> dict[str, Any]:
        name = url.split("/pypi/")[1].split("/")[0]
        filename = wheel_name(name.replace("-", "_"))
        return {
            "info": {"name": name},
            "urls": [
                {
                    "packagetype": "bdist_wheel",
                    "filename": filename,
                    "url": f"https://files.pythonhosted.org/{filename}",
                    "size": sizes[name],
                    "yanked": False,
                }
            ],
        }

    return fetch


def wheel_bytes(module: str = "onnxruntime_gpu") -> bytes:
    """A real zip shaped like a wheel, with a `.data` tree to place."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{module}/__init__.py", "# a runtime module\n")
        archive.writestr(f"{module}-1.0.0.dist-info/METADATA", "Name: x\n")
        archive.writestr("x-1.0.0.data/purelib/from_data_lib.py", "# rebased\n")
        archive.writestr("x-1.0.0.data/scripts/a-script", "#!/bin/sh\n")
    return buffer.getvalue()


def fetcher_stub(payload: bytes) -> Callable[..., DownloadResult]:
    """A `download_file` that writes `payload` instead of going to the network."""

    def fetch(
        url: str,
        dest: Path,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
        emit: Callable[[PipelineEvent], None] | None = None,
        **kwargs: Any,
    ) -> DownloadResult:
        del expected_sha256, kwargs
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        assert expected_size, "the install downloaded before the size was resolved"
        if emit is not None:
            emit(
                PipelineEvent(
                    kind="device_progress",
                    processed=len(payload),
                    total=len(payload),
                    message=dest.name,
                )
            )
        return DownloadResult(path=dest, bytes_written=len(payload), resumed_from=0)

    return fetch


CUDA_128_SIZES = {"onnxruntime-gpu": 2_000_000_000, "torch": 1_000_000_000}


# --------------------------------------------------------------------------
# the rows
# --------------------------------------------------------------------------


def test_the_four_runtimes_are_listed(make_tab: Callable[..., RuntimesTab]) -> None:
    tab = make_tab()
    assert [spec.name for spec in tab.specs] == [
        "cpu",
        "cuda-12.8",
        "cuda-13.0",
        "migraphx",
    ]
    shown = [cell(tab, spec.name, "Runtime") for spec in RUNTIMES]
    assert shown == [spec.summary for spec in RUNTIMES]
    assert shown == [
        "CPU (bundled)",
        "CUDA 12.8 (ONNX Runtime 1.26.0 + torch 2.7.1)",
        "CUDA 13.0 (ONNX Runtime 1.30.0 + torch latest)",
        "MIGraphX (ROCm 7.2)",
    ]


def test_cpu_is_present_and_never_installable(
    make_tab: Callable[..., RuntimesTab], qapp: QApplication
) -> None:
    app = UpscalerApp()
    tab = make_tab(app=app)
    assert cell(tab, "cpu", "State") == "installed"
    assert button_of(tab, "cpu").isEnabled() is False

    tab.install("cpu")
    settle(qapp)
    assert [t for t in threading.enumerate() if t.name.startswith("upscaler:")] == []
    assert cell(tab, "cpu", "State") == "installed"
    assert button_of(tab, "cpu").isEnabled() is False
    # The bundled runtime is the interpreter's own; there is no tree to point at.
    assert tab.resolved("cpu") is None


def test_cuda_rows_start_missing_and_offered(
    make_tab: Callable[..., RuntimesTab],
) -> None:
    tab = make_tab()
    for name in ("cuda-12.8", "cuda-13.0"):
        assert cell(tab, name, "State") == "not installed"
        assert button_of(tab, name).isEnabled() is True
    assert cell(tab, "cuda-12.8", "Requires") == "a CUDA 12.8 driver, 525 or newer"
    assert cell(tab, "cuda-13.0", "Requires") == "a CUDA 13.0 driver, 580 or newer"


def test_migraphx_is_unavailable_without_rocm(
    make_tab: Callable[..., RuntimesTab],
) -> None:
    tab = make_tab()
    assert cell(tab, "migraphx", "State") == "unavailable"
    assert button_of(tab, "migraphx").isEnabled() is False
    assert cell(tab, "migraphx", "Requires") == (
        "AMD GPU acceleration on Linux requires ROCm 7.x; see the Runtimes tab"
    )
    assert tab.is_offered("migraphx") is False
    # A row that can never be installed must not sit on "calculating…".
    assert cell(tab, "migraphx", "Size") == "not available"


def test_migraphx_is_offered_on_a_rocm_host(
    make_tab: Callable[..., RuntimesTab], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(devices, "rocm_version", lambda: "7.2.0")
    tab = make_tab()
    assert tab.is_offered("migraphx") is True
    assert cell(tab, "migraphx", "State") == "not installed"
    assert button_of(tab, "migraphx").isEnabled() is True
    assert cell(tab, "migraphx", "Requires") == "ROCm 7.2.0 at /opt/rocm"

    table_of(tab).selectRow(tab.row_of("migraphx"))
    detail = visible_text(tab)
    assert "onnxruntime_migraphx-1.23.2-cp312-cp312-" in detail
    assert "torch-2.12.1%2Brocm7.2-cp312-cp312-" in detail
    assert "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/" in detail


def test_an_unknown_runtime_is_an_error(make_tab: Callable[..., RuntimesTab]) -> None:
    tab = make_tab()
    with pytest.raises(KeyError, match="no runtime called 'rocm'"):
        tab.install("rocm")
    assert tab.resolved("rocm") is None


# --------------------------------------------------------------------------
# the size the user is shown before committing
# --------------------------------------------------------------------------


def test_size_starts_as_calculating(make_tab: Callable[..., RuntimesTab]) -> None:
    tab = make_tab()
    assert cell(tab, "cuda-12.8", "Size") == "calculating…"
    assert cell(tab, "cpu", "Size") == "bundled"


def test_sizes_are_scaled_without_losing_the_exact_count() -> None:
    """The unit arithmetic is the one thing a user reads before committing."""
    assert rt.human_bytes(0) == "0 bytes"
    assert rt.human_bytes(1023) == "1,023 bytes"
    assert rt.human_bytes(1024) == "1.00 KB (1,024 bytes)"
    assert rt.human_bytes(1024**2) == "1.00 MB (1,048,576 bytes)"
    assert rt.human_bytes(3 * 1024**3) == "3.00 GB (3,221,225,472 bytes)"
    assert rt.human_bytes(2 * 1024**4) == "2.00 TB (2,199,023,255,552 bytes)"


def test_resolved_size_is_shown_with_the_exact_byte_count(
    make_tab: Callable[..., RuntimesTab],
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rt, "_http_json", pypi_stub(CUDA_128_SIZES))
    app = UpscalerApp()
    tab = make_tab(app=app)
    settle(qapp)
    drain(app, tab, size_probes(tab))

    assert cell(tab, "cuda-12.8", "Size") == "2.79 GB (3,000,000,000 bytes)"
    assert cell(tab, "cuda-13.0", "Size") == "2.79 GB (3,000,000,000 bytes)"
    assert "calculating" not in cell(tab, "cuda-12.8", "Size")


def test_unreachable_pypi_shows_no_number(
    make_tab: Callable[..., RuntimesTab], qapp: QApplication
) -> None:
    app = UpscalerApp()
    tab = make_tab(app=app)
    settle(qapp)
    drain(app, tab, size_probes(tab))

    assert cell(tab, "cuda-12.8", "Size") == "size unavailable"
    assert not any(character.isdigit() for character in cell(tab, "cuda-12.8", "Size"))
    assert "the test is offline" in visible_text(tab)


# --------------------------------------------------------------------------
# installing
# --------------------------------------------------------------------------


def stub_the_world(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    """Point PyPI and the downloader at fixed, local answers.

    `payload` is what arrives in place of a wheel; a test that wants a good
    install passes a real zip and one that wants a failure passes anything
    that is not one.
    """
    monkeypatch.setattr(rt, "_http_json", pypi_stub(CUDA_128_SIZES))
    monkeypatch.setattr(rt, "download_file", fetcher_stub(payload))


def run_install(tab: RuntimesTab, app: UpscalerApp, qapp: QApplication) -> None:
    """Click Install on `cuda-12.8` and deliver every event, as a pump would."""
    tab.install("cuda-12.8")
    settle(qapp)
    drain(app, tab, size_probes(tab) + 1)


def test_a_successful_install_unpacks_and_marks_the_row(
    make_tab: Callable[..., RuntimesTab],
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_the_world(monkeypatch, wheel_bytes())
    app = UpscalerApp()
    tab = make_tab(app=app)
    run_install(tab, app, qapp)

    site = runtimes_dir() / "cuda-12.8" / "lib" / PYTHON_DIR / "site-packages"
    assert (site / "onnxruntime_gpu" / "__init__.py").is_file()
    # `.data/purelib` is the wheel's own name for "this goes in site-packages".
    assert (site / "from_data_lib.py").is_file()
    # A script has nowhere to go in a runtime directory, and is not invented.
    assert not (site / "x-1.0.0.data").exists()
    assert cell(tab, "cuda-12.8", "State") == "installed"
    assert button_of(tab, "cuda-12.8").isEnabled() is False
    assert tab.resolved("cuda-12.8") == site
    assert not (runtimes_dir() / "cuda-12.8" / "lib.tmp").exists()
    assert tab.is_restart_required() is True


def test_an_install_that_fails_leaves_nothing_behind(
    make_tab: Callable[..., RuntimesTab],
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_the_world(monkeypatch, b"this is not a zip")
    app = UpscalerApp()
    tab = make_tab(app=app)
    run_install(tab, app, qapp)

    assert not (runtimes_dir() / "cuda-12.8" / "lib").exists()
    assert not (runtimes_dir() / "cuda-12.8" / "lib.tmp").exists()
    assert cell(tab, "cuda-12.8", "State") == "not installed"
    assert button_of(tab, "cuda-12.8").isEnabled() is True
    assert "File is not a zip file" in app.last_error["runtime:cuda-12.8"]
    assert "File is not a zip file" in visible_text(tab)


def test_a_requirement_with_no_matching_wheel_is_refused_by_name(
    make_tab: Callable[..., RuntimesTab],
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_the_world(monkeypatch, wheel_bytes())
    monkeypatch.setattr(rt, "_wheel_matches", lambda filename: False)
    app = UpscalerApp()
    tab = make_tab(app=app)
    run_install(tab, app, qapp)

    assert (
        "onnxruntime-gpu[cuda,cudnn]==1.26.0 has no wheel for"
        in app.last_error["runtime:cuda-12.8"]
    )
    assert cell(tab, "cuda-12.8", "Size") == "size unavailable"
    assert not (runtimes_dir() / "cuda-12.8" / "lib.tmp").exists()


# --------------------------------------------------------------------------
# the restart
# --------------------------------------------------------------------------


def test_restart_is_required_only_after_an_install(
    make_tab: Callable[..., RuntimesTab],
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_the_world(monkeypatch, wheel_bytes())
    app = UpscalerApp()
    tab = make_tab(app=app)

    assert tab.is_restart_required() is False
    assert restart_button(tab).isEnabled() is False
    assert banner(tab).isVisible() is False

    run_install(tab, app, qapp)

    assert tab.is_restart_required() is True
    assert RuntimesTab.pending_restart_marker().is_file()
    assert banner(tab).isVisible() is True
    assert restart_button(tab).isEnabled() is True


def test_restart_replaces_the_process(
    make_tab: Callable[..., RuntimesTab],
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_the_world(monkeypatch, wheel_bytes())
    app = UpscalerApp()
    tab = make_tab(app=app)
    run_install(tab, app, qapp)

    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(os, "execv", lambda path, argv: calls.append((path, argv)))
    tab.restart()

    assert calls == [(sys.executable, [sys.executable, *sys.argv])]


def test_the_marker_is_consumed_at_the_next_start(
    make_tab: Callable[..., RuntimesTab], caplog: pytest.LogCaptureFixture
) -> None:
    RuntimesTab.pending_restart_marker().parent.mkdir(parents=True, exist_ok=True)
    RuntimesTab.pending_restart_marker().write_text("cuda-12.8\n", encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="ui_pyside.runtimes_tab"):
        tab = make_tab()
    assert tab.is_restart_required() is False
    assert not RuntimesTab.pending_restart_marker().exists()
    assert "a runtime was installed" in caplog.text


# --------------------------------------------------------------------------
# startup
# --------------------------------------------------------------------------


def test_startup_logs_the_resolved_modules(
    make_tab: Callable[..., RuntimesTab], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="ui_pyside.runtimes_tab"):
        tab = make_tab()
    for module in ("onnxruntime", "torch"):
        origin = rt.resolved_origin(module)
        assert origin is not None, f"{module} is not installed in this environment"
        assert f"resolved {module}: {origin}" in caplog.text
    assert f"onnxruntime: {rt.resolved_origin('onnxruntime')}" in visible_text(tab)


def test_startup_survives_both_modules_being_absent(
    make_tab: Callable[..., RuntimesTab],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module in ("onnxruntime", "torch"):
        monkeypatch.setitem(sys.modules, module, None)
    with caplog.at_level(logging.INFO, logger="ui_pyside.runtimes_tab"):
        tab = make_tab()
    assert "resolved onnxruntime: not installed" in caplog.text
    assert "resolved torch: not installed" in caplog.text
    assert cell(tab, "cpu", "State") == "installed"


def test_the_tab_works_with_no_event_pump(
    make_tab: Callable[..., RuntimesTab], qapp: QApplication
) -> None:
    """The main window's headless smoke builds a tab with no pump at all."""
    app = UpscalerApp()
    tab = make_tab(app=app)
    tab.show()
    qapp.processEvents()
    assert table_of(tab).rowCount() == len(RUNTIMES)
    settle(qapp)
    assert [cell(tab, spec.name, "State") for spec in RUNTIMES][:2] == [
        "installed",
        "not installed",
    ]


def test_another_producers_events_are_left_alone(
    make_tab: Callable[..., RuntimesTab],
) -> None:
    tab = make_tab()
    before = bar_of(tab).value()
    tab.handle_event(
        PipelineEvent(kind="device_progress", task_id="job", processed=9, total=10)
    )
    tab.handle_event(
        PipelineEvent(kind="error", task_id="model:x4plus", message="a model failed")
    )
    assert bar_of(tab).value() == before
    assert "a model failed" not in visible_text(tab)
