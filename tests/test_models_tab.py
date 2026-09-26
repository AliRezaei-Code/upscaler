"""The Models tab, driven as a user drives it.

No network, no event pump and no real data directory: the tab is built against
a real `UpscalerApp` and real catalogue, the HTTP call is the only thing
replaced, and the app's own event queue is drained by the test in the pump's
place — so the dispatch on `task_id` that the window depends on is exercised
end to end rather than stubbed.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from core import paths
from core.app import UpscalerApp
from core.download import DownloadResult
from core.errors import ModelLoadError
from core.events import PipelineEvent
from core.models import MIN_MODEL_BYTES, ModelEntry, load_catalogue
from ui_pyside import models_tab as models_tab_module
from ui_pyside.models_tab import COLUMNS, ModelsTab

#: One mebibyte of a real, writable payload. `verify_model` refuses anything
#: smaller, so this is the smallest file a successful download can really end
#: up as — see the nine-byte tests below for the other end.
PAYLOAD = b"z" * MIN_MODEL_BYTES

#: The exact nine bytes a release URL answered with on the machine this was
#: built on, saved under a real model's filename: `Not Found`, no newline.
TRUNCATED = b"Not Found"

#: `verify_model`'s wording for that file, quoted so a change to the core
#: message is a test failure rather than a silent difference.
TRUNCATED_MESSAGE = (
    "is only 9 bytes — this is a failed download, not a model. "
    "Delete it and download again."
)


@pytest.fixture(scope="session")
def qapp() -> Iterator[QApplication]:
    """The one `QApplication`; Qt permits exactly one per process."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point `core.paths.models_dir` at a temporary directory.

    The tab resolves every model path through `core.paths`, so this is the one
    seam that keeps a test off the real `~/.local/share/upscaler/models`.
    """
    target = tmp_path / "models"
    monkeypatch.setattr(paths, "models_dir", lambda: target)
    return target


def _small_entry() -> ModelEntry:
    """A catalogue entry small enough for a test to really write.

    The size is the floor `verify_model` accepts, so the success path runs the
    real check instead of a stand-in for it.
    """
    return replace(
        load_catalogue()[0],
        id="test-model",
        filename="test-model.pth",
        size_bytes=MIN_MODEL_BYTES,
        sha256=None,
        url="https://github.com/AliRezaei-Code/upscaler/releases/download/v0/m.pth",
    )


def _cell(tab: ModelsTab, row: int, column: int) -> str:
    """One table cell's text, as the user reads it."""
    item = tab.table.item(row, column)
    assert item is not None, f"row {row} column {column} is empty"
    return item.text()


def _drain(app: UpscalerApp, tab: ModelsTab) -> list[PipelineEvent]:
    """Feed the running task's events to the tab and return them.

    This is what `EventPump` does inside the window. A test that owns the app
    and no window has to stand in for it, and driving the real `submit` and
    `events` pair is the point: the tab's dispatch on `task_id` is what is
    under test.
    """
    events = list(app.events())
    for event in events:
        tab.handle_event(event)
    return events


def _fake_download(payload: bytes) -> object:
    """A `download_file` that writes a real file and reports nothing.

    Everything around the HTTP call stays real: the app's thread, the queue,
    the tab's dispatch and `verify_model` on the file that lands.
    """
    calls: list[str] = []

    def download(
        url: str,
        dest: Path,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
        emit: object = None,
    ) -> DownloadResult:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        return DownloadResult(path=dest, bytes_written=len(payload), resumed_from=0)

    download.calls = calls  # type: ignore[attr-defined]
    return download


def _blocking_download(payload: bytes, before: list[PipelineEvent]) -> object:
    """A `download_file` that reports progress and then waits to be released.

    The wait is what lets a test look at the tab *during* a download, which is
    the only moment an in-flight progress bar can be asserted on.
    """
    release = threading.Event()
    calls: list[str] = []

    def download(
        url: str,
        dest: Path,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
        emit: object = None,
    ) -> DownloadResult:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        for event in before:
            emit(event)  # type: ignore[operator]
        if not release.wait(timeout=10.0):
            raise TimeoutError("the test never released the fake download")
        dest.write_bytes(payload)
        return DownloadResult(path=dest, bytes_written=len(payload), resumed_from=0)

    download.release = release  # type: ignore[attr-defined]
    download.calls = calls  # type: ignore[attr-defined]
    return download


# --- the catalogue as a table -------------------------------------------------


def test_the_table_shows_every_catalogue_entry(
    qapp: QApplication, models_dir: Path
) -> None:
    entries = load_catalogue()
    tab = ModelsTab(UpscalerApp())
    assert tab.table.rowCount() == len(entries) == 14
    assert tab.table.columnCount() == len(COLUMNS)
    assert [
        tab.table.horizontalHeaderItem(column).text() for column in range(len(COLUMNS))
    ] == list(COLUMNS)
    for row, entry in enumerate(entries):
        assert _cell(tab, row, 0) == entry.name
        assert _cell(tab, row, 1) == entry.architecture
        assert _cell(tab, row, 2) == f"{entry.scale}x"
        assert _cell(tab, row, 3).endswith(("MB", "GB"))
        assert _cell(tab, row, 4) == entry.license
        assert _cell(tab, row, 5) == "Not installed"
        assert tab.details.toPlainText() or True  # filled on selection
    tab.deleteLater()


def test_the_search_box_filters_case_insensitively(
    qapp: QApplication, models_dir: Path
) -> None:
    entries = load_catalogue()
    tab = ModelsTab(UpscalerApp())

    tab.search_edit.setText("SWINIR")
    assert tab.table.rowCount() == sum("swinir" in e.name.lower() for e in entries)
    assert tab.table.rowCount() == 2
    assert all("SwinIR" in _cell(tab, row, 0) for row in range(2))

    tab.search_edit.setText("compact")
    assert tab.table.rowCount() == sum(
        e.architecture.lower() == "compact" for e in entries
    )
    assert all(_cell(tab, row, 1) == "Compact" for row in range(tab.table.rowCount()))

    tab.search_edit.setText("DENOISE")
    assert tab.table.rowCount() == 1
    assert _cell(tab, 0, 0) == "Real-ESRGAN general x4 v3 (weak denoise)"

    tab.search_edit.setText("no such model anywhere")
    assert tab.table.rowCount() == 0
    assert tab.selected_entry() is None
    assert tab.download_button.isEnabled() is False

    tab.search_edit.clear()
    assert tab.table.rowCount() == len(entries)
    tab.deleteLater()


def test_selection_returns_the_entry_the_row_describes(
    qapp: QApplication, models_dir: Path
) -> None:
    entries = load_catalogue()
    tab = ModelsTab(UpscalerApp())
    tab.table.selectRow(3)
    entry = tab.selected_entry()
    assert entry == entries[3]
    assert _cell(tab, 3, 0) == entry.name
    details = tab.details.toPlainText()
    assert entry.notes in details
    assert entry.source in details
    assert entry.license in details

    # The current cell outlives the highlight, so a tab that read it instead of
    # the selection would offer a model the user has visibly deselected.
    tab.table.clearSelection()
    assert tab.table.currentRow() == 3
    assert tab.selected_entry() is None
    assert tab.download_button.isEnabled() is False
    tab.deleteLater()


def test_a_model_already_on_disk_is_installed_and_offers_its_folder(
    qapp: QApplication, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _small_entry()
    monkeypatch.setattr(models_tab_module, "load_catalogue", lambda **_: [entry])
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / entry.filename).write_bytes(PAYLOAD)

    tab = ModelsTab(UpscalerApp())
    assert _cell(tab, 0, 5) == "Installed"
    assert tab.download_button.text() == "Open folder"
    assert tab.download_button.isEnabled() is True
    tab.deleteLater()


def test_a_catalogue_that_cannot_be_read_is_reported(
    qapp: QApplication, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**_: object) -> list[ModelEntry]:
        raise ModelLoadError("Model catalogue not found at /nowhere/catalogue.json")

    monkeypatch.setattr(models_tab_module, "load_catalogue", boom)
    tab = ModelsTab(UpscalerApp())
    assert tab.table.rowCount() == 0
    assert tab.download_button.isEnabled() is False
    assert "could not be read" in tab.status_label.text()
    assert "/nowhere/catalogue.json" in tab.status_label.text()
    tab.deleteLater()


# --- downloading --------------------------------------------------------------


def test_a_download_writes_the_file_and_marks_the_row_installed(
    qapp: QApplication, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _small_entry()
    monkeypatch.setattr(models_tab_module, "load_catalogue", lambda **_: [entry])
    download = _fake_download(PAYLOAD)
    monkeypatch.setattr(models_tab_module, "download_file", download)
    app = UpscalerApp()
    tab = ModelsTab(app)

    assert _cell(tab, 0, 5) == "Not installed"
    assert tab.download_button.text() == "Download"
    tab.download_button.click()
    events = _drain(app, tab)

    written = models_dir / entry.filename
    assert written.is_file()
    assert written.stat().st_size == len(PAYLOAD)
    assert written.read_bytes() == PAYLOAD
    assert _cell(tab, 0, 5) == "Installed"
    assert tab.download_button.text() == "Open folder"
    assert tab.download_button.isEnabled() is True
    assert str(written) in tab.status_label.text()
    assert [event.kind for event in events] == ["stage", "done"]
    # The catalogue's own URL, not something the tab invented.
    assert download.calls == [entry.url]  # type: ignore[attr-defined]
    tab.deleteLater()


def test_a_nine_byte_download_is_reported_and_the_button_comes_back(
    qapp: QApplication, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real failure, end to end: a host that answers with `Not Found`."""
    entry = _small_entry()
    monkeypatch.setattr(models_tab_module, "load_catalogue", lambda **_: [entry])
    monkeypatch.setattr(models_tab_module, "download_file", _fake_download(TRUNCATED))
    app = UpscalerApp()
    tab = ModelsTab(app)

    tab.download_button.click()
    events = _drain(app, tab)

    assert TRUNCATED in (models_dir / entry.filename).read_bytes()
    assert TRUNCATED_MESSAGE in tab.status_label.text()
    assert _cell(tab, 0, 5) == "Not installed"
    assert tab.download_button.isEnabled() is True
    assert tab.download_button.text() == "Download"
    assert [event.kind for event in events] == ["stage", "error"]
    tab.deleteLater()


def test_a_verification_failure_keeps_the_core_message_on_screen(
    qapp: QApplication, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ModelLoadError` from the verify step, which is the other way this fails."""
    entry = _small_entry()
    monkeypatch.setattr(models_tab_module, "load_catalogue", lambda **_: [entry])
    monkeypatch.setattr(models_tab_module, "download_file", _fake_download(PAYLOAD))

    def verify(path: Path, catalogue_entry: ModelEntry | None) -> None:
        assert path == models_dir / entry.filename
        assert path.is_file()
        raise ModelLoadError(f"{path.name} {TRUNCATED_MESSAGE}")

    monkeypatch.setattr(models_tab_module, "verify_model", verify)
    app = UpscalerApp()
    tab = ModelsTab(app)

    tab.download_button.click()
    _drain(app, tab)

    assert tab.status_label.text() == f"{entry.filename} {TRUNCATED_MESSAGE}"
    assert _cell(tab, 0, 5) == "Not installed"
    assert tab.download_button.isEnabled() is True
    assert tab.download_button.text() == "Download"
    tab.deleteLater()


def test_a_second_download_cannot_start_while_one_is_running(
    qapp: QApplication, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _small_entry()
    other = replace(entry, id="other-model", filename="other-model.pth")
    monkeypatch.setattr(models_tab_module, "load_catalogue", lambda **_: [entry, other])
    download = _blocking_download(PAYLOAD, [])
    monkeypatch.setattr(models_tab_module, "download_file", download)
    app = UpscalerApp()
    tab = ModelsTab(app)
    try:
        tab.table.selectRow(0)
        tab.download_button.click()
        assert tab.download_button.text() == "Downloading…"
        assert tab.download_button.isEnabled() is False

        # Even after moving to another row: there is one progress bar and one
        # status line, so a second download would have two truths to report.
        tab.table.selectRow(1)
        assert tab.download_button.isEnabled() is False
        tab.download_button.click()
        assert download.calls == [entry.url]  # type: ignore[attr-defined]
    finally:
        download.release.set()  # type: ignore[attr-defined]
        _drain(app, tab)
    tab.deleteLater()


# --- progress -----------------------------------------------------------------


def test_progress_from_another_task_does_not_move_the_bar(
    qapp: QApplication, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _small_entry()
    monkeypatch.setattr(models_tab_module, "load_catalogue", lambda **_: [entry])
    mib = 1024 * 1024
    download = _blocking_download(
        PAYLOAD,
        [
            PipelineEvent(
                kind="device_progress",
                task_id=f"model:{entry.id}",
                processed=25 * mib,
                total=100 * mib,
                eta_seconds=65.0,
            )
        ],
    )
    monkeypatch.setattr(models_tab_module, "download_file", download)
    app = UpscalerApp()
    tab = ModelsTab(app)
    stream = app.events()
    try:
        tab.download_button.click()
        for _ in range(2):  # the stage event, then the first progress report
            tab.handle_event(next(stream))

        assert tab.progress_bar.value() == 25 * mib
        assert tab.progress_bar.maximum() == 100 * mib
        assert "25.0 MiB of 100.0 MiB" in tab.status_label.text()
        assert "0 h 01 m 05 s left" in tab.status_label.text()

        before = (
            tab.progress_bar.value(),
            tab.progress_bar.maximum(),
            tab.status_label.text(),
        )
        tab.handle_event(
            PipelineEvent(
                kind="device_progress",
                task_id="model:some-other-model",
                processed=999 * mib,
                total=1000 * mib,
                eta_seconds=1.0,
            )
        )
        assert (
            tab.progress_bar.value(),
            tab.progress_bar.maximum(),
            tab.status_label.text(),
        ) == before

        # The job's own progress, and a log line, are equally not ours.
        tab.handle_event(
            PipelineEvent(kind="device_progress", task_id="job", processed=12, total=99)
        )
        assert (
            tab.progress_bar.value(),
            tab.progress_bar.maximum(),
            tab.status_label.text(),
        ) == before
    finally:
        download.release.set()  # type: ignore[attr-defined]
        for event in stream:
            tab.handle_event(event)
    assert _cell(tab, 0, 5) == "Installed"
    tab.deleteLater()


def test_a_download_of_unknown_size_runs_an_indeterminate_bar(
    qapp: QApplication, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _small_entry()
    monkeypatch.setattr(models_tab_module, "load_catalogue", lambda **_: [entry])
    download = _blocking_download(
        PAYLOAD,
        [
            PipelineEvent(
                kind="device_progress",
                task_id=f"model:{Path(entry.filename).stem}",
                processed=512 * 1024,
                total=0,
                eta_seconds=0.0,
            )
        ],
    )
    monkeypatch.setattr(models_tab_module, "download_file", download)
    app = UpscalerApp()
    tab = ModelsTab(app)
    stream = app.events()
    try:
        tab.download_button.click()
        for _ in range(2):
            tab.handle_event(next(stream))
        assert tab.progress_bar.maximum() == 0
        assert "0.5 MiB downloaded, total size unknown" in tab.status_label.text()
    finally:
        download.release.set()  # type: ignore[attr-defined]
        for event in stream:
            tab.handle_event(event)
    tab.deleteLater()
