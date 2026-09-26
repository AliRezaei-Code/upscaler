"""The Community tab, driven as a user drives it.

The HTTP fetch is the only thing replaced. The listing shape is the one the
live site produces on 2026-09-26: a card with no download link, and — because
`core.community` is built to notice the day that stops being true — one card
whose link is a host that does serve file bytes.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from core import paths
from core.app import UpscalerApp
from core.community import OPENMODELDB_URL, CommunityModel, host_for
from core.download import DownloadResult
from core.errors import DownloadError
from core.models import MIN_MODEL_BYTES
from ui_pyside import community_tab as community_tab_module
from ui_pyside.community_tab import DRIFT_MESSAGE, CommunityTab

#: One mebibyte, the smallest file `verify_model` will accept as a model.
PAYLOAD = b"z" * MIN_MODEL_BYTES

#: The listing as `fetch_openmodeldb` returns it today: the first card has no
#: download link at all, and the other two do because their pages do.
LISTING: tuple[CommunityModel, ...] = (
    CommunityModel(
        name="2x 90s Sonic",
        architecture="Compact",
        scale=2,
        author="joncusa",
        tags=("game", "sprite"),
        page_url="https://openmodeldb.info/models/2x-90s-Sonic",
        direct_url=None,
        host="other",
    ),
    CommunityModel(
        name="4x Waifu2x",
        architecture="Waifu2x",
        scale=4,
        author="AICasual",
        tags=("anime", "illustration"),
        page_url="https://openmodeldb.info/models/4x-waifu2x",
        direct_url=(
            "https://github.com/AICasual/waifu2x-ort/releases/download/v1.1/w.onnx"
        ),
        host="github",
    ),
    CommunityModel(
        name="4x Bird GUI",
        architecture="Compact",
        scale=4,
        author="maxim synthesizes",
        tags=("illustration",),
        page_url="https://openmodeldb.info/models/4x-bird-gui",
        direct_url="https://huggingface.co/somebody/bird/resolve/main/bird.pth",
        host="huggingface",
    ),
)

#: A landing page behind a file icon: the shape that must never be fetched.
DRIVE_URL = "https://drive.google.com/file/d/1AbC/view"


@pytest.fixture(scope="session")
def qapp() -> Iterator[QApplication]:
    """The one `QApplication`; Qt permits exactly one per process."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point `core.paths.models_dir` at a temporary directory."""
    target = tmp_path / "models"
    monkeypatch.setattr(paths, "models_dir", lambda: target)
    return target


class _FakeListing:
    """A `fetch_openmodeldb` stand-in that records what it was asked for.

    It can be made to block, so a test can catch the tab mid-refresh, and to
    fail, so a test can see what the tab does when the site has moved.
    """

    def __init__(
        self,
        models: tuple[CommunityModel, ...] = LISTING,
        *,
        error: Exception | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self.models = models
        self.error = error
        self.gate = gate
        self.calls: list[str] = []

    def __call__(
        self, url: str = OPENMODELDB_URL, timeout: float = 30.0
    ) -> list[CommunityModel]:
        self.calls.append(url)
        if self.gate is not None and not self.gate.wait(timeout=10.0):
            raise TimeoutError("the test never released the fake listing")
        if self.error is not None:
            raise self.error
        return list(self.models)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Point the tab at this stand-in."""
        monkeypatch.setattr(community_tab_module, "fetch_openmodeldb", self)


class _NoBrowser:
    """Stands in for `QDesktopServices`, which would open a real browser."""

    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.opened: list[str] = []

    def openUrl(self, url: QUrl) -> bool:
        self.opened.append(url.toString())
        return self.result


def _drain(app: UpscalerApp, tab: CommunityTab) -> list[object]:
    """Feed the running task's events to the tab, in the pump's place."""
    events = list(app.events())
    for event in events:
        tab.handle_event(event)
    return events


def _cell(tab: CommunityTab, row: int, column: int) -> str:
    """One table cell's text, as the user reads it."""
    item = tab.table.item(row, column)
    assert item is not None, f"row {row} column {column} is empty"
    return item.text()


def _listing(
    monkeypatch: pytest.MonkeyPatch, **kwargs: object
) -> tuple[UpscalerApp, CommunityTab, _FakeListing]:
    """A tab whose listing fetch is the stand-in."""
    fake = _FakeListing(**kwargs)  # type: ignore[arg-type]
    fake.install(monkeypatch)
    app = UpscalerApp()
    return app, CommunityTab(app), fake


# --- the listing --------------------------------------------------------------


def test_the_listing_populates_the_table(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, tab, fake = _listing(monkeypatch)
    tab.refresh()
    _drain(app, tab)

    assert fake.calls == [OPENMODELDB_URL]
    assert tab.table.rowCount() == 3
    assert tab.table.columnCount() == 4
    for row, model in enumerate(LISTING):
        assert _cell(tab, row, 0) == model.name
        assert _cell(tab, row, 1) == model.architecture
        assert _cell(tab, row, 2) == f"{model.scale}x"
        assert _cell(tab, row, 3) == model.author
    assert "3 models" in tab.status_label.text()
    assert tab.refresh_button.isEnabled() is True
    tab.deleteLater()


def test_selection_returns_the_model_the_row_describes(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, tab, _ = _listing(monkeypatch)
    tab.refresh()
    _drain(app, tab)

    tab.table.selectRow(1)
    assert tab.selected() == LISTING[1]
    details = tab.details.toPlainText()
    assert "4x" in details
    assert "by AICasual" in details
    assert "anime" in details
    assert LISTING[1].page_url in details
    assert tab.open_button.isEnabled() is True
    assert tab.copy_button.isEnabled() is True
    tab.deleteLater()


def test_the_filter_narrows_the_list(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, tab, _ = _listing(monkeypatch)
    tab.refresh()
    _drain(app, tab)

    tab.filter_edit.setText("sonic")
    assert tab.table.rowCount() == 1
    assert _cell(tab, 0, 0) == "2x 90s Sonic"

    tab.filter_edit.setText("COMPACT")
    assert tab.table.rowCount() == 2

    tab.filter_edit.setText("aicAsual")
    assert tab.table.rowCount() == 1
    assert _cell(tab, 0, 3) == "AICasual"

    tab.filter_edit.setText("nothing at all")
    assert tab.table.rowCount() == 0
    assert tab.selected() is None
    assert tab.copy_button.isEnabled() is False

    tab.filter_edit.clear()
    assert tab.table.rowCount() == 3
    tab.deleteLater()


def test_the_listing_survives_being_refreshed_twice(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, tab, fake = _listing(monkeypatch)
    tab.refresh()
    _drain(app, tab)
    tab.table.selectRow(2)
    assert tab.selected() == LISTING[2]

    tab.refresh()
    _drain(app, tab)
    assert fake.calls == [OPENMODELDB_URL, OPENMODELDB_URL]
    assert tab.table.rowCount() == 3
    assert tab.selected() == LISTING[2]
    assert _cell(tab, 2, 0) == "4x Bird GUI"
    tab.deleteLater()


def test_a_second_refresh_is_refused_while_one_is_running(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = threading.Event()
    app, tab, fake = _listing(monkeypatch, gate=gate)
    try:
        tab.refresh()
        assert tab.refresh_button.isEnabled() is False
        assert "Fetching" in tab.status_label.text()

        tab.refresh()
        tab.refresh()
        assert fake.calls == [OPENMODELDB_URL]
    finally:
        gate.set()
        _drain(app, tab)
    assert tab.refresh_button.isEnabled() is True
    assert tab.table.rowCount() == 3
    tab.deleteLater()


def test_a_parse_failure_says_so_and_offers_the_site(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, tab, _ = _listing(monkeypatch)
    tab.refresh()
    _drain(app, tab)
    assert tab.table.rowCount() == 3

    drift = DownloadError(
        f"{OPENMODELDB_URL} yielded no model cards — the page layout has changed"
    )
    app2, tab2, _ = _listing(monkeypatch, error=drift)
    tab2.refresh()
    _drain(app2, tab2)

    status = tab2.status_label.text()
    assert DRIFT_MESSAGE in status
    assert "yielded no model cards" in status
    assert OPENMODELDB_URL in status
    assert tab2.site_link.isVisibleTo(tab2) is True
    assert tab2.site_link.text().startswith("<a href=")
    assert tab2.refresh_button.isEnabled() is True
    assert "curated models" in status
    tab.deleteLater()
    tab2.deleteLater()


def test_a_listing_that_fails_keeps_the_models_already_on_screen(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, tab, _ = _listing(monkeypatch)
    tab.refresh()
    _drain(app, tab)

    broken = _FakeListing(error=DownloadError("Could not reach openmodeldb.info"))
    broken.install(monkeypatch)
    tab.refresh()
    _drain(app, tab)

    assert tab.table.rowCount() == 3
    assert "Could not reach" in tab.status_label.text()
    tab.deleteLater()


# --- what a row can do --------------------------------------------------------


def test_copy_link_puts_the_page_url_on_the_clipboard(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, tab, _ = _listing(monkeypatch)
    tab.refresh()
    _drain(app, tab)

    tab.table.selectRow(0)
    assert tab.copy_button.isEnabled() is True
    tab.copy_button.click()
    assert QGuiApplication.clipboard().text() == LISTING[0].page_url
    assert "Copied" in tab.status_label.text()

    # Taking a selection back leaves the current cell exactly where it was, so
    # a tab that asked `currentRow()` would keep offering a model the user has
    # visibly deselected. This is the state an empty-space click produces.
    tab.table.clearSelection()
    assert tab.table.currentRow() == 0
    assert tab.table.selectedItems() == []
    assert tab.selected() is None
    assert tab.copy_button.isEnabled() is False
    assert tab.open_button.isEnabled() is False
    tab.deleteLater()


def test_open_in_browser_uses_the_page_url(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, tab, _ = _listing(monkeypatch)
    browser = _NoBrowser()
    monkeypatch.setattr(community_tab_module, "QDesktopServices", browser)
    tab.refresh()
    _drain(app, tab)

    tab.table.selectRow(0)
    tab.open_button.click()
    assert browser.opened == [LISTING[0].page_url]
    assert "open in your browser" in tab.status_label.text()

    refusing = _NoBrowser(result=False)
    monkeypatch.setattr(community_tab_module, "QDesktopServices", refusing)
    tab.open_button.click()
    assert refusing.opened == [LISTING[0].page_url]
    assert "Could not open a browser" in tab.status_label.text()
    assert QGuiApplication.clipboard().text() == LISTING[0].page_url
    tab.deleteLater()


def test_only_a_direct_host_offers_a_download_button(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, tab, _ = _listing(monkeypatch)
    tab.refresh()
    _drain(app, tab)

    tab.table.selectRow(0)
    assert LISTING[0].direct_url is None
    assert tab.download_button.isVisibleTo(tab) is False

    tab.table.selectRow(1)
    assert tab.download_button.isVisibleTo(tab) is True
    assert tab.download_button.isEnabled() is True
    assert LISTING[1].direct_url in tab.download_button.toolTip()

    tab.table.selectRow(2)
    assert tab.download_button.isVisibleTo(tab) is True

    tab.table.setCurrentCell(-1, -1)
    assert tab.download_button.isVisibleTo(tab) is False
    tab.deleteLater()


def test_the_direct_host_set_agrees_with_the_core_parser(
    qapp: QApplication,
) -> None:
    """The tab keeps its own copy of core's private host set; keep them equal."""
    direct = community_tab_module._DIRECT_HOSTS
    assert host_for(LISTING[1].direct_url or "") in direct
    assert host_for(LISTING[2].direct_url or "") in direct
    for landing in (
        DRIVE_URL,
        "https://www.mediafire.com/file/abc/model.pth",
        "https://example.com/model.pth",
    ):
        assert host_for(landing) not in direct


def test_a_direct_download_writes_the_file(
    qapp: QApplication, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, tab, _ = _listing(monkeypatch)

    def download(
        url: str,
        dest: Path,
        *,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
        emit: object = None,
    ) -> DownloadResult:
        assert url == LISTING[1].direct_url
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(PAYLOAD)
        return DownloadResult(path=dest, bytes_written=len(PAYLOAD), resumed_from=0)

    monkeypatch.setattr(community_tab_module, "download_file", download)
    tab.refresh()
    _drain(app, tab)

    tab.table.selectRow(1)
    tab.download_button.click()
    events = _drain(app, tab)

    written = models_dir / "w.onnx"
    assert written.is_file()
    assert written.read_bytes() == PAYLOAD
    assert [event.kind for event in events] == ["done"]  # type: ignore[attr-defined]
    assert str(written) in tab.status_label.text()
    assert tab.download_button.isEnabled() is True
    tab.deleteLater()
