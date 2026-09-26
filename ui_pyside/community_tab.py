"""The OpenModelDB listing, as a browser rather than a second catalogue.

This tab exists so that the fourteen curated models are not the only thing the
app can see, and it deliberately does not compete with them. Every download
here goes to somebody else's site over a link that site chose, so the curated
Models tab stays the supported path and this one stays a browser.
"""

from __future__ import annotations

import html
import re
import threading
from pathlib import Path
from urllib.parse import urlparse

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from core import paths
from core.app import Emit, TaskFn, UpscalerApp
from core.community import OPENMODELDB_URL, CommunityModel, fetch_openmodeldb, host_for
from core.download import download_file
from core.events import PipelineEvent
from core.models import verify_model

#: The task id the listing is fetched under, as documented in `PipelineEvent`.
COMMUNITY_TASK_ID = "community"

#: What a listing that no longer parses must say. The weekly CI job opens an
#: issue when `fetch_openmodeldb` starts failing, and this is the sentence the
#: user reads while that is happening; the underlying message is shown under it
#: so the reason — drift, or a host that could not be reached — is never hidden
#: behind the headline.
DRIFT_MESSAGE = "Could not parse OpenModelDB — the page layout changed"

#: Hosts that serve file bytes to a plain GET. `core.community` fills in
#: `direct_url` for exactly these and keeps the set private, so this is the UI's
#: half of that decision; `tests/test_community_tab.py` asserts the two agree,
#: because a copy that drifted would offer to download an interstitial page.
_DIRECT_HOSTS = frozenset({"github", "huggingface"})

#: What a downloaded model file is allowed to be called.
_MODEL_SUFFIXES = (".pth", ".onnx", ".safetensors", ".ckpt")

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")

COLUMNS = ("Name", "Architecture", "Scale", "Author")


def _slug(model: CommunityModel) -> str:
    """A filesystem-safe name for `model`, from its own page URL."""
    last = model.page_url.rstrip("/").rsplit("/", 1)[-1]
    cleaned = _UNSAFE.sub("-", last).strip("-.")
    return cleaned or "model"


def _filename_for(model: CommunityModel) -> str:
    """The name to save a direct download under.

    The host's own basename where it has a model extension, because that is the
    name the weights are published under; otherwise a slug off the page URL, so
    the file is at least recognisable in the models directory.
    """
    direct = model.direct_url or ""
    basename = Path(urlparse(direct).path).name
    if basename.lower().endswith(_MODEL_SUFFIXES):
        return basename
    return f"{_slug(model)}.pth"


def _selected_row(table: QTableWidget, count: int) -> int:
    """The single highlighted row of `table`, or -1 when nothing is highlighted.

    The selection, not `currentRow()`: clicking the empty space below the last
    row clears the highlight and leaves the current cell where it was, so a tab
    that asked `currentRow()` would keep offering a model the user has visibly
    deselected.
    """
    rows = {index.row() for index in table.selectionModel().selectedRows()}
    if len(rows) != 1:
        return -1
    row = rows.pop()
    return row if 0 <= row < count else -1


class CommunityTab(QWidget):
    """The OpenModelDB listing, with a link out to every model's own page.

    Why there is no download button: a listing card carries no file link. The
    weights live behind the model's own page, behind a host-specific
    interstitial, and scraping one detail page per model would turn a browsing
    tab into several hundred HTTP requests against a site that owes this app
    nothing. So the tab opens the page and copies the link, and a Download
    button appears only for a row whose `direct_url` is a host that serves file
    bytes to a plain GET — which, as of 2026-09-26, no listing card does.

    Wiring, for the window that owns this tab: `EventBridge.event` →
    `handle_event`. The tab constructs, fills and shows with no pump attached.

    Attributes:
        refresh_button: Fetches the listing again; disabled while one fetch is
            running, because a second one would be queued behind nothing.
        filter_edit: Narrows the list on name, architecture and author.
        table: One row per model — name, architecture, scale and author.
        details: The selected model's author, tags, host and page link.
        open_button: Opens the model's page in a browser.
        copy_button: Copies the page URL to the clipboard.
        download_button: Only for a row with a direct file URL; see above.
        status_label: The last thing that happened, in the user's words.
        site_link: Appears when the listing could not be read, because the one
            thing a user can do about drift is go and look at the site.
    """

    def __init__(self, app: UpscalerApp, parent: QWidget | None = None) -> None:
        """Build the tab. Nothing is fetched until `refresh()` is called.

        Args:
            app: The driver the listing fetch is submitted to.
            parent: The Qt parent, normally the window's `QTabWidget`.
        """
        super().__init__(parent)
        self._app = app
        self._models: list[CommunityModel] = []
        self._visible: list[CommunityModel] = []
        self._fetching = False
        self._fetched: list[CommunityModel] = []
        self._downloads: dict[str, CommunityModel] = {}

        self.refresh_button = QPushButton("Refresh", self)
        self.refresh_button.clicked.connect(self._on_refresh_clicked)

        self.filter_edit = QLineEdit(self)
        self.filter_edit.setPlaceholderText("Filter by name, architecture or author")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._on_filter_changed)

        top_row = QHBoxLayout()
        top_row.addWidget(self.refresh_button)
        top_row.addWidget(self.filter_edit, 1)

        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        self.details = QTextBrowser(self)
        self.details.setOpenExternalLinks(True)
        self.details.setMaximumHeight(140)

        self.open_button = QPushButton("Open in browser", self)
        self.open_button.clicked.connect(self._on_open_clicked)
        self.copy_button = QPushButton("Copy link", self)
        self.copy_button.clicked.connect(self._on_copy_clicked)
        self.download_button = QPushButton("Download", self)
        self.download_button.clicked.connect(self._on_download_clicked)
        self.download_button.setVisible(False)

        button_row = QHBoxLayout()
        button_row.addWidget(self.open_button)
        button_row.addWidget(self.copy_button)
        button_row.addWidget(self.download_button)
        button_row.addStretch(1)

        self.site_link = QLabel(self)
        self.site_link.setOpenExternalLinks(True)
        self.site_link.setTextFormat(Qt.TextFormat.RichText)
        self.site_link.setText(
            f'<a href="{OPENMODELDB_URL}">Open {OPENMODELDB_URL} to see why</a>'
        )
        self.site_link.setVisible(False)

        self.status_label = QLabel(self)
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(top_row)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.details)
        layout.addLayout(button_row)
        layout.addWidget(self.site_link)
        layout.addWidget(self.status_label)

        self._on_selection_changed()
        self._set_status(
            "Press Refresh to read the OpenModelDB listing. Models are opened "
            "in a browser; nothing is downloaded from here unless the site "
            "offers the file itself."
        )

    # --- what the window asks -------------------------------------------------

    def refresh(self) -> None:
        """Fetch the listing again, unless a fetch is already running.

        A second fetch is refused rather than queued: the answer would be the
        same page, and two requests for it is one more thing for a site that
        publishes this for free to answer.
        """
        if self._fetching:
            return
        self._fetching = True
        self.refresh_button.setEnabled(False)
        self._set_status(f"Fetching the OpenModelDB listing from {OPENMODELDB_URL}…")
        self._app.submit(COMMUNITY_TASK_ID, self._fetch_task())

    def selected(self) -> CommunityModel | None:
        """The model the highlighted row describes, or `None`."""
        row = _selected_row(self.table, len(self._visible))
        return self._visible[row] if row >= 0 else None

    def handle_event(self, event: PipelineEvent) -> None:
        """Show one event from `UpscalerApp`, on the GUI thread.

        The window connects this to `EventBridge.event`. Anything belonging to a
        task this tab did not start is ignored.

        Args:
            event: One `PipelineEvent`, already delivered on the GUI thread by
                the bridge.
        """
        if event.task_id == COMMUNITY_TASK_ID:
            self._on_listing_event(event)
            return
        model = self._downloads.get(event.task_id)
        if model is None:
            return
        if event.kind in ("stage", "log"):
            self._set_status(event.message)
        elif event.kind == "device_progress":
            self._on_download_progress(model, event)
        elif event.kind == "error":
            self._end_download()
            self._set_status(event.message or f"{model.name} could not be downloaded")
        elif event.kind == "done":
            self._end_download()
            self._set_status(f"{model.name} is installed at {self._local_path(model)}")

    # --- the listing ----------------------------------------------------------

    def _fetch_task(self) -> TaskFn:
        """The function the app runs on its own thread: fetch and parse."""

        def task(emit: Emit, stop: threading.Event) -> list[CommunityModel]:
            models = fetch_openmodeldb()
            # Stashed here, before `done` goes on the queue, because the app
            # records what a task returned only after it has returned: a tab
            # reading `app.results` when it saw `done` would be racing the
            # producer thread it is meant to follow.
            self._fetched = models
            emit(
                PipelineEvent(
                    kind="done", message=f"Read {len(models)} models from OpenModelDB"
                )
            )
            return models

        return task

    def _on_listing_event(self, event: PipelineEvent) -> None:
        """React to the fetch: a list, an error, or a line of progress."""
        if event.kind in ("stage", "log"):
            self._set_status(event.message)
        elif event.kind == "error":
            self._fetching = False
            self.refresh_button.setEnabled(True)
            # The list already on screen is left alone: a site that has moved
            # does not get to empty the tab, and the curated Models tab is
            # unaffected either way.
            self.site_link.setVisible(True)
            self._set_status(
                f"{DRIFT_MESSAGE}\n{event.message or 'OpenModelDB returned nothing.'}\n"
                "The curated models on the Models tab are unaffected."
            )
        elif event.kind == "done":
            self._fetching = False
            self.refresh_button.setEnabled(True)
            self.site_link.setVisible(False)
            self._show_models(self._fetched)

    def _show_models(self, models: list[CommunityModel]) -> None:
        """Replace the list, keeping the selection if the same model survives."""
        previous = self.selected()
        self._models = list(models)
        self._rebuild(previous)
        if not models:
            self._set_status("OpenModelDB returned no models.")
        else:
            self._set_status(
                f"{len(models)} models from OpenModelDB. Open one in a browser to "
                "find the file; the Models tab has the verified ones."
            )

    def _filter_models(self) -> list[CommunityModel]:
        """The listing narrowed by the filter box."""
        needle = self.filter_edit.text().strip().lower()
        if not needle:
            return list(self._models)
        return [
            model
            for model in self._models
            if needle in model.name.lower()
            or needle in model.architecture.lower()
            or needle in model.author.lower()
        ]

    def _on_filter_changed(self, _text: str) -> None:
        """Rebuild the table for what is left of the listing."""
        self._rebuild(self.selected())

    def _rebuild(self, previous: CommunityModel | None) -> None:
        """Fill the table from the listing, keeping the selection if it survives."""
        self._visible = self._filter_models()
        self.table.setRowCount(len(self._visible))
        self.table.clearContents()
        for row, model in enumerate(self._visible):
            cells = (model.name, model.architecture, f"{model.scale}x", model.author)
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(model.page_url)
                if column == 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)
        if self._visible:
            row = self._row_of(previous)
            self.table.selectRow(max(row, 0))
        self._on_selection_changed()

    def _row_of(self, model: CommunityModel | None) -> int:
        """The visible row for `model`, or -1 when the filter hides it."""
        if model is None:
            return -1
        for row, visible in enumerate(self._visible):
            if visible.page_url == model.page_url:
                return row
        return -1

    def _on_refresh_clicked(self, _checked: bool = False) -> None:
        """Fetch the listing again."""
        self.refresh()

    # --- selection ------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        """Show the newly selected model, and what can be done with it."""
        model = self.selected()
        if model is None:
            self.details.setHtml("<p>No model selected.</p>")
            self.open_button.setEnabled(False)
            self.copy_button.setEnabled(False)
            self.download_button.setVisible(False)
            return
        self.details.setHtml(
            f"<p><b>{html.escape(model.name)}</b> — {html.escape(model.architecture)},"
            f" {model.scale}x, by {html.escape(model.author)}</p>"
            f"<p>{html.escape(', '.join(model.tags))}</p>"
            f'<p><a href="{html.escape(model.page_url)}">'
            f"{html.escape(model.page_url)}</a></p>"
        )
        self.open_button.setEnabled(True)
        self.copy_button.setEnabled(True)
        self.download_button.setVisible(self._has_direct_file(model))
        self.download_button.setToolTip(
            f"Fetch {model.direct_url}" if self._has_direct_file(model) else ""
        )

    def _has_direct_file(self, model: CommunityModel) -> bool:
        """Whether this row's model can be fetched without a browser."""
        if model.direct_url is None:
            return False
        return host_for(model.direct_url) in _DIRECT_HOSTS

    # --- what a selected row can do ------------------------------------------

    def _on_open_clicked(self, _checked: bool = False) -> None:
        """Open the model's page in a browser, or say the desktop would not."""
        model = self.selected()
        if model is None:
            return
        if QDesktopServices.openUrl(QUrl(model.page_url)):
            self._set_status(f"{model.name} is open in your browser")
            return
        # A desktop with no browser is not rare enough to be an error the user
        # can do nothing about, so the link goes to the clipboard and the
        # reason the browser did not open is what is left on screen.
        QGuiApplication.clipboard().setText(model.page_url)
        self._set_status(
            f"Could not open a browser; {model.page_url} is on the clipboard"
        )

    def _on_copy_clicked(self, _checked: bool = False) -> None:
        """Copy the model's page URL to the clipboard."""
        model = self.selected()
        if model is None:
            return
        QGuiApplication.clipboard().setText(model.page_url)
        self._set_status(f"Copied {model.page_url}")

    def _local_path(self, model: CommunityModel) -> Path:
        """Where a direct download of `model` is written."""
        return paths.models_dir() / _filename_for(model)

    def _on_download_clicked(self, _checked: bool = False) -> None:
        """Fetch the selected model's file, when its host will serve one."""
        model = self.selected()
        if model is None or not self._has_direct_file(model):
            return
        if self._downloads:
            return
        task_id = f"model:{_slug(model)}"
        self._downloads[task_id] = model
        self.download_button.setEnabled(False)
        self._set_status(f"Downloading {model.name}…")
        self._app.submit(task_id, self._download_task(model))

    def _download_task(self, model: CommunityModel) -> TaskFn:
        """The function the app runs on its own thread: fetch, then verify.

        A Google Drive or MediaFire link never reaches this tab — those hosts
        answer with an interstitial page, and saving one as a `.pth` is the
        nine-byte failure `verify_model` exists to catch. The check stays
        anyway, because a direct host can be wrong about what it is serving.

        The `stop` event the app hands every task is not consulted: `download_file`
        owns the HTTP transfer loop and exposes no cancellation hook, so the
        download runs to its end or fails. There is no cancel button rather than
        one that would not cancel.
        """
        direct = model.direct_url or ""

        def task(emit: Emit, stop: threading.Event) -> Path:
            dest = self._local_path(model)
            download_file(direct, dest, emit=emit)
            verify_model(dest, None)
            emit(PipelineEvent(kind="done", message=f"{model.name} is installed"))
            return dest

        return task

    def _on_download_progress(
        self, model: CommunityModel, event: PipelineEvent
    ) -> None:
        """Say how far a direct download has got.

        No progress bar: this tab is a browser, and the Models tab is where a
        download is something to watch.
        """
        if event.total > 0:
            self._set_status(
                f"Downloading {model.name} — {event.processed:,} of "
                f"{event.total:,} bytes"
            )
        else:
            self._set_status(
                f"Downloading {model.name} — {event.processed:,} bytes, "
                "total size unknown"
            )

    def _end_download(self) -> None:
        """Forget the running download and give the button back."""
        self._downloads.clear()
        self.download_button.setEnabled(True)

    def _set_status(self, text: str) -> None:
        """Put one block of plain text in the status label."""
        self.status_label.setText(text)
