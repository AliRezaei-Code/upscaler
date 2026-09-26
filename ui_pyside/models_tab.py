"""The curated model catalogue, as a browsable table with a download button.

The catalogue is the supported way to get a model: fourteen entries, each with
its size and digest taken from its publisher's own release page, and every
architecture in it one spandrel can load. The Community tab is a browser for
everything else, not an alternative catalogue.

Nothing in this module does work on the GUI thread that is not a `stat`. A
download is submitted to `UpscalerApp` — which is also what carries its progress
back as `PipelineEvent`s — and the window routes those events here.
"""

from __future__ import annotations

import html
import threading
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from core import paths
from core.app import Emit, TaskFn, UpscalerApp
from core.download import download_file
from core.errors import UpscalerError
from core.events import PipelineEvent
from core.models import ModelEntry, load_catalogue, verify_model

COLUMNS = ("Name", "Architecture", "Scale", "Size", "Licence", "Installed")

#: Column indices, named so the row builder and the tests cannot drift.
NAME_COLUMN = 0
INSTALLED_COLUMN = 5

#: A download is measured in MiB; the catalogue is measured in MB and GB, which
#: is how the model publishers write them.
_MIB = 1024 * 1024


def _format_mib(byte_count: int) -> str:
    """Bytes as MiB."""
    return f"{byte_count / _MIB:.1f} MiB"


def _format_size(size_bytes: int) -> str:
    """A size for a table cell: megabytes, or gigabytes once it is over a gig."""
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1e9:.2f} GB"
    return f"{size_bytes / 1e6:.1f} MB"


def _format_duration(seconds: float) -> str:
    """Seconds as `h m s`, the form a 67 MB download on a slow line is read in."""
    whole = int(seconds)
    return f"{whole // 3600} h {whole % 3600 // 60:02d} m {whole % 60:02d} s"


def _progress_text(processed: int, total: int, eta_seconds: float) -> str:
    """What the status label reads while bytes are arriving."""
    if total > 0:
        text = f"{_format_mib(processed)} of {_format_mib(total)}"
    else:
        # The host sent no Content-Length and the catalogue has no size for it;
        # a bar that divides by zero is worse than one that does not divide.
        text = f"{_format_mib(processed)} downloaded, total size unknown"
    if eta_seconds > 0:
        return f"{text} — {_format_duration(eta_seconds)} left"
    return text


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


class ModelsTab(QWidget):
    """The curated model catalogue: fourteen entries and a download button.

    Why a separate tab: the curated list is the supported path. Every entry was
    fetched from its publisher's release API and its size and digest verified,
    and every architecture in it is one spandrel can actually load. The Community
    tab is a browser, not a catalogue.

    Wiring, for the window that owns this tab:

    * `EventBridge.event` → `handle_event`, for every `PipelineEvent`. Nothing else
      is needed; events belonging to tasks this tab did not start are ignored.
    * `selected_entry()` and `local_path()` to fill in a `JobConfig`.
    * `on_download_started()` and `on_download_finished()` are the state changes
      a download makes. The tab calls them itself; the window may call them too.

    The tab needs no event pump to exist: it can be constructed, populated and
    shown with nothing draining the app's queue, and it will sit there correct
    until the pump starts and delivers the first event.

    One download runs at a time. There is one progress bar and one status line,
    so a second concurrent download would have two truths to report into one of
    them, and the commonest mistake is pressing Download twice on the same row.

    Attributes:
        search_edit: Filters the table on name, architecture and tags.
        table: One row per model — name, architecture, scale, size, licence and
            whether the file is already on disk.
        details: Notes, licence, source and tags for the selected model.
        download_button: Downloads the selected model, or opens the folder of
            one that is already installed.
        progress_bar: The running download's bytes, indeterminate when nobody
            said how big the file is.
        status_label: The last thing that happened, in the user's words.
    """

    def __init__(self, app: UpscalerApp, parent: QWidget | None = None) -> None:
        """Build the tab and read the catalogue.

        Args:
            app: The driver every download is submitted to. Nothing else in
                this tab opens a thread or a socket.
            parent: The Qt parent, normally the window's `QTabWidget`.
        """
        super().__init__(parent)
        self._app = app
        self._entries: list[ModelEntry] = []
        self._visible: list[ModelEntry] = []
        self._downloading: dict[str, ModelEntry] = {}
        self._rejected: set[str] = set()
        self._catalogue_error = ""

        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("Filter by name, architecture or tag")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._on_filter_changed)

        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            NAME_COLUMN, QHeaderView.ResizeMode.Stretch
        )
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        self.details = QTextBrowser(self)
        self.details.setOpenExternalLinks(True)
        self.details.setMaximumHeight(150)

        # Hidden until a download starts: a bar reading 0% on a catalogue
        # nobody has downloaded anything from looks like something is stuck.
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        self.download_button = QPushButton("Download", self)
        # Fixed, so hiding the progress bar does not stretch the button across
        # the window it is meant to be a control in.
        self.download_button.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self.download_button.clicked.connect(self._on_download_clicked)

        button_row = QHBoxLayout()
        button_row.addWidget(self.download_button)
        button_row.addWidget(self.progress_bar, 1)
        # Absorbs what the bar is not using, so the button stays on the left
        # whether or not a download is running.
        button_row.addStretch(1)

        self.status_label = QLabel(self)
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.search_edit)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.details)
        layout.addLayout(button_row)
        layout.addWidget(self.status_label)

        self._load_catalogue()

    # --- what the window asks -------------------------------------------------

    def selected_entry(self) -> ModelEntry | None:
        """The catalogue entry the highlighted row describes, or `None`."""
        row = _selected_row(self.table, len(self._visible))
        return self._visible[row] if row >= 0 else None

    def local_path(self, entry: ModelEntry) -> Path:
        """Where `entry` is, or would be, on disk.

        Every tab that needs a model's path goes through here, so the download,
        the installed check and a `JobConfig` built by the window cannot end up
        disagreeing about where the file is.
        """
        return paths.models_dir() / entry.filename

    def on_download_started(self, entry: ModelEntry) -> None:
        """Put the tab into its downloading state for `entry`."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self._set_status(f"Downloading {entry.name}…")
        self._update_button()

    def on_download_finished(self, entry: ModelEntry, path: Path) -> None:
        """Mark `entry` installed, now that `path` is a verified model."""
        self._rejected.discard(entry.id)
        row = self._row_of(entry)
        if row >= 0:
            cell = self.table.item(row, INSTALLED_COLUMN)
            if cell is not None:
                cell.setText("Installed")
            if self.table.currentRow() != row:
                self.table.selectRow(row)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_bar.setVisible(True)
        self._set_status(f"{entry.name} is installed at {path}")
        self._update_button()

    def handle_event(self, event: PipelineEvent) -> None:
        """Show one event from `UpscalerApp`, on the GUI thread.

        The window connects this to `EventBridge.event`. Events for a task this
        tab did not start — an upscale job, a community fetch, another model's
        download — are ignored, which is the whole reason `task_id` exists.

        Args:
            event: One `PipelineEvent`, already delivered on the GUI thread by
                the bridge.
        """
        entry = self._downloading.get(event.task_id)
        if entry is None:
            return
        if event.kind in ("stage", "log"):
            self._set_status(event.message)
        elif event.kind == "device_progress":
            self._on_download_progress(event)
        elif event.kind == "error":
            self._mark_rejected(entry)
            # A half-filled bar left on screen after a failure reads as a
            # download that is still going; the message is the news.
            self.progress_bar.setVisible(False)
            self._end_download()
            self._set_status(event.message or f"{entry.name} could not be downloaded")
        elif event.kind == "done":
            self._end_download()
            self.on_download_finished(entry, self.local_path(entry))

    # --- the catalogue --------------------------------------------------------

    def _load_catalogue(self) -> None:
        """Read the catalogue, or say why it could not be read.

        A broken catalogue is shown as a broken catalogue. An empty table with a
        Download button that does nothing is the same failure wearing a working
        costume, and this app is specifically about telling the user which of
        the two happened.
        """
        self._catalogue_error = ""
        try:
            self._entries = load_catalogue()
        except (UpscalerError, OSError) as exc:
            self._entries = []
            self._catalogue_error = str(exc)
        self._rebuild()
        if self._catalogue_error:
            self._set_status(
                f"The model catalogue could not be read: {self._catalogue_error}"
            )
        else:
            self._set_status(
                f"{len(self._entries)} models. Select one to download it, or "
                "point a job at the file with Browse local…"
            )

    def _filter_entries(self) -> list[ModelEntry]:
        """The catalogue narrowed by the search box, on name, architecture, tags."""
        needle = self.search_edit.text().strip().lower()
        if not needle:
            return list(self._entries)
        return [
            entry
            for entry in self._entries
            if needle in entry.name.lower()
            or needle in entry.architecture.lower()
            or any(needle in tag.lower() for tag in entry.tags)
        ]

    def _on_filter_changed(self, _text: str) -> None:
        """Rebuild the table for what is left of the catalogue."""
        self._rebuild()

    def _rebuild(self) -> None:
        """Fill the table from the catalogue, keeping the selection if it survives."""
        previous = self.selected_entry()
        self._visible = self._filter_entries()
        self.table.setRowCount(len(self._visible))
        self.table.clearContents()
        for row, entry in enumerate(self._visible):
            cells = (
                entry.name,
                entry.architecture,
                f"{entry.scale}x",
                _format_size(entry.size_bytes),
                entry.license,
                "Installed" if self._is_installed(entry) else "Not installed",
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, entry.id)
                if column in (2, INSTALLED_COLUMN):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)
        row = self._row_of(previous) if previous is not None else 0
        if self._visible:
            self.table.selectRow(max(row, 0))
        self._on_selection_changed()

    def _row_of(self, entry: ModelEntry | None) -> int:
        """The visible row for `entry`, or -1 when the filter hides it."""
        if entry is None:
            return -1
        for row, visible in enumerate(self._visible):
            if visible.id == entry.id:
                return row
        return -1

    def _is_installed(self, entry: ModelEntry) -> bool:
        """Whether the model is already on disk, judged by size alone.

        A `stat` is microseconds; hashing a 2.4 GB checkpoint to answer the same
        question is a second of frozen window, so the digest is checked where it
        matters — after a download, in `verify_model`, off the GUI thread.

        A model whose last download failed verification is never installed, even
        when the file it left behind is exactly the right number of bytes: that
        is what a wrong file of the right size looks like, and offering it as a
        ready model is how a job fails two minutes later with a worse message.
        """
        if entry.id in self._rejected:
            return False
        try:
            size = self.local_path(entry).stat().st_size
        except OSError:
            return False
        return entry.size_bytes == 0 or size == entry.size_bytes

    def _mark_rejected(self, entry: ModelEntry) -> None:
        """Record that `entry` is on disk but is not a model the app can load."""
        self._rejected.add(entry.id)
        row = self._row_of(entry)
        if row >= 0:
            cell = self.table.item(row, INSTALLED_COLUMN)
            if cell is not None:
                cell.setText("Not installed")

    # --- selection ------------------------------------------------------------

    def _on_selection_changed(self) -> None:
        """Show the newly selected model, and what can be done with it."""
        entry = self.selected_entry()
        if entry is None:
            self.details.setHtml(
                "<p>No model selected. Clear the filter to see the whole "
                "catalogue again.</p>"
            )
        else:
            name = html.escape(entry.name)
            architecture = html.escape(entry.architecture)
            source = html.escape(entry.source)
            self.details.setHtml(
                f"<p><b>{name}</b> — {architecture}, {entry.scale}x,"
                f" {_format_size(entry.size_bytes)},"
                f" {html.escape(entry.license)}</p>"
                f"<p>{html.escape(entry.notes)}</p>"
                f'<p>Source: <a href="{source}">{source}</a></p>'
                f"<p>Tags: {html.escape(', '.join(entry.tags))}</p>"
            )
        self._update_button()

    def _update_button(self) -> None:
        """The download button is whatever the selected model calls for."""
        entry = self.selected_entry()
        if entry is None:
            self.download_button.setText("Download")
            self.download_button.setEnabled(False)
            self.download_button.setToolTip("Select a model first")
            return
        if self._downloading:
            self.download_button.setText("Downloading…")
            self.download_button.setEnabled(False)
            self.download_button.setToolTip(
                "One download at a time; this one has to finish or fail first"
            )
            return
        if self._is_installed(entry):
            path = self.local_path(entry)
            self.download_button.setText("Open folder")
            self.download_button.setEnabled(True)
            self.download_button.setToolTip(f"Show {path.parent} in a file manager")
            return
        self.download_button.setText("Download")
        self.download_button.setEnabled(not self._catalogue_error)
        self.download_button.setToolTip(
            f"Fetch {entry.name} ({_format_size(entry.size_bytes)})"
        )

    # --- downloading ----------------------------------------------------------

    def _on_download_clicked(self, _checked: bool = False) -> None:
        """Download the selected model, or show the one already on disk."""
        entry = self.selected_entry()
        if entry is None:
            return
        if self._is_installed(entry):
            self._open_folder(entry)
            return
        self._start_download(entry)

    def _start_download(self, entry: ModelEntry) -> None:
        """Submit the download and index it under every id it will report under.

        `download_file` stamps its own events `model:<destination stem>`, and
        for eleven of the fourteen catalogue entries that stem is not the entry
        id — `RealESRGAN_x4plus.pth` belongs to `realesrgan-x4plus`. Both keys
        are registered, or the progress bar would sit at zero for every model
        the catalogue does not happen to name after itself.
        """
        self._downloading[f"model:{entry.id}"] = entry
        self._downloading[f"model:{Path(entry.filename).stem}"] = entry
        # A retry is a fresh attempt: whatever the last one left behind is
        # about to be overwritten, so it no longer disqualifies the model.
        self._rejected.discard(entry.id)
        self.on_download_started(entry)
        self._app.submit(f"model:{entry.id}", self._download_task(entry))

    def _download_task(self, entry: ModelEntry) -> TaskFn:
        """The function the app runs on its own thread: fetch, then verify.

        Verification is not optional and is not the UI's business: a host that
        answers a release URL with the nine bytes of `Not Found` produces a file
        that spandrel would fail on two minutes later, with a far worse message.
        Raising here turns that into one `error` event carrying the core layer's
        own words, and the file is never left looking installed.

        The `stop` event the app hands every task is not consulted: `download_file`
        owns the HTTP transfer loop and exposes no cancellation hook, so a
        download here runs to its end or fails. There is no cancel button rather
        than one that would not cancel.
        """

        def task(emit: Emit, stop: threading.Event) -> Path:
            dest = self.local_path(entry)
            emit(
                PipelineEvent(
                    kind="stage",
                    stage="download",
                    message=f"Downloading {entry.name} from {entry.url}",
                )
            )
            download_file(
                entry.url,
                dest,
                expected_size=entry.size_bytes,
                expected_sha256=entry.sha256,
                emit=emit,
            )
            verify_model(dest, entry)
            emit(PipelineEvent(kind="done", message=f"{entry.name} is installed"))
            return dest

        return task

    def _on_download_progress(self, event: PipelineEvent) -> None:
        """Move the bar and the status line for one progress report."""
        if event.total > 0:
            self.progress_bar.setRange(0, event.total)
            self.progress_bar.setValue(min(event.processed, event.total))
        else:
            self.progress_bar.setRange(0, 0)
        self._set_status(
            _progress_text(event.processed, event.total, event.eta_seconds)
        )

    def _end_download(self) -> None:
        """Forget the running download and give the button back."""
        self._downloading.clear()
        self._update_button()

    def _open_folder(self, entry: ModelEntry) -> None:
        """Show the model in the file manager, or say that the desktop would not."""
        folder = self.local_path(entry).parent
        if QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self._set_status(f"{folder} is open in your file manager")
        else:
            self._set_status(f"Could not open {folder} in a file manager")

    def _set_status(self, text: str) -> None:
        """Put one line of plain text in the status label."""
        self.status_label.setText(text)
