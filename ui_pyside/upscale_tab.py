"""The Upscale tab: the form, the device list, the run bar and the log.

This tab owns no logic. It builds a `JobConfig` from what the user typed, hands
it to `UpscalerApp.start()`, and renders whatever `PipelineEvent`s come back.
The two rules that keep it that way:

* **nothing slow on the GUI thread.** Probing three wedged GPUs costs up to ten
  seconds, the source probe costs a subprocess, and measuring a work directory
  costs a directory walk — so all three go through `app.submit()`;
* **every widget this tab shows is reachable from a public method**, so the
  headless smoke can drive the whole tab without a mouse.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.app import UpscalerApp
from core.config import Device, JobConfig
from core.devices import probe_all
from core.errors import ConfigError
from core.events import PipelineEvent
from core.models import ModelEntry, load_catalogue
from core.paths import default_work_dir, models_dir

#: The catalogue entry that means "the user picked a file themselves".
BROWSE_LOCAL = "Browse local…"
MODEL_FILTER = "Model files (*.pth *.ckpt *.safetensors *.pt *.onnx);;All files (*)"
VIDEO_FILTER = "Videos (*.mp4 *.mkv *.mov *.avi *.webm);;All files (*)"

#: The log's last N lines, dropped by Qt itself rather than by a buffer we
#: would have to trim and keep in step with the text.
LOG_LINES = 2000

_DEVICE_ROLE = Qt.ItemDataRole.UserRole


def default_output_path(source: Path) -> Path:
    """`<stem>.upscaled<ext>` beside the source.

    Beside it, not in the work directory: someone who upscales `holiday.mkv` on
    their desktop expects `holiday.upscaled.mkv` on their desktop.
    """
    return source.with_name(f"{source.stem}.upscaled{source.suffix or '.mp4'}")


class UpscaleTab(QWidget):
    """The job form and its run bar."""

    def __init__(self, app: UpscalerApp, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._app = app
        self._devices: list[Device] = []
        self._active: list[Device] = []
        self._user_edited_output = False
        self._local_model: Path | None = None
        self._work_dir = default_work_dir(Path.cwd() / "untitled.mp4")
        self._build()
        self._reload_models()
        self.refresh_devices()

    # --- construction ---------------------------------------------------------

    def _build(self) -> None:
        layout = QVBoxLayout(self)

        source_box = QGroupBox("Source")
        source_form = QFormLayout(source_box)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Choose a video…")
        self.source_edit.textChanged.connect(self._on_source_changed)
        browse_source = QPushButton("Browse…")
        browse_source.clicked.connect(self.browse_source)
        source_row = QHBoxLayout()
        source_row.addWidget(self.source_edit)
        source_row.addWidget(browse_source)
        source_form.addRow("Video", source_row)

        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        browse_model = QPushButton(BROWSE_LOCAL)
        browse_model.clicked.connect(self.browse_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo)
        model_row.addWidget(browse_model)
        source_form.addRow("Model", model_row)

        self.output_edit = QLineEdit()
        self.output_edit.textEdited.connect(self._on_output_edited)
        browse_output = QPushButton("Choose…")
        browse_output.clicked.connect(self.browse_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit)
        output_row.addWidget(browse_output)
        source_form.addRow("Output", output_row)
        layout.addWidget(source_box)

        devices_box = QGroupBox("Devices")
        devices_layout = QVBoxLayout(devices_box)
        self.device_list = QListWidget()
        self.device_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.device_list.itemChanged.connect(self._refresh_device_table)
        devices_layout.addWidget(self.device_list)
        rescan = QPushButton("Rescan devices")
        rescan.clicked.connect(self.refresh_devices)
        devices_layout.addWidget(rescan)
        layout.addWidget(devices_box)

        settings_box = QGroupBox("Settings")
        settings_form = QFormLayout(settings_box)
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["auto", "fp16", "fp32"])
        self.precision_combo.setToolTip(
            "auto follows the hardware: a Pascal GPU, or any GPU whose driver "
            "would not say, gets FP32."
        )
        settings_form.addRow("Precision", self.precision_combo)
        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(0, 4096)
        self.tile_spin.setValue(0)
        self.tile_spin.setToolTip("0 decides from the frame size; 64 is the minimum.")
        settings_form.addRow("Tile size", self.tile_spin)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.0, 1000.0)
        self.fps_spin.setDecimals(3)
        self.fps_spin.setValue(0.0)
        self.fps_spin.setSpecialValueText("inherit from the source")
        settings_form.addRow("Frame rate", self.fps_spin)
        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)
        self.crf_spin.setValue(18)
        self.crf_spin.setToolTip("0 is lossless, 51 is the worst; 18 is the default.")
        settings_form.addRow("CRF", self.crf_spin)
        self.delete_frames_check = QCheckBox("Delete frames after encoding")
        self.delete_frames_check.setToolTip(
            f"frames_out is {self._work_dir / 'frames_out'}; it is 820 GB on a "
            "165,303-frame run."
        )
        settings_form.addRow("", self.delete_frames_check)
        layout.addWidget(settings_box)

        run_box = QGroupBox("Run")
        run_layout = QVBoxLayout(run_box)
        self.progress_bar = QProgressBar()
        # The range stays 0-100 and the value is a percentage. QProgressBar
        # silently *ignores* a value above its maximum rather than clamping it,
        # so a bar whose maximum is the frame count freezes a few frames short
        # of the end — 40 of 60 frames and the bar sits at 33.
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._total_frames = 0
        run_layout.addWidget(self.progress_bar)
        self.device_table = QTableWidget(0, 3)
        self.device_table.setHorizontalHeaderLabels(["Device", "Frames", "fps"])
        self.device_table.verticalHeader().setVisible(False)
        self.device_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        run_layout.addWidget(self.device_table)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._app.stop)
        self.open_output_button = QPushButton("Open output folder")
        self.open_output_button.clicked.connect(self.open_output_folder)
        self.open_work_button = QPushButton("Open work dir")
        self.open_work_button.clicked.connect(self.open_work_dir)
        self.clean_button = QPushButton("Clean work dir")
        self.clean_button.clicked.connect(self.clean_work_dir)
        for button in (
            self.start_button,
            self.stop_button,
            self.open_output_button,
            self.open_work_button,
            self.clean_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)
        run_layout.addLayout(buttons)
        layout.addWidget(run_box)

        self.status_label = QLabel("Ready.")
        layout.addWidget(self.status_label)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(LOG_LINES)
        self.log_view.setPlaceholderText("The job log appears here.")
        layout.addWidget(self.log_view, 1)

    def _reload_models(self) -> None:
        """Fill the model combo from the catalogue, or explain why it is empty."""
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        try:
            entries = load_catalogue()
        except Exception as exc:
            self.model_combo.addItem(f"Catalogue unavailable: {exc}")
            self.model_combo.setEnabled(False)
            self.status_label.setText(str(exc))
            self.model_combo.blockSignals(False)
            return
        for entry in entries:
            self.model_combo.addItem(
                f"{entry.name} — {entry.scale}x — {entry.architecture}", entry.id
            )
        self.model_combo.addItem(BROWSE_LOCAL)
        self.model_combo.setEnabled(True)
        self.model_combo.blockSignals(False)

    # --- devices --------------------------------------------------------------

    def refresh_devices(self) -> None:
        """Probe the machine on a worker thread and fill the list.

        A wedged driver costs the full `PROBE_TIMEOUT_SECONDS`, so this must not
        be a direct call: a ten-second frozen window on startup is exactly what
        the probe was built to stop becoming the user's experience.
        """

        def _probe(
            emit: Callable[[PipelineEvent], None], _stop: object
        ) -> list[Device]:
            return probe_all(emit)

        self.device_list.clear()
        self.status_label.setText("Probing devices…")
        self._app.submit("devices", _probe)

    def set_devices(self, devices: list[Device]) -> None:
        """Show the probed devices. Called from the window's event handler."""
        self._devices = list(devices)
        self.device_list.clear()
        for device in devices:
            item = QListWidgetItem(device.label)
            item.setData(_DEVICE_ROLE, device)
            if device.usable:
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setFlags(Qt.ItemFlag.NoItemFlags)
                font = item.font()
                font.setStrikeOut(True)
                item.setFont(font)
                item.setText(f"{device.label} — unavailable")
                item.setToolTip(device.unusable_reason or "")
            self.device_list.addItem(item)
        self._refresh_device_table()
        unavailable = [device for device in devices if not device.usable]
        if unavailable:
            self.status_label.setText(
                f"{len(devices)} devices; {len(unavailable)} unusable — hover a "
                "greyed row for the reason."
            )
        else:
            self.status_label.setText(f"{len(devices)} devices.")

    def selected_devices(self) -> list[Device]:
        """The devices the user has ticked."""
        chosen: list[Device] = []
        for row in range(self.device_list.count()):
            item = self.device_list.item(row)
            device = item.data(_DEVICE_ROLE)
            if item.checkState() == Qt.CheckState.Checked and isinstance(
                device, Device
            ):
                chosen.append(device)
        return chosen

    def tick_device(self, index: int, checked: bool) -> None:
        """Tick or untick a row by position. Used by the headless smoke."""
        item = self.device_list.item(index)
        if item is None:
            return
        item.setCheckState(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )
        self._refresh_device_table()

    def _refresh_device_table(self) -> None:
        """Fill the run bar with the devices this run will actually use.

        Only the *selected* devices, in selection order, because the pipeline
        reports progress by a worker's ordinal in that selection. A table of
        every probed device would put the first GPU's row where the CPU's
        progress belongs whenever the CPU is the only device running.
        """
        self._active = self.selected_devices()
        self.device_table.setRowCount(len(self._active))
        for row, device in enumerate(self._active):
            name = QTableWidgetItem(device.name)
            name.setToolTip(device.unusable_reason or device.backend)
            self.device_table.setItem(row, 0, name)
            self.device_table.setItem(row, 1, QTableWidgetItem("0"))
            self.device_table.setItem(row, 2, QTableWidgetItem("—"))

    # --- source, model, output ------------------------------------------------

    def browse_source(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Choose a video", "", VIDEO_FILTER
        )
        if chosen:
            self.set_source(Path(chosen))

    def set_source(self, path: Path) -> None:
        """Set the source and, unless the user has edited the output, the output."""
        self.source_edit.setText(str(path))
        if not self._user_edited_output:
            self.output_edit.setText(str(default_output_path(path)))
        self._work_dir = default_work_dir(path)
        self.delete_frames_check.setToolTip(
            f"frames_out is {self._work_dir / 'frames_out'}; it is 820 GB on a "
            "165,303-frame run."
        )
        self._probe_source(path)

    def _on_source_changed(self, text: str) -> None:
        if not text:
            return
        path = Path(text)
        self._work_dir = default_work_dir(path)
        if not self._user_edited_output:
            self.output_edit.setText(str(default_output_path(path)))

    def _on_output_edited(self, _text: str) -> None:
        self._user_edited_output = True

    def browse_output(self) -> None:
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save the result as", self.output_edit.text()
        )
        if chosen:
            self.output_edit.setText(chosen)

    def browse_model(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Choose a model", "", MODEL_FILTER
        )
        if chosen:
            self.set_local_model(Path(chosen))

    def set_local_model(self, path: Path) -> None:
        """Use a model from outside the catalogue."""
        self._local_model = path
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
        self.model_combo.blockSignals(False)
        self.status_label.setText(f"Model: {path}")

    def _on_model_changed(self, index: int) -> None:
        if index < 0:
            return
        self.model_combo.setToolTip(self.selected_model_description())

    def selected_model_description(self) -> str:
        """The notes, source and licence of the chosen model, for a tooltip."""
        entry = self.current_model()
        if entry is not None:
            return f"{entry.notes}\n\n{entry.source}\nLicence: {entry.license}"
        return str(self._local_model) if self._local_model else "No model chosen."

    def current_model(self) -> ModelEntry | None:
        """The catalogue entry chosen, or `None` for a local file."""
        data = self.model_combo.currentData()
        if not isinstance(data, str):
            return None
        for entry in load_catalogue():
            if entry.id == data:
                return entry
        return None

    def _probe_source(self, path: Path) -> None:
        """Read the source's dimensions and frame rate on a worker thread."""

        def _probe(emit: Callable[[PipelineEvent], None], _stop: object) -> None:
            from core.ffmpeg import probe

            info = probe(path)
            emit(
                PipelineEvent(
                    kind="stage",
                    task_id=f"probe:{path}",
                    stage="probe",
                    message=(
                        f"{info.width}x{info.height}, {info.frame_count} frames at "
                        f"{info.fps:.3f} fps, audio "
                        f"{'yes' if info.has_audio else 'no'}"
                    ),
                )
            )

        self._app.submit(f"probe:{path}", _probe)

    def set_source_fps(self, fps: float) -> None:
        self.fps_spin.setValue(fps)

    @staticmethod
    def _fps_from_message(message: str) -> float | None:
        if " at " not in message or " fps" not in message:
            return None
        try:
            return float(message.split(" at ")[-1].split(" fps")[0])
        except ValueError:
            return None

    # --- the run bar ----------------------------------------------------------

    def model_path(self) -> Path:
        """Where the chosen model is, or should be."""
        entry = self.current_model()
        if entry is not None:
            return models_dir() / entry.filename
        if self._local_model is not None:
            return self._local_model
        raise ConfigError("Choose a model first")

    def build_config(self) -> JobConfig:
        """The `JobConfig` for what the user typed, or a clear failure."""
        devices = tuple(self.selected_devices())
        if not devices:
            raise ConfigError("Select at least one GPU")
        return JobConfig(
            input_path=Path(self.source_edit.text()),
            output_path=Path(self.output_edit.text()),
            model_path=self.model_path(),
            work_dir=self._work_dir,
            devices=devices,
            precision=self.precision_combo.currentText(),  # type: ignore[arg-type]
            tile_size=self.tile_spin.value(),
            fps=self.fps_spin.value() or None,
            crf=self.crf_spin.value(),
            delete_frames_after_encode=self.delete_frames_check.isChecked(),
        )

    def start(self) -> None:
        """Start a job, or say why it cannot start."""
        try:
            config = self.build_config()
        except (ConfigError, OSError) as exc:
            self.status_label.setText(str(exc))
            return
        self._total_frames = 0
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.log_view.clear()
        self.set_running(True)
        self.status_label.setText(f"Starting on {len(config.devices)} device(s)…")
        self._app.start(config)

    def set_running(self, running: bool) -> None:
        """Start and Stop are never both live, whatever the job is doing."""
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.clean_button.setEnabled(not running)

    def open_output_folder(self) -> None:
        self._open(Path(self.output_edit.text()).parent)

    def open_work_dir(self) -> None:
        self._open(self._work_dir)

    def _open(self, directory: Path) -> None:
        if not directory.is_dir():
            self.status_label.setText(f"{directory} does not exist yet.")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory))):
            self.status_label.setText(f"Could not open {directory}.")

    def clean_work_dir(self, parent: QDialog | None = None) -> None:
        """Delete the work directory, after naming it and its size.

        The confirmation is not decoration: on the reference run this directory
        holds 846 GB, and the path is derived from the source file, so the user
        has to see which directory is about to disappear.
        """
        from core.pipeline import work_dir_size

        target = self._work_dir
        if self._app.is_running():
            self.status_label.setText("A job is running; stop it before cleaning.")
            return
        if not target.is_dir():
            self.status_label.setText(f"{target} does not exist.")
            return
        try:
            size = work_dir_size(target)
        except OSError as exc:
            self.status_label.setText(f"Could not read {target}: {exc}")
            return
        answer = QMessageBox.question(
            parent or self,
            "Delete the work directory?",
            f"Delete {target}?\n\n"
            f"{size / 1024**3:.1f} GiB of extracted and upscaled frames will be "
            "removed. The source video and the model are not touched.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.status_label.setText("Nothing was deleted.")
            return
        try:
            self._app.clean_work_dir(self.build_config())
        except (ConfigError, OSError) as exc:
            self.status_label.setText(str(exc))
            return
        self.status_label.setText(f"Deleted {target}.")

    # --- events ---------------------------------------------------------------

    def handle_event(self, event: PipelineEvent) -> None:
        """Render one event. Dispatch on `task_id` — never on `kind` alone."""
        if event.task_id.startswith("probe:"):
            if event.kind == "stage":
                self.status_label.setText(event.message)
                fps = self._fps_from_message(event.message)
                if fps:
                    self.set_source_fps(fps)
            return
        if event.task_id == "devices":
            if event.kind in {"log", "stage"}:
                self.status_label.setText(event.message)
            return
        if event.task_id != "job":
            return

        if event.kind == "stage":
            self.status_label.setText(f"{event.stage}: {event.message}")
            self.log_view.appendPlainText(f"[{event.stage}] {event.message}")
            if event.stage == "upscale":
                self._set_total_from_message(event.message)
        elif event.kind == "device_progress":
            self._update_device_progress(event)
        elif event.kind in {"log", "error"}:
            self.log_view.appendPlainText(event.message)
            if event.kind == "error":
                self.status_label.setText(event.message)
        elif event.kind == "done":
            self.set_running(False)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("done")
            self.status_label.setText(f"Done: {event.message}")
            self.log_view.appendPlainText(f"Done: {event.message}")

    def _set_total_from_message(self, message: str) -> None:
        """Remember how many frames this job has left to do.

        The message is the only place the count appears before the first frame
        is written, and it is this tab's job to read its own pipeline's log.
        """
        if " remaining" not in message:
            return
        try:
            self._total_frames = int(message.rsplit(", ", 1)[-1].split(" ")[0])
        except (ValueError, IndexError):
            self._total_frames = 0
            # Indeterminate beats a wrong number.
            self.progress_bar.setRange(0, 0)

    def _update_device_progress(self, event: PipelineEvent) -> None:
        row = self._row_for_device(event.device_ordinal)
        if row < 0:
            return
        self.device_table.setItem(row, 1, QTableWidgetItem(str(event.processed)))
        self.device_table.setItem(row, 2, QTableWidgetItem(f"{event.fps:.1f}"))
        total = self._total_frames
        if total <= 0:
            return
        done = min(total, self._completed_frames())
        self.progress_bar.setValue(int(100 * done / total))
        self.progress_bar.setFormat(f"{done} / {total} frames")

    def _row_for_device(self, ordinal: int) -> int:
        """The table row for a worker's ordinal in the selection.

        Keyed on the ordinal, never on `Device.index`: that is unique only
        within a vendor's device space, so the CPU (index 0) and the first
        Tesla P40 (index 0) are indistinguishable.
        """
        return ordinal if 0 <= ordinal < len(self._active) else -1

    def _completed_frames(self) -> int:
        total = 0
        for row in range(self.device_table.rowCount()):
            item = self.device_table.item(row, 1)
            if item is not None and item.text().isdigit():
                total += int(item.text())
        return total

    def append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)
