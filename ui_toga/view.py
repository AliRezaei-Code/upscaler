"""The Toga view: the same form as the Qt one, built from Toga's own widgets.

Toga has no checkable list, so every GPU gets a `toga.Switch` inside a
`toga.Box`; it has no tab container, so the four pages are a
`toga.OptionContainer`; and `toga.ProgressBar` carries no text at all, so every
bar is paired with a `toga.Label` that says what the number means. The dialogs
are `async` because Toga's are.

The rules are the Qt front-end's: this module builds a `JobConfig` and renders
`PipelineEvent`s, and does nothing else.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any, cast

import toga
from toga.style import Pack

from core.app import UpscalerApp
from core.config import Device, JobConfig, Precision
from core.errors import ConfigError
from core.events import PipelineEvent
from core.models import ModelEntry, load_catalogue
from core.paths import default_work_dir, models_dir

#: The log's last N lines, for the Qt log's reason: a job is long and the user
#: is not reading all of it an hour later.
LOG_LINES = 2000

BROWSE_LOCAL = "Browse local…"
CATALOGUE_PREFIX = "catalogue:"


def column_style() -> Pack:
    return Pack(direction="column", margin=5)


def row_style() -> Pack:
    return Pack(direction="row", align_items="start", margin=2)


def selected_text(selection: toga.Selection) -> str:
    """The selected item as a string.

    A real platform backend hands `Selection.value` back as the item itself; the
    dummy backend wraps it in a `Row` that carries the item as `.value`. Reading
    both is what lets the same view be driven by the real GTK app and by the
    headless tests.
    """
    value = selection.value
    inner = getattr(value, "value", value)
    return inner if isinstance(inner, str) else ""


def default_output_path(source: Path) -> Path:
    """`<stem>.upscaled<ext>` beside the source, as in the Qt front-end."""
    return source.with_name(f"{source.stem}.upscaled{source.suffix or '.mp4'}")


class UpscaleView:
    """The job form, the device switches, the run bar and the log."""

    def __init__(self, app: UpscalerApp, toga_app: toga.App) -> None:
        self.app = app
        self.toga_app = toga_app
        self.devices: list[Device] = []
        self.local_model: Path | None = None
        self.total_frames = 0
        self._entries: list[ModelEntry] = []
        self._output_edited = False
        self._work_dir = default_work_dir(Path.cwd() / "untitled.mp4")
        self._log_lines: list[str] = []
        self._switches: dict[int, toga.Switch] = {}
        self._progress_by_ordinal: dict[int, int] = {}
        self._pending: list[asyncio.Future[object]] = []
        self._build()

    # --- construction ---------------------------------------------------------

    def _build(self) -> None:
        self.source_input = toga.TextInput(
            placeholder="Choose a video…", on_change=self._on_source
        )
        self.source_button = toga.Button("Browse…", on_press=self.browse_source)
        self.model_selection = toga.Selection(items=[], on_change=self._on_model)
        self.model_label = toga.Label("", style=Pack(margin=2))
        self.output_input = toga.TextInput(
            placeholder="Output path", on_change=self._on_output
        )
        self.output_button = toga.Button("Choose…", on_press=self.browse_output)

        self.device_box = toga.Box(style=column_style(), children=[])
        self.rescan_button = toga.Button("Rescan devices", on_press=self.rescan_devices)

        self.precision = toga.Selection(items=["auto", "fp16", "fp32"], value="auto")
        self.tile = toga.NumberInput(min=0, max=4096, value=0)
        self.fps = toga.NumberInput(min=0, max=1000, value=0)
        self.crf = toga.NumberInput(min=0, max=51, value=18)
        self.delete_frames = toga.Switch("Delete frames after encoding", value=False)

        self.progress = toga.ProgressBar(max=1.0, value=0.0, running=False)
        self.progress_label = toga.Label("no frame count yet")
        self.device_status = toga.Label("No devices yet.")
        self.log_view = toga.MultilineTextInput(readonly=True, value="")
        self.status = toga.Label("Ready.")

        self.start_button = toga.Button("Start", on_press=self.start)
        self.stop_button = toga.Button("Stop", enabled=False, on_press=self.app.stop)
        self.clean_button = toga.Button("Clean work dir", on_press=self.clean_work_dir)
        self.open_output_button = toga.Button(
            "Open output folder", on_press=self.open_output_folder
        )

        self.models_selection = toga.Selection(
            items=[], on_change=self._on_catalogue_entry
        )
        self.models_detail = toga.MultilineTextInput(readonly=True, value="")
        self.community_status = toga.Label("Not fetched.")
        self.runtimes_label = toga.MultilineTextInput(readonly=True, value="")

        self.widget = toga.OptionContainer(
            id="upscaler",
            content=[
                ("Upscale", self._upscale_page()),
                ("Models", self._models_page()),
                ("Community", self._community_page()),
                ("Runtimes", self._runtimes_page()),
            ],
        )
        self._reload_models()

    def _upscale_page(self) -> toga.Box:
        return toga.Box(
            style=column_style(),
            children=[
                _labelled(
                    "Video", toga.Box(children=[self.source_input, self.source_button])
                ),
                _labelled("Model", self.model_selection),
                self.model_label,
                _labelled(
                    "Output", toga.Box(children=[self.output_input, self.output_button])
                ),
                _labelled("Devices", self.device_box),
                self.rescan_button,
                _labelled("Precision", self.precision),
                _labelled("Tile size", self.tile),
                _labelled("Frame rate (0 = from the source)", self.fps),
                _labelled("CRF", self.crf),
                self.delete_frames,
                toga.Divider(),
                _labelled("Progress", self.progress),
                self.progress_label,
                self.device_status,
                toga.Box(
                    children=[
                        self.start_button,
                        self.stop_button,
                        self.clean_button,
                        self.open_output_button,
                    ]
                ),
                self.status,
                toga.ScrollContainer(content=self.log_view, style=Pack(flex=1)),
            ],
        )

    def _models_page(self) -> toga.Box:
        """The catalogue read-out.

        Toga has no columned table, so the list is a `Selection` of one line per
        model and the detail is a read-only text box under it.
        """
        return toga.Box(
            style=column_style(),
            children=[
                toga.Label("The 14 catalogue models, with their licences and sizes."),
                self.models_selection,
                self.models_detail,
            ],
        )

    def _community_page(self) -> toga.Box:
        """OpenModelDB, opened in the user's browser rather than scraped.

        The curated Models tab is the download path in both front-ends; this
        page says so instead of offering a listing that would go stale.
        """
        return toga.Box(
            style=column_style(),
            children=[
                toga.Label(
                    "OpenModelDB is browsed in your browser. The curated Models tab "
                    "is the download path, and every model in it has a verified "
                    "size and digest."
                ),
                self.community_status,
            ],
        )

    def _runtimes_page(self) -> toga.Box:
        """What is on disk, read with no network and no imports of the engines."""
        return toga.Box(
            style=column_style(),
            children=[
                toga.Label(
                    "Optional runtimes live in the data directory. The PySide6 "
                    "front-end installs them from its Runtimes tab; this build "
                    "reads what is already there."
                ),
                self.runtimes_label,
            ],
        )

    def _reload_models(self) -> None:
        try:
            self._entries = load_catalogue()
        except Exception as exc:  # shown where the user is looking
            self._entries = []
            self.model_label.text = f"Catalogue unavailable: {exc}"
            self.model_selection.items = [str(exc)]
            self.models_selection.items = [str(exc)]
            return
        self.model_label.text = ""
        items = [
            f"{CATALOGUE_PREFIX}{entry.id}|{entry.name} — {entry.scale}x — "
            f"{entry.architecture}"
            for entry in self._entries
        ]
        items.append(BROWSE_LOCAL)
        self.model_selection.items = items
        self.models_selection.items = [
            f"{entry.name} — {entry.scale}x — {entry.architecture} — "
            f"{entry.size_bytes / 1024**2:.1f} MiB — {entry.license}"
            for entry in self._entries
        ]

    def set_runtimes(self, text: str) -> None:
        self.runtimes_label.value = text

    def set_community(self, text: str) -> None:
        self.community_status.text = text

    # --- devices --------------------------------------------------------------

    def rescan_devices(self) -> None:
        """Probe on a worker thread: a wedged driver costs ten seconds."""

        def _probe(
            emit: Callable[[PipelineEvent], None], _stop: object
        ) -> list[Device]:
            from core.devices import probe_all

            return probe_all(emit)

        self.status.text = "Probing devices…"
        self.app.submit("devices", _probe)

    def set_devices(self, devices: list[Device]) -> None:
        """One switch per device, with the reason spelled out for the greyed ones."""
        self.devices = list(devices)
        self._switches.clear()
        # `toga.Box.children` has no setter in 0.5.6; `clear()` plus `add()` is
        # the supported way to change a box's contents.
        self.device_box.clear()
        for ordinal, device in enumerate(self.devices):
            switch = toga.Switch(
                device.label,
                value=device.usable,
                enabled=device.usable,
                on_change=self._on_device_toggle,
            )
            self._switches[ordinal] = switch
            self.device_box.add(switch)
            if not device.usable:
                self.device_box.add(
                    toga.Label(
                        f"    {device.unusable_reason or ''}", style=Pack(margin=2)
                    )
                )
        unusable = [device for device in devices if not device.usable]
        count = f"{len(devices)} devices"
        if unusable:
            count += f", {len(unusable)} unavailable"
        self.device_status.text = count
        self._on_device_toggle(None)

    def selected_devices(self) -> list[Device]:
        """The devices whose switch is on and which are usable."""
        return [
            device
            for ordinal, device in enumerate(self.devices)
            if ordinal in self._switches
            and self._switches[ordinal].value
            and device.usable
        ]

    def _device_label(self, ordinal: int) -> str:
        """`GPU 0 — Tesla P40` for one device.

        Not the name: three Tesla P40s would render as three identical labels in
        a progress readout, which is the same reason the Qt table keys on the
        ordinal.
        """
        if 0 <= ordinal < len(self.devices):
            device = self.devices[ordinal]
            kind = "CPU" if device.vendor == "cpu" else f"GPU {device.index}"
            return f"{kind} — {device.name}"
        return f"device {ordinal}"

    def _on_device_toggle(self, _widget: object) -> None:
        names = ", ".join(
            self._device_label(ordinal)
            for ordinal, device in enumerate(self.devices)
            if device in self.selected_devices()
        )
        self.device_status.text = f"Running on: {names or 'nothing selected'}"

    # --- source, model, output ------------------------------------------------

    async def _ask(self, dialog: object) -> object:
        """Await a Toga dialog and return its result.

        `toga.App.dialog` is annotated as returning a Coroutine *of* a
        Coroutine. It is an `async def` that yields the dialog's result, so one
        `await` is what the runtime does; this helper is the one place that has
        to know the annotation is wrong.
        """
        return await cast(
            "Awaitable[object]",
            self.toga_app.dialog(cast("Any", dialog)),
        )

    async def browse_source(self, _widget: object) -> None:
        chosen = await self._ask(
            toga.OpenFileDialog(
                "Choose a video",
                file_types=["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"],
            )
        )
        if chosen:
            self.set_source(Path(str(chosen)))

    def set_source(self, path: Path) -> None:
        self.source_input.value = str(path)
        if not self._output_edited:
            self.output_input.value = str(default_output_path(path))
        self._work_dir = default_work_dir(path)
        self._probe_source(path)

    def _on_source(self, _widget: object) -> None:
        if self.source_input.value:
            self._work_dir = default_work_dir(Path(self.source_input.value))

    def _on_output(self, _widget: object) -> None:
        self._output_edited = True

    async def browse_output(self, _widget: object) -> None:
        chosen = await self._ask(
            toga.SaveFileDialog("Save the result as", self.output_input.value)
        )
        if chosen:
            self.output_input.value = str(chosen)
            self._output_edited = True

    async def browse_model(self, _widget: object) -> None:
        chosen = await self._ask(
            toga.OpenFileDialog(
                "Choose a model",
                file_types=["*.pth", "*.ckpt", "*.safetensors", "*.onnx"],
            )
        )
        if chosen:
            self.set_local_model(Path(str(chosen)))

    def _probe_source(self, path: Path) -> None:
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

        self.app.submit(f"probe:{path}", _probe)

    # --- model ----------------------------------------------------------------

    def _on_model(self, _widget: object) -> None:
        value = selected_text(self.model_selection)
        if value.startswith(CATALOGUE_PREFIX):
            entry = self.current_model()
            if entry is not None:
                self.model_label.text = f"{entry.notes} — {entry.license}"
                self.local_model = None
        elif value == BROWSE_LOCAL:
            # Toga's handler wrapper awaits a coroutine handler, so the browse
            # dialog runs from here rather than from a sync callback that cannot.
            self._schedule(self.browse_model(None))

    def _schedule(self, coro: Coroutine[object, object, None]) -> None:
        """Start a coroutine on the app's loop from a synchronous handler.

        Toga's handler wrapper already awaits a coroutine handler, but
        `_on_model` is called from a `Selection` change, and a selection change
        has nowhere to await.
        """
        self._pending.append(asyncio.ensure_future(coro))

    def set_local_model(self, path: Path) -> None:
        """Use a model from outside the catalogue.

        The selection is moved to "Browse local…" as well, because
        `model_path()` asks the selection first: leaving a catalogue entry
        selected while a local file is chosen would silently run the catalogue
        model instead.
        """
        self.local_model = path
        self.model_selection.value = BROWSE_LOCAL
        self.model_label.text = f"Model: {path}"

    def current_model(self) -> ModelEntry | None:
        value = selected_text(self.model_selection)
        if not value.startswith(CATALOGUE_PREFIX):
            return None
        wanted = value.split("|", 1)[0][len(CATALOGUE_PREFIX) :]
        for entry in self._entries:
            if entry.id == wanted:
                return entry
        return None

    def _on_catalogue_entry(self, _widget: object) -> None:
        value = selected_text(self.models_selection)
        if not value:
            return
        for entry in self._entries:
            if value.startswith(entry.name):
                self.models_detail.value = (
                    f"{entry.name}\n{entry.architecture}, {entry.scale}x\n"
                    f"{entry.size_bytes / 1024**2:.1f} MiB\nLicence: {entry.license}\n"
                    f"{entry.url}\n\n{entry.notes}"
                )
                return

    # --- the run bar ----------------------------------------------------------

    def model_path(self) -> Path:
        entry = self.current_model()
        if entry is not None:
            return models_dir() / entry.filename
        if self.local_model is not None:
            return self.local_model
        raise ConfigError("Choose a model first")

    def build_config(self) -> JobConfig:
        devices = tuple(self.selected_devices())
        if not devices:
            raise ConfigError("Select at least one GPU")
        return JobConfig(
            input_path=Path(self.source_input.value),
            output_path=Path(self.output_input.value),
            model_path=self.model_path(),
            work_dir=self._work_dir,
            devices=devices,
            precision=cast(Precision, str(self.precision.value)),
            tile_size=int(self.tile.value or 0),
            fps=float(self.fps.value or 0) or None,
            crf=int(self.crf.value or 0),
            delete_frames_after_encode=bool(self.delete_frames.value),
        )

    def start(self, _widget: object = None) -> None:
        try:
            config = self.build_config()
        except (ConfigError, OSError) as exc:
            self.status.text = str(exc)
            return
        self.progress.value = 0.0
        self.progress.running = True
        self.total_frames = 0
        self._progress_by_ordinal.clear()
        self.set_running(True)
        self.status.text = f"Starting on {len(config.devices)} device(s)…"
        self.app.start(config)

    def set_running(self, running: bool) -> None:
        self.start_button.enabled = not running
        self.stop_button.enabled = running
        self.clean_button.enabled = not running

    def open_output_folder(self, _widget: object) -> None:
        target = Path(self.output_input.value).parent
        if not target.is_dir():
            self.status.text = f"{target} does not exist yet."

    def clean_work_dir(self, _widget: object) -> None:
        """Name the directory and its size, then ask before deleting it.

        The confirmation is a `ConfirmDialog`, which is asynchronous; a tab
        that deleted 846 GB from a single click would be the worst bug in the
        application, and Toga's dialog is the same `ConfirmDialog` the Qt front-
        end's `QMessageBox` is.
        """
        from core.pipeline import work_dir_size

        target = self._work_dir
        if self.app.is_running():
            self.status.text = "A job is running; stop it before cleaning."
            return
        if not target.is_dir():
            self.status.text = f"{target} does not exist."
            return
        size = work_dir_size(target)
        self.status.text = (
            f"{target} holds {size / 1024**3:.1f} GiB. Use the PySide6 front-end's "
            "Clean work dir, which asks for confirmation on the same path."
        )
        self._schedule(self._confirm_delete(target, size))

    async def _confirm_delete(self, target: Path, size: int) -> None:
        answer = await self.toga_app.dialog(
            toga.ConfirmDialog(
                "Delete the work directory?",
                f"Delete {target}?\n\n{size / 1024**3:.1f} GiB of extracted and "
                "upscaled frames will be removed. The source video and the model "
                "are not touched.",
            )
        )
        if not answer:
            self.status.text = "Nothing was deleted."
            return
        from core.config import JobConfig

        config = JobConfig(
            input_path=Path(self.source_input.value or target / "none.mp4"),
            output_path=Path(self.output_input.value or target / "none.mp4"),
            model_path=self.model_path(),
            work_dir=target,
            devices=(),
        )
        try:
            self.app.clean_work_dir(config)
        except (ConfigError, OSError) as exc:
            self.status.text = str(exc)
            return
        self.status.text = f"Deleted {target}."

    # --- events ---------------------------------------------------------------

    def handle_event(self, event: PipelineEvent) -> None:
        """Render one event, dispatched on `task_id`."""
        if event.task_id.startswith("probe:"):
            if event.kind == "stage":
                self.status.text = event.message
                fps = self._fps_from_message(event.message)
                if fps:
                    self.fps.value = fps
            return
        if event.task_id == "devices":
            if event.kind in {"log", "stage"}:
                self.status.text = event.message
            return
        if event.task_id != "job":
            return

        if event.kind == "stage":
            self.status.text = f"{event.stage}: {event.message}"
            self.append_log(f"[{event.stage}] {event.message}")
            if event.stage == "upscale":
                self._set_total(event.message)
        elif event.kind == "device_progress":
            self._progress_by_ordinal[event.device_ordinal] = event.processed
            self._refresh_progress()
        elif event.kind in {"log", "error"}:
            self.append_log(event.message)
            if event.kind == "error":
                self.status.text = event.message
        elif event.kind == "done":
            self.progress.running = False
            self.progress.value = 1.0
            self.progress_label.text = "done"
            self.set_running(False)
            self.status.text = f"Done: {event.message}"
            self.append_log(f"Done: {event.message}")

    @staticmethod
    def _fps_from_message(message: str) -> float | None:
        if " at " not in message or " fps" not in message:
            return None
        try:
            return float(message.split(" at ")[-1].split(" fps")[0])
        except ValueError:
            return None

    def _set_total(self, message: str) -> None:
        if " remaining" not in message:
            return
        try:
            self.total_frames = int(message.rsplit(", ", 1)[-1].split(" ")[0])
        except (ValueError, IndexError):
            self.total_frames = 0
        self._refresh_progress()

    def _refresh_progress(self) -> None:
        """The bar and its label, from the per-device counts.

        Toga's `ProgressBar` has no text, so the label is the only place the
        user sees "37 / 60" — and it is the only place they can see that a
        second device is not contributing at all.
        """
        total = self.total_frames
        done = min(total, sum(self._progress_by_ordinal.values())) if total > 0 else 0
        if total > 0:
            self.progress.value = done / total
        parts: list[str] = []
        for ordinal, count in self._progress_by_ordinal.items():
            if count <= 0 or not 0 <= ordinal < len(self.devices):
                continue
            parts.append(f"{self._device_label(ordinal)}: {count}")
        summary = f"{done} / {total} frames" if total > 0 else "no frame count yet"
        self.progress_label.text = (
            f"{summary} — {', '.join(parts)}" if parts else summary
        )

    def append_log(self, text: str) -> None:
        self._log_lines.append(text)
        del self._log_lines[:-LOG_LINES]
        self.log_view.value = "\n".join(self._log_lines)


def _labelled(text: str, widget: toga.Widget) -> toga.Box:
    return toga.Box(
        children=[toga.Label(text, style=Pack(width=30)), widget], style=row_style()
    )
