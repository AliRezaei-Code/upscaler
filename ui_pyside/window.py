"""The main window: four tabs, one event pump.

Every tab is handed the same `UpscalerApp` and the same `EventBridge`, so there
is exactly one queue, exactly one thread crossing from a producer to the GUI,
and one place that decides which widget an event belongs to. The window is the
only place that knows about tabs; no tab imports another.
"""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget, QWidget

from core.app import UpscalerApp
from core.events import PipelineEvent
from ui_pyside.worker_bridge import EventBridge, EventPump

APP_NAME = "Upscaler"


class MainWindow(QMainWindow):
    """The window, and the only dispatcher of pipeline events to widgets."""

    def __init__(self, app: UpscalerApp | None = None) -> None:
        super().__init__()
        self.app = app if app is not None else UpscalerApp()
        self.setWindowTitle(APP_NAME)
        self.resize(1100, 820)

        self.bridge = EventBridge()
        self.pump: EventPump | None = None
        self.bridge.event.connect(self._dispatch)
        self.bridge.finished.connect(self._on_drained)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        from ui_pyside.upscale_tab import UpscaleTab

        self.upscale_tab = UpscaleTab(self.app, self)
        self.tabs.addTab(self.upscale_tab, "Upscale")
        self._add_optional_tabs()

    def _add_optional_tabs(self) -> None:
        """Add the tabs that are importable, so a partial install still runs.

        A missing tab must not stop the Upscale tab from opening: the whole
        point of this application is to run a job, and a user whose PySide6
        install is missing `selectolax` still needs that.
        """
        from ui_pyside.community_tab import CommunityTab
        from ui_pyside.models_tab import ModelsTab
        from ui_pyside.runtimes_tab import RuntimesTab

        self.models_tab = ModelsTab(self.app, self)
        self.tabs.addTab(self.models_tab, "Models")
        self.community_tab = CommunityTab(self.app, self)
        self.tabs.addTab(self.community_tab, "Community")
        self.runtimes_tab = RuntimesTab(self.app, self)
        self.tabs.addTab(self.runtimes_tab, "Runtimes")

    def start_pump(self) -> None:
        """Begin draining the app's queue. Safe to call once."""
        if self.pump is not None:
            return
        self.pump = EventPump(self.app, self.bridge)
        self.pump.start()

    def stop_pump(self) -> None:
        if self.pump is not None:
            self.pump.requestInterruption()
            self.pump.wait(2000)
            self.pump = None

    def _on_drained(self) -> None:
        self.statusBar().showMessage("Idle")

    def _dispatch(self, event: object) -> None:
        """Route one event. Every branch is on `task_id`, never on `kind`."""
        if not isinstance(event, PipelineEvent):
            return
        if event.task_id == "job":
            self.upscale_tab.handle_event(event)
        elif event.task_id == "devices":
            devices = self.app.results.get("devices")
            if devices:
                self.upscale_tab.set_devices(list(devices))
        elif event.task_id == "community" and hasattr(self, "community_tab"):
            self.community_tab.handle_event(event)
        elif event.task_id.startswith("model:") and hasattr(self, "models_tab"):
            self.models_tab.handle_event(event)
        elif event.task_id.startswith("runtime:") and hasattr(self, "runtimes_tab"):
            self.runtimes_tab.handle_event(event)
        elif event.task_id.startswith("probe:") or event.task_id == "devices":
            self.upscale_tab.handle_event(event)

    def closeEvent(self, event: object) -> None:
        """Stop the pump before Qt destroys the widgets it emits to."""
        self.stop_pump()
        if event is not None and hasattr(event, "accept"):
            event.accept()  # type: ignore[attr-defined]


def build_window(app: UpscalerApp | None = None) -> tuple[MainWindow, QWidget]:
    """The window and its central widget, for tests and for `main()`."""
    window = MainWindow(app)
    return window, window.centralWidget()
