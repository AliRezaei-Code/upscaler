"""The Toga application: the same job, driven by the same core.

Toga has no `QThread` and no signal/slot, so the shape differs while the job is
identical. One plain `threading.Thread` reads `UpscalerApp.events()` and pushes
into a `queue.Queue`; one asyncio task — the loop Toga already runs — drains that
queue and touches widgets. A thread and an event loop meet exactly once, at that
queue, which is the only safe place for them to meet.

**The lifecycle is the other thing to know.** `toga.App.__init__` constructs the
platform app, which calls `startup()` *before* the constructor returns. So
everything `startup()` touches is assigned before `super().__init__()` is
called, and the drain task is created in `on_running()`, which is when the loop
exists.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading

import toga

from core.app import UpscalerApp
from core.paths import log_file
from ui_toga.view import UpscaleView

#: How often the loop drains the bridge. Toga's loop is idle by design, so this
#: costs nothing when there is no job, and 0.05 s is imperceptible when there is.
DRAIN_INTERVAL_SECONDS = 0.05

APP_ID = "org.alirezaei.upscaler"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class Upscaler(toga.App):
    """The Toga front-end of the same application."""

    def __init__(self, **kwargs: object) -> None:
        # Before `super().__init__`, because Toga calls `startup()` from inside
        # it. See the module docstring.
        self.core = UpscalerApp()
        self.view: UpscaleView | None = None
        self._bridge: queue.Queue[object] = queue.Queue()
        self._pump: threading.Thread | None = None
        self._stop = threading.Event()
        super().__init__(
            formal_name="Upscaler",
            app_id=APP_ID,
            app_name="Upscaler",
            author="AliRezaei-Code",
            version="0.1.0",
            description="Cross-platform GUI video upscaler",
            home_page="https://github.com/AliRezaei-Code/upscaler",
            **kwargs,  # type: ignore[arg-type]
        )

    def startup(self) -> None:
        """Build the window and start the forwarder thread."""
        self.view = UpscaleView(self.core, self)
        window = toga.MainWindow(title="Upscaler", size=(980, 760))  # type: ignore[no-untyped-call]
        window.content = self.view.widget
        window.show()
        # `App.main_window` is what the platform backend and the About box read.
        self.main_window = window  # type: ignore[assignment]
        self._pump = threading.Thread(
            target=self._forward_events, name="toga-events", daemon=True
        )
        self._pump.start()
        self.view.rescan_devices()
        self._log_resolved_runtimes()

    def on_running(self) -> None:
        """Start the drain task, which needs a running loop."""
        self.loop.create_task(self._drain())

    def shutdown(self) -> None:
        """Stop the forwarder before Toga tears the loop down.

        Setting the flag is not enough: the thread is parked inside
        `UpscalerApp.events()`, which returns only when a task pushes its
        sentinel. So a no-op task is submitted to wake it, and the thread leaves
        at the next pass. Without the wake-up the join times out and every
        `Upscaler` instance leaves a thread behind — five tests, five threads.
        """
        if self._pump is None:
            return
        self._stop.set()
        self.core.submit("shutdown", lambda emit, stop: None)
        self._pump.join(timeout=5.0)
        self._pump = None

    def _forward_events(self) -> None:
        """Producer side: the core's queue into the bridge.

        `UpscalerApp.events()` returns at each task's sentinel, so this is a
        loop rather than a single pass.
        """
        while not self._stop.is_set():
            for event in self.core.events():
                if self._stop.is_set():
                    return
                self._bridge.put(event)

    async def _drain(self) -> None:
        """Consumer side: the bridge into widgets, on the event loop."""
        view = self.view
        if view is None:  # pragma: no cover - startup always builds the view
            return
        while True:
            while True:
                try:
                    event = self._bridge.get_nowait()
                except queue.Empty:
                    break
                view.handle_event(event)  # type: ignore[arg-type]
            await asyncio.sleep(DRAIN_INTERVAL_SECONDS)

    def _log_resolved_runtimes(self) -> None:
        """Record which torch and onnxruntime were loaded, for bug reports.

        Two lines in a log file are the difference between "it used the bundled
        CUDA" and a guess. Neither module is imported at module scope: the slim
        build ships neither, and their absence is not an error.
        """
        root = logging.getLogger()
        if not root.handlers:
            handler = logging.FileHandler(log_file(), encoding="utf-8")
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            root.addHandler(handler)
        log = logging.getLogger(__name__)
        for name in ("onnxruntime", "torch"):
            try:
                module = __import__(name)
            except ImportError:
                log.info("%s is not installed", name)
                continue
            log.info("%s from %s", name, getattr(module, "__file__", "?"))


def main() -> int:
    """Run the Toga front-end. Returns the exit code."""
    Upscaler().main_loop()
    return 0
