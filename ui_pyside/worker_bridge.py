"""The Qt thread that carries `PipelineEvent` from the app to the widgets.

`UpscalerApp.events()` is a blocking generator over a `queue.Queue`, and a Qt
widget may only be touched from the GUI thread. So one `QThread` per running
app drains the queue and re-emits each event as a **queued** signal, which Qt
delivers on the GUI thread. No widget in this application is ever touched from
a worker, and no Qt call is ever made from a pipeline thread.

The one rule the front-end has to respect: dispatch on `event.task_id`. A model
download and a job can be in flight at once, and without that key a download's
progress would land in the job's progress bar.

This module is excluded from `mypy --strict` (see `pyproject.toml`): PySide6's
`Signal`/`Slot` decorators erase the types of everything they wrap, so a strict
run over the class produces errors that describe the stubs rather than the
code.
"""

from __future__ import annotations

from collections.abc import Iterator

from PySide6.QtCore import QObject, QThread, Signal

from core.app import UpscalerApp
from core.events import PipelineEvent


class EventBridge(QObject):
    """Re-emits pipeline events on the GUI thread.

    Attributes:
        event: Emitted once per `PipelineEvent`, carrying the dataclass itself.
            `PipelineEvent` is frozen and small, and a queued connection with a
            Python object is what `QThread` is for; unpacking it into a dozen
            signal arguments would be a worse trade.
        finished: Emitted when the app's queue is drained and the app has
            nothing left to say. A tab that keeps a spinner uses this.
    """

    event = Signal(object)
    finished = Signal()


class EventPump(QThread):
    """Drains `UpscalerApp.events()` and re-emits it through an `EventBridge`.

    One pump per app, started when the window opens and stopped when it closes.
    `events()` returns when the app pushes its sentinel, so the loop below ends
    on its own; `stop()` exists for the case where the window closes with a job
    still running, and it is the reason the loop is a `while True` with a
    `requestInterruption` check rather than a `for` over the generator.
    """

    def __init__(self, app: UpscalerApp, bridge: EventBridge) -> None:
        super().__init__()
        self._app = app
        self._bridge = bridge

    def run(self) -> None:
        """Body of the thread: yield every event, then say so."""
        try:
            for event in self._iter_events():
                if self.isInterruptionRequested():
                    return
                self._bridge.event.emit(event)
            self._bridge.finished.emit()
        except RuntimeError:
            # The bridge was destroyed with the window while this thread was
            # mid-emit. There is nothing to report to and nothing to repair.
            return

    def _iter_events(self) -> Iterator[PipelineEvent]:
        while not self.isInterruptionRequested():
            yield from self._app.events()
