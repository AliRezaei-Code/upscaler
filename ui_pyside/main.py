"""Entry point.

Two module-scope statements and one guard, and all three are load-bearing.

`multiprocessing.freeze_support()` **must** run before anything else. In a
frozen build the child process re-executes this file; without the guard it
executes it as `__mp_main__`, constructs its own `QApplication`, and opens its
own window — so the first run of a packaged app shows one window per device
plus one, all fighting over the same work directory.

`set_start_method("spawn", force=True)` is module scope and parent-only on
purpose. The start method is fixed when a child process is created, so setting
it in a pool *initializer* — the obvious place, next to the cancel event — is a
no-op. Spawning also means a worker does not inherit the Qt event loop, the
device handles or the loaded model, which is why `freeze_support` and
`set_start_method` have to be here rather than inside `run_job`.
"""

from __future__ import annotations

import multiprocessing as mp
import sys

from PySide6.QtWidgets import QApplication

from core.app import UpscalerApp
from core.paths import log_file

#: Qt owns stdout on some platforms, so the log is a file as well as the widget.
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging() -> None:
    """Write the same log the UI shows to a file, for bug reports."""
    import logging

    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.FileHandler(log_file(), encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def main() -> int:
    """Open the window and run the event loop. Returns the exit code."""
    if not getattr(sys, "frozen", False):
        # Only meaningful in a frozen build, and it must be the first thing that
        # runs: see the module docstring.
        mp.freeze_support()
    configure_logging()
    mp.set_start_method("spawn", force=True)
    qt_app = QApplication(sys.argv[:1])
    from ui_pyside.window import MainWindow

    window = MainWindow(UpscalerApp())
    window.start_pump()
    window.show()
    return int(qt_app.exec())


# `multiprocessing.freeze_support()` is idempotent, but the guard below is what
# makes `python -m ui_pyside.main` and the frozen executable behave the same.
if __name__ == "__main__":
    sys.exit(main())
