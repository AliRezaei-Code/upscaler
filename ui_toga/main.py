"""Entry point for the Toga front-end.

`multiprocessing.freeze_support()` is the same requirement as the Qt front-end
and for the same reason: a pool worker re-executes this file, and without the
guard each of them builds its own application object on its way to
`upscale_chunk`. Toga is packaged with Briefcase rather than frozen by
PyInstaller — Toga's widget backends cannot be frozen — but the worker
mechanism is identical, so the guard is identical too.
"""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    """Open the window and run the event loop. Returns the exit code."""
    multiprocessing.freeze_support()
    from ui_toga.app import Upscaler

    Upscaler().main_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
