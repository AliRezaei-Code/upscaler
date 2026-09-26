"""Ctrl-C that stops a job instead of hanging the application.

**Call `install()` once, from the process main thread, before the first
`UpscalerApp.start()`.** `signal.signal` raises `ValueError: signal only works
in main thread of the main interpreter` anywhere else, and a job runs on a
UI-spawned thread, so installing from inside `run_job` would raise on every
single job rather than on the first.

The handler sets the cancel event, kills the worker pids and **returns**. It
never calls `sys.exit` and never raises: `SystemExit` raised inside a
`with ProcessPoolExecutor(...)` block runs `__exit__` → `shutdown(wait=True)` →
an unbounded `join` on workers that are still holding a GPU, which is the
"Ctrl-C does nothing and then the app hangs" bug. Letting `KeyboardInterrupt`
escape does the same thing, so this handler is installed for `SIGINT` as well
as `SIGTERM` and swallows it deliberately.
"""

from __future__ import annotations

import os
import signal
import threading
from collections.abc import Callable
from types import FrameType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Imported for the type only: `core.worker` pulls in numpy and the backend
    # ABC, and a signal handler has to be installable before any of that
    # loads. The Protocol is structural, so a real `ctx.Event()` satisfies it.
    from .worker import CancelEvent

_INSTALLED_SIGNALS = (signal.SIGINT, signal.SIGTERM)

# Windows has no SIGKILL; `os.kill(pid, signal.SIGTERM)` there calls
# TerminateProcess, which is the same untrappable kill with a POSIX name.
_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)

# What `signal.getsignal` gives back: our handler, SIG_DFL, SIG_IGN, or None on
# a platform where a signal was never registered.
PreviousHandler = Callable[[int, "FrameType | None"], Any] | int | None

_lock = threading.Lock()
_previous: dict[int, PreviousHandler] = {}
_installed = False


def install(get_pids: Callable[[], list[int]], cancel: CancelEvent) -> None:
    """Route `SIGINT` and `SIGTERM` to a cancel-and-kill handler.

    `get_pids` is called once per signal, from the handler, so it must be
    cheap and must not block: it snapshots the worker pids, and a caller that
    has none returns an empty list. `cancel` is the event `run_job` watches.

    Calling this a second time restores the handlers the first call replaced
    and installs these arguments instead of silently keeping the old ones, so
    a caller that re-plumbs the cancel event gets the new event rather than a
    handler wired to a dead one. `uninstall()` afterwards still restores the
    handlers that were in place before the *first* install.
    """
    if threading.current_thread() is not threading.main_thread():
        raise ValueError(
            "signals.install() must be called from the process main thread; "
            "signal.signal only works in the main thread of the main interpreter"
        )
    global _installed
    with _lock:
        if _installed:
            _restore_locked()
        handler = _make_handler(get_pids, cancel)
        for signum in _INSTALLED_SIGNALS:
            _previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)
        _installed = True


def uninstall() -> None:
    """Put back the handlers that were in place before `install()`.

    Idempotent, and it has the same main-thread requirement as `install()`:
    with nothing installed it touches no signal and cannot fail, from any
    thread.
    """
    with _lock:
        _restore_locked()


def _restore_locked() -> None:
    global _installed
    for signum, previous in _previous.items():
        signal.signal(signum, previous)
    _previous.clear()
    _installed = False


def _make_handler(
    get_pids: Callable[[], list[int]], cancel: CancelEvent
) -> Callable[[int, FrameType | None], None]:
    def handler(signum: int, frame: FrameType | None) -> None:
        # The snapshot is taken once and iterated, so a pid list that changes
        # underneath the handler cannot make it iterate a mutating list.
        cancel.set()
        pids = get_pids()
        own = os.getpid()
        for pid in pids:
            if pid == own:
                # Our own pid. A provider that returns the parent — or the
                # pool's own bookkeeping on a platform where `os.kill` is
                # TerminateProcess — would take the UI down with the workers it
                # meant to be stopping, with no traceback and no chance to
                # finish the cancel. Measured: on windows-latest this killed
                # the pytest process mid-suite, exit code 2, no output.
                continue
            if pid <= 0:
                # `os.kill(0, ...)` signals the caller's whole process group
                # and a negative pid signals the group with that id, so a
                # provider that returns one of these takes the application
                # down with the workers it meant to be stopping.
                continue
            try:
                os.kill(pid, _KILL_SIGNAL)
            except OSError:
                # A worker that exited between the snapshot and the kill is
                # the normal race here, and a handler that raises turns a
                # Ctrl-C into a crash at an arbitrary point in the frame it
                # interrupted. Cleanup continues through the cancel event.
                continue

    return handler
