from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

import core.signals as signals_module
from core.signals import install, uninstall


@pytest.fixture(autouse=True)
def handlers_restored() -> Iterator[None]:
    """Put the process's signal handlers back whatever a test did to them.

    A leaked handler in the pytest process would make an unrelated test's
    Ctrl-C cancel nothing and kill an arbitrary pid.
    """
    before = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    yield
    uninstall()
    for signum, handler in before.items():
        signal.signal(signum, handler)


def _cancel_event() -> threading.Event:
    """A cancel handle of the kind `run_job` watches.

    `threading.Event` rather than a `multiprocessing` one because the handler
    only ever needs `is_set()` and `set()`, which is exactly why the cancel
    argument is a Protocol and not a concrete class.
    """
    return threading.Event()


def _sleeping_child() -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


# A pid no running system can have: `pid_max` is 4194304 on Linux and 99999 on
# macOS. Killing it raises `ProcessLookupError`, which is what these tests want
# from the handler — the call was made, and the handler survived the answer.
A_PID_THAT_CANNOT_EXIST = 2**31 - 1


def _install_recording(cancel: threading.Event) -> list[list[int]]:
    calls: list[list[int]] = []

    def get_pids() -> list[int]:
        pids = [A_PID_THAT_CANNOT_EXIST]
        calls.append(pids)
        return pids

    install(get_pids, cancel)
    return calls


# --- install / uninstall ------------------------------------------------------


def test_uninstall_puts_back_the_handler_that_was_there_before() -> None:
    before_int = signal.getsignal(signal.SIGINT)
    before_term = signal.getsignal(signal.SIGTERM)
    install(lambda: [], _cancel_event())

    assert signal.getsignal(signal.SIGINT) is not before_int
    assert signal.getsignal(signal.SIGTERM) is not before_term

    uninstall()

    assert signal.getsignal(signal.SIGINT) is before_int
    assert signal.getsignal(signal.SIGTERM) is before_term


def test_uninstall_twice_changes_nothing() -> None:
    before = signal.getsignal(signal.SIGINT)
    install(lambda: [], _cancel_event())
    uninstall()

    uninstall()

    assert signal.getsignal(signal.SIGINT) is before


def test_installing_twice_restores_rather_than_nesting() -> None:
    before = signal.getsignal(signal.SIGINT)
    install(lambda: [], _cancel_event())
    first = signal.getsignal(signal.SIGINT)

    install(lambda: [], _cancel_event())

    assert signal.getsignal(signal.SIGINT) is not first
    uninstall()
    assert signal.getsignal(signal.SIGINT) is before


# --- what the handler does ---------------------------------------------------


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_the_handler_sets_the_cancel_event_and_returns(
    signum: signal.Signals,
) -> None:
    cancel = _cancel_event()
    _install_recording(cancel)
    handler = signal.getsignal(signum)
    assert callable(handler)

    handler(signum, None)  # must return, not raise and not sys.exit

    assert cancel.is_set()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_one_signal_asks_for_the_pids_once(signum: signal.Signals) -> None:
    calls = _install_recording(_cancel_event())
    handler = signal.getsignal(signum)
    assert callable(handler)

    handler(signum, None)

    assert len(calls) == 1


def test_a_process_group_pid_does_not_take_the_app_down_with_the_workers() -> None:
    # `os.kill(0, ...)` signals the caller's whole process group, which under a
    # frozen build is the application itself. Without the guard this test
    # would take the pytest process with it: reaching the assertion is the
    # assertion.
    cancel = _cancel_event()
    install(lambda: [0, -1, 0], cancel)

    os.kill(os.getpid(), signal.SIGINT)

    assert cancel.is_set()


def test_the_pids_the_handler_is_given_are_killed() -> None:
    child = _sleeping_child()
    try:
        assert child.poll() is None
        cancel = _cancel_event()

        def get_pids() -> list[int]:
            return [child.pid]

        install(get_pids, cancel)
        os.kill(os.getpid(), signal.SIGINT)

        assert cancel.is_set()
        # A killed child is reaped here; if the handler had left it running,
        # this would block for the full 120 s the child sleeps.
        assert child.wait(timeout=10) != 0
    finally:
        if child.poll() is None:  # pragma: no cover - only on a handler failure
            child.kill()
            child.wait(timeout=10)


def test_a_pid_that_is_already_gone_does_not_stop_the_handler() -> None:
    # The normal race: a worker exits between the snapshot and the kill, and a
    # handler that raised there would crash the app at an arbitrary point.
    cancel = _cancel_event()
    finished = _sleeping_child()
    finished.kill()
    finished.wait(timeout=10)
    still_running = _sleeping_child()
    try:

        def get_pids() -> list[int]:
            return [finished.pid, still_running.pid]

        install(get_pids, cancel)
        os.kill(os.getpid(), signal.SIGTERM)

        assert cancel.is_set()
        assert still_running.wait(timeout=10) != 0
    finally:
        if still_running.poll() is None:  # pragma: no cover
            still_running.kill()
            still_running.wait(timeout=10)


def test_a_signal_with_no_worker_pids_still_cancels() -> None:
    cancel = _cancel_event()
    install(lambda: [], cancel)

    os.kill(os.getpid(), signal.SIGINT)

    assert cancel.is_set()


def test_a_new_cancel_event_is_the_one_the_second_install_wired_up() -> None:
    first, second = _cancel_event(), _cancel_event()
    install(lambda: [], first)
    install(lambda: [], second)

    os.kill(os.getpid(), signal.SIGINT)

    assert second.is_set()
    assert not first.is_set()


# --- the main-thread requirement ----------------------------------------------


def test_installing_off_the_main_thread_says_so() -> None:
    # pytest runs tests on the main thread, so this is the only place the
    # restriction can be shown to be checked rather than assumed.
    failures: list[BaseException] = []

    def from_a_worker_thread() -> None:
        try:
            install(lambda: [], _cancel_event())
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=from_a_worker_thread, name="ui-job")
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert "main thread" in str(failures[0])


def test_a_failed_install_off_the_main_thread_leaves_the_handlers_alone() -> None:
    before = signal.getsignal(signal.SIGINT)
    failures: list[BaseException] = []

    def from_a_worker_thread() -> None:
        try:
            install(lambda: [], _cancel_event())
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=from_a_worker_thread)
    worker.start()
    worker.join(timeout=10)

    assert failures and isinstance(failures[0], ValueError)
    assert signal.getsignal(signal.SIGINT) is before
    assert signals_module._installed is False


# --- the module's own surface -------------------------------------------------


def test_the_module_never_imports_a_heavy_runtime() -> None:
    # The UI must be able to import this before torch or onnxruntime exist.
    source = Path(signals_module.__file__).read_text(encoding="utf-8")
    for heavy in ("import torch", "import cv2", "import onnxruntime"):
        assert heavy not in source


def test_the_handler_never_signals_its_own_process_or_a_group() -> None:
    """A pid list containing this process must not take this process down.

    Measured on windows-latest: the handler killed the pytest process itself,
    so the suite died mid-run with exit code 2 and no traceback. On POSIX a
    SIGKILL to the parent is equally untrappable; the guard is in the handler
    because `get_pids` is supplied by the caller and the handler is the only
    place that knows who "we" are.
    """
    import os

    signals_module.uninstall()
    cancel = threading.Event()
    killed: list[int] = []
    real_kill = os.kill
    os.kill = lambda pid, sig: killed.append(pid)  # type: ignore[assignment]
    try:
        install(lambda: [os.getpid(), 0, -1, 4242], cancel)
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
    finally:
        os.kill = real_kill  # type: ignore[assignment]
        uninstall()
    assert cancel.is_set(), "the cancel event must still be set"
    assert os.getpid() not in killed, "the handler signalled its own process"
    assert 0 not in killed and -1 not in killed, "a process group was signalled"
    assert 4242 in killed, "the real worker pid was not signalled"
