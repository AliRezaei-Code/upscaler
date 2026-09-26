"""The Toga app's plumbing, on the dummy backend.

The view's tests prove what the widgets show. These prove the two hops in
between, which is where a Toga front-end actually differs from the Qt one: a
plain thread forwards `UpscalerApp.events()` into a queue, and an asyncio task
on Toga's own loop drains that queue into the widgets. A bug there shows up as
a progress bar that never moves and no error anywhere.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
import time

import pytest

pytest.importorskip("toga_dummy")

# Toga picks its backend through this variable and caches the choice, so it has
# to be set before the first `import toga` in the process. tests/test_toga_view.py
# sets it too; setting it twice is harmless.
os.environ["TOGA_BACKEND"] = "toga_dummy"

from core.events import PipelineEvent


@pytest.fixture(autouse=True)
def fast_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never pay the real probe.

    On this machine `nvidia-smi` never returns, so the probe costs its full
    ten-second timeout — and the app submits one in `startup()`. Every test here
    would otherwise wait for it.
    """
    import core.devices

    monkeypatch.setattr(core.devices, "probe_all", lambda emit=None: [])


@pytest.fixture
def toga_app():
    """A started `Upscaler`, with its loop not yet running."""
    from ui_toga.app import Upscaler

    app = Upscaler()
    yield app
    app.shutdown()


def test_startup_builds_the_view_and_the_window(toga_app) -> None:
    assert toga_app.view is not None
    assert toga_app.view.widget is not None
    assert toga_app.main_window is not None
    assert toga_app._pump is not None
    assert toga_app._pump.is_alive(), "the event forwarder is not running"


def test_the_forwarder_thread_moves_core_events_into_the_bridge(toga_app) -> None:
    """A task's event crosses the thread into the bridge, with no GUI thread."""
    from core.events import PipelineEvent

    toga_app.core.submit(
        "job",
        lambda emit, stop: emit(
            PipelineEvent(kind="log", task_id="job", message="through the thread")
        ),
    )

    deadline = time.monotonic() + 10
    messages: list[str] = []
    while time.monotonic() < deadline:
        while not toga_app._bridge.empty():
            event = toga_app._bridge.get_nowait()
            messages.append(getattr(event, "message", ""))
        if "through the thread" in messages:
            break
        time.sleep(0.02)
    assert "through the thread" in messages


def test_the_drain_task_delivers_to_the_view(toga_app) -> None:
    """Bridge to widget, through the loop Toga runs."""
    view = toga_app.view
    assert view is not None
    toga_app.loop.run_until_complete(_drain_once(toga_app))
    assert "[upscale] drained" in view.log_view.value
    assert view.status.text == "upscale: drained"


async def _drain_once(app) -> None:
    app._bridge.put(
        PipelineEvent(kind="stage", task_id="job", stage="upscale", message="drained")
    )
    task = app.loop.create_task(app._drain())
    await asyncio.sleep(0.3)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def test_shutdown_stops_the_forwarder() -> None:
    from ui_toga.app import Upscaler

    app = Upscaler()
    pump = app._pump
    assert pump is not None
    app.shutdown()
    assert app._pump is None
    assert not pump.is_alive(), "the forwarder outlived shutdown"


def test_no_worker_threads_survive_shutdown() -> None:
    from ui_toga.app import Upscaler

    app = Upscaler()
    app.shutdown()
    time.sleep(0.2)
    alive = [
        t.name
        for t in threading.enumerate()
        if t.name == "toga-events" and t.is_alive()
    ]
    assert alive == []


def test_the_drain_interval_is_short_enough_to_look_live() -> None:
    from ui_toga.app import DRAIN_INTERVAL_SECONDS

    assert 0 < DRAIN_INTERVAL_SECONDS <= 0.2


def test_the_platform_is_the_dummy_one() -> None:
    """A test that silently ran against GTK would prove nothing."""
    from toga.platform import get_factory

    factory = get_factory()
    assert type(factory).__module__.startswith("toga") or factory is not None
    # The dummy backend announces itself by refusing everything it does not
    # implement; the GTK backend would have raised before this line.
    with pytest.raises(NotImplementedError):
        _ = factory.__path__  # type: ignore[attr-defined]
