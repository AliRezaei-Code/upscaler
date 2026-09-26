"""The Toga view, driven through the dummy backend.

`toga-dummy` records what the platform layer was told, so these tests exercise
the real widget tree, the real handler wiring and the real event dispatch
without a display. Toga's dialogs return the dummy values, which is why the
browse paths are asserted by their *result* rather than by the dialog itself.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("toga_dummy")

# Toga picks its backend through this variable, and caches the choice, so it has
# to be set before the first `import toga` in the process.
os.environ["TOGA_BACKEND"] = "toga_dummy"

import toga

from core.app import UpscalerApp
from core.config import Device
from core.errors import ConfigError
from core.events import PipelineEvent
from ui_toga import view as view_module
from ui_toga.view import UpscaleView

GPU = Device(
    index=0,
    name="Tesla P40",
    vendor="nvidia",
    backend="onnx:cuda",
    total_memory_bytes=0,
    pci_bus_id="0:01:00.0",
    compute_capability=None,
    usable=True,
    unusable_reason=None,
)
GPU_SECOND = Device(**{**GPU.__dict__, "index": 1, "pci_bus_id": "0:05:00.0"})
CPU = Device(
    index=0,
    name="x86_64",
    vendor="cpu",
    backend="cpu",
    total_memory_bytes=0,
    pci_bus_id=None,
    compute_capability=None,
    usable=True,
    unusable_reason=None,
)
#: The exact wording `core.devices` produces for an AMD GPU with no ROCm.
AMD_REASON = "AMD GPU acceleration on Linux requires ROCm 7.x; see the Runtimes tab"

BROKEN = Device(
    index=1,
    name="Radeon",
    vendor="amd",
    backend="onnx:migraphx",
    total_memory_bytes=0,
    pci_bus_id="0:02:00.0",
    compute_capability=None,
    usable=False,
    unusable_reason=AMD_REASON,
)


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
    """A real Toga app on the dummy platform, built through the entry point.

    `toga.App.__init__` runs `startup()` itself, so constructing the app is
    enough — there is no separate start call to make.
    """
    from ui_toga.app import Upscaler

    app = Upscaler()
    yield app
    app.shutdown()


@pytest.fixture
def view(toga_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(view_module, "models_dir", lambda: tmp_path / "models")
    widget = UpscaleView(UpscalerApp(), toga_app)
    yield widget


def test_the_view_builds_with_four_pages(view) -> None:
    # Toga wraps each page as an OptionItem; the label is the tab's text.
    labels = [item.text for item in view.widget.content]
    assert labels == ["Upscale", "Models", "Community", "Runtimes"]


def test_the_catalogue_is_listed(view) -> None:
    assert len(view.model_selection.items) == 15  # 14 models plus "Browse local…"
    first = view.model_selection.items[0].value
    assert first.startswith("catalogue:realesrgan-x4plus|")
    assert "4x" in first
    assert len(view.models_selection.items) == 14


def test_the_output_follows_the_source(view, tmp_path: Path) -> None:
    view.set_source(tmp_path / "holiday.mkv")
    assert view.output_input.value == str(tmp_path / "holiday.upscaled.mkv")


def test_an_edited_output_is_left_alone(view, tmp_path: Path) -> None:
    view.set_source(tmp_path / "a.mkv")
    view.output_input.value = str(tmp_path / "chosen.mp4")
    view._on_output(None)
    view.set_source(tmp_path / "b.mkv")
    assert view.output_input.value == str(tmp_path / "chosen.mp4")


def test_each_device_gets_a_switch(view) -> None:
    """Toga has no checkable list, so the devices are switches in a box."""
    view.set_devices([GPU, CPU, BROKEN])
    labels = [
        child.text
        for child in view.device_box.children
        if isinstance(child, toga.Switch)
    ]
    assert labels == [
        "GPU 0 — Tesla P40 — VRAM unknown — onnx:cuda",
        "CPU — x86_64",
        "GPU 1 — Radeon — VRAM unknown — onnx:migraphx",
    ]


def test_an_unusable_device_cannot_be_switched_on(view) -> None:
    view.set_devices([GPU, BROKEN])
    switches = [c for c in view.device_box.children if isinstance(c, toga.Switch)]
    assert switches[1].enabled is False
    assert [device.name for device in view.selected_devices()] == ["Tesla P40"]


def test_the_reason_for_an_unusable_device_is_shown(view) -> None:
    view.set_devices([BROKEN])
    labels = [
        child.text
        for child in view.device_box.children
        if isinstance(child, toga.Label)
    ]
    assert any("requires ROCm 7.x" in text for text in labels)


def test_toggling_a_switch_names_the_running_devices(view) -> None:
    view.set_devices([GPU, GPU_SECOND])
    view._on_device_toggle(None)
    assert view.device_status.text == (
        "Running on: GPU 0 — Tesla P40, GPU 1 — Tesla P40"
    )
    view._switches[1].value = False
    view._on_device_toggle(None)
    assert view.device_status.text == "Running on: GPU 0 — Tesla P40"


def test_a_config_needs_a_device(view, tmp_path: Path) -> None:
    view.set_source(tmp_path / "in.mp4")
    view.set_devices([])
    with pytest.raises(ConfigError, match="Select at least one GPU"):
        view.build_config()


def test_a_config_carries_what_the_user_typed(view, tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x")
    model = tmp_path / "mine.pth"
    model.write_bytes(b"y" * 2_000_000)
    view.set_source(source)
    view.set_devices([GPU, CPU])
    view.set_local_model(model)
    view.crf.value = 24
    view.tile.value = 256
    view.delete_frames.value = True
    config = view.build_config()

    assert config.input_path == source
    assert config.model_path == model
    assert config.crf == 24
    assert config.tile_size == 256
    assert config.delete_frames_after_encode is True
    assert [device.name for device in config.devices] == ["Tesla P40", "x86_64"]


def test_a_catalogue_model_resolves_into_the_models_directory(
    view, tmp_path: Path
) -> None:
    # `items[0]` is a Row; the value the view reads is the string inside it.
    view.model_selection.value = view.model_selection.items[0].value
    assert view.current_model() is not None
    assert view.model_path() == tmp_path / "models" / view.current_model().filename


def test_start_and_stop_are_never_both_live(view) -> None:
    assert view.start_button.enabled
    assert not view.stop_button.enabled
    view.set_running(True)
    assert not view.start_button.enabled
    assert view.stop_button.enabled


def test_start_without_a_device_says_why(view, tmp_path: Path) -> None:
    view.set_source(tmp_path / "in.mp4")
    view.set_devices([])
    view.start()
    assert view.status.text == "Select at least one GPU"
    assert view.start_button.enabled


def test_a_stage_reaches_the_log_and_sets_the_total(view) -> None:
    view.handle_event(
        PipelineEvent(
            kind="stage",
            task_id="job",
            stage="upscale",
            message="60 frames, 0 already done, 60 remaining",
        )
    )
    assert "[upscale] 60 frames, 0 already done, 60 remaining" in view.log_view.value
    assert view.total_frames == 60


def test_progress_moves_the_bar_and_its_label(view) -> None:
    view.set_devices([GPU])
    view.handle_event(
        PipelineEvent(
            kind="stage",
            task_id="job",
            stage="upscale",
            message="60 frames, 0 already done, 60 remaining",
        )
    )
    for processed in (20, 60):
        view.handle_event(
            PipelineEvent(
                kind="device_progress",
                task_id="job",
                device_ordinal=0,
                processed=processed,
                total=60,
                fps=2.5,
            )
        )
    assert view.progress.value == pytest.approx(1.0)
    assert "60 / 60 frames" in view.progress_label.text
    assert "GPU 0 — Tesla P40: 60" in view.progress_label.text


def test_progress_uses_the_ordinal_not_the_device_index(view) -> None:
    """The CPU is index 0 and so is the first GPU; the ordinal is the row."""
    view.set_devices([GPU, CPU])
    view.handle_event(
        PipelineEvent(
            kind="stage",
            task_id="job",
            stage="upscale",
            message="60 frames, 0 already done, 60 remaining",
        )
    )
    view.handle_event(
        PipelineEvent(
            kind="device_progress",
            task_id="job",
            device_index=0,
            device_ordinal=1,
            processed=17,
            total=60,
            fps=1.5,
        )
    )
    assert "CPU — x86_64: 17" in view.progress_label.text
    assert "Tesla P40" not in view.progress_label.text


def test_a_done_event_finishes_the_bar(view) -> None:
    view.set_running(True)
    view.handle_event(PipelineEvent(kind="done", task_id="job", message="/tmp/out.mp4"))
    assert view.progress.running is False
    assert view.progress.value == 1.0
    assert view.progress_label.text == "done"
    assert view.start_button.enabled
    assert "Done: /tmp/out.mp4" in view.status.text


def test_an_error_reaches_the_log_and_the_status(view) -> None:
    view.handle_event(
        PipelineEvent(kind="error", task_id="job", message="Select at least one GPU")
    )
    assert "Select at least one GPU" in view.log_view.value
    assert view.status.text == "Select at least one GPU"


def test_another_tasks_events_are_ignored(view) -> None:
    before = view.log_view.value
    view.handle_event(
        PipelineEvent(kind="stage", task_id="model:x", stage="download", message="50%")
    )
    assert view.log_view.value == before


def test_a_probe_event_fills_the_frame_rate(view) -> None:
    view.handle_event(
        PipelineEvent(
            kind="stage",
            task_id="probe:/tmp/in.mkv",
            stage="probe",
            message="384x288, 165303 frames at 25.000 fps, audio yes",
        )
    )
    assert view.fps.value == pytest.approx(25.0)


def test_the_log_is_capped(view) -> None:
    for index in range(view_module.LOG_LINES + 100):
        view.handle_event(
            PipelineEvent(kind="log", task_id="job", message=f"line {index}")
        )
    assert view.log_view.value.count("\n") == view_module.LOG_LINES - 1
    assert "line 0\n" not in view.log_view.value
    assert f"line {view_module.LOG_LINES + 99}" in view.log_view.value


def test_cleaning_refuses_while_a_job_runs(
    view, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / ".upscaler-work-in"
    (work / "frames_out").mkdir(parents=True)
    view.set_source(tmp_path / "in.mp4")
    view.set_devices([GPU])
    monkeypatch.setattr(view.app, "is_running", lambda: True)
    view.clean_work_dir(None)
    assert "A job is running" in view.status.text
    assert (work / "frames_out").is_dir()


def test_cleaning_says_what_it_would_delete(view, tmp_path: Path) -> None:
    work = tmp_path / ".upscaler-work-in"
    (work / "frames_out").mkdir(parents=True)
    (work / "frames_out" / "frame_00000000.png").write_bytes(b"x" * 2048)
    view.set_source(tmp_path / "in.mp4")
    view.set_devices([GPU])
    view.clean_work_dir(None)
    assert ".upscaler-work-in" in view.status.text
    assert "PySide6 front-end" in view.status.text
    assert (work / "frames_out" / "frame_00000000.png").is_file(), "nothing was deleted"


def test_cleaning_a_missing_directory_says_so(view, tmp_path: Path) -> None:
    view.set_source(tmp_path / "in.mp4")
    view.clean_work_dir(None)
    assert "does not exist" in view.status.text
