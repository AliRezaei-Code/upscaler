"""The Upscale tab, driven headlessly.

No mouse: every assertion goes through a public method and checks what a user
would see — the text in a label, the state of a button, the cells in the
per-device table. The device probe is monkeypatched because on this machine it
costs the full ten-second timeout, which is the behaviour it is supposed to
have and not something a UI test should pay for.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.app import UpscalerApp
from core.config import Device
from core.errors import ConfigError
from core.events import PipelineEvent

DEVICE_P40 = Device(
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
DEVICE_P40_SECOND = Device(
    index=1,
    name="Tesla P40",
    vendor="nvidia",
    backend="onnx:cuda",
    total_memory_bytes=0,
    pci_bus_id="0:05:00.0",
    compute_capability=None,
    usable=True,
    unusable_reason=None,
)
DEVICE_CPU = Device(
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

DEVICE_BROKEN = Device(
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


@pytest.fixture(scope="session")
def qt_app() -> QApplication:
    existing = QApplication.instance()
    return existing if existing is not None else QApplication([])


@pytest.fixture
def tab(qt_app: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from ui_pyside import upscale_tab as module

    monkeypatch.setattr(module, "probe_all", lambda emit=None: [DEVICE_P40, DEVICE_CPU])
    monkeypatch.setattr(module, "models_dir", lambda: tmp_path / "models")
    widget = module.UpscaleTab(UpscalerApp())
    yield widget
    widget.deleteLater()


def test_the_output_follows_the_source(tab, tmp_path: Path) -> None:
    from ui_pyside.upscale_tab import default_output_path

    source = tmp_path / "Mosaferan.Mahtab.mkv"
    source.write_bytes(b"x")
    tab.set_source(source)
    assert tab.output_edit.text() == str(tmp_path / "Mosaferan.Mahtab.upscaled.mkv")
    assert default_output_path(source).name == "Mosaferan.Mahtab.upscaled.mkv"


def test_an_edited_output_is_left_alone(tab, tmp_path: Path) -> None:
    tab.set_source(tmp_path / "a.mkv")
    tab.output_edit.setText(str(tmp_path / "chosen.mp4"))
    tab._on_output_edited(str(tmp_path / "chosen.mp4"))
    tab.set_source(tmp_path / "b.mkv")
    assert tab.output_edit.text() == str(tmp_path / "chosen.mp4")


def test_the_work_directory_is_derived_from_the_source(tab, tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    tab.set_source(source)
    assert tab._work_dir == tmp_path / ".upscaler-work-clip"
    assert "frames_out is" in tab.delete_frames_check.toolTip()


def test_devices_are_listed_with_their_state(tab) -> None:
    tab.set_devices([DEVICE_P40, DEVICE_BROKEN, DEVICE_CPU])
    assert tab.device_list.count() == 3
    assert (
        tab.device_list.item(0).text() == "GPU 0 — Tesla P40 — VRAM unknown — onnx:cuda"
    )
    assert tab.device_list.item(0).checkState() == Qt.CheckState.Checked
    # An unusable row cannot be ticked, and says why where the user can see it.
    assert tab.device_list.item(1).text().endswith("— unavailable")
    assert tab.device_list.item(1).checkState() == Qt.CheckState.Unchecked
    assert tab.device_list.item(1).toolTip() == DEVICE_BROKEN.unusable_reason
    assert tab.device_list.item(2).text().startswith("CPU — ")


def test_only_ticked_devices_are_selected(tab) -> None:
    tab.set_devices([DEVICE_P40, DEVICE_CPU])
    assert [d.name for d in tab.selected_devices()] == ["Tesla P40", "x86_64"]
    tab.tick_device(1, False)
    assert [d.name for d in tab.selected_devices()] == ["Tesla P40"]


def test_the_device_table_shows_the_selected_devices(tab) -> None:
    """The run bar reports the devices *this run* uses, in selection order."""
    tab.set_devices([DEVICE_P40, DEVICE_CPU, DEVICE_P40_SECOND])
    assert tab.device_table.rowCount() == 3
    assert tab.device_table.item(0, 0).text() == "Tesla P40"
    assert tab.device_table.item(0, 1).text() == "0"
    tab.tick_device(0, False)
    assert tab.device_table.rowCount() == 2
    assert tab.device_table.item(0, 0).text() == "x86_64"


def test_a_config_needs_a_device(tab, tmp_path: Path) -> None:
    tab.set_source(tmp_path / "in.mp4")
    tab.set_devices([])
    with pytest.raises(ConfigError, match="Select at least one GPU"):
        tab.build_config()


def test_a_config_needs_a_model(tab, tmp_path: Path) -> None:
    tab.set_source(tmp_path / "in.mp4")
    tab.set_devices([DEVICE_P40])
    tab.model_combo.setCurrentIndex(-1)
    with pytest.raises(ConfigError, match="Choose a model first"):
        tab.build_config()


def test_a_config_carries_what_the_user_typed(tab, tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x")
    model = tmp_path / "somebody-elses.pth"
    model.write_bytes(b"y" * 2_000_000)
    tab.set_source(source)
    tab.set_devices([DEVICE_P40, DEVICE_CPU])
    tab.set_local_model(model)
    tab.crf_spin.setValue(24)
    tab.tile_spin.setValue(256)
    tab.delete_frames_check.setChecked(True)
    config = tab.build_config()

    assert config.input_path == source
    assert config.model_path == model
    assert config.output_path == tmp_path / "in.upscaled.mp4"
    assert config.crf == 24
    assert config.tile_size == 256
    assert config.delete_frames_after_encode is True
    assert config.work_dir == tmp_path / ".upscaler-work-in"
    assert [d.name for d in config.devices] == ["Tesla P40", "x86_64"]


def test_a_catalogue_model_resolves_into_the_models_directory(
    tab, tmp_path: Path
) -> None:
    tab.model_combo.setCurrentIndex(0)
    entry = tab.current_model()
    assert entry is not None
    assert tab.model_path() == tmp_path / "models" / entry.filename


def test_start_and_stop_are_never_both_live(tab) -> None:
    assert tab.start_button.isEnabled()
    assert not tab.stop_button.isEnabled()
    tab.set_running(True)
    assert not tab.start_button.isEnabled()
    assert tab.stop_button.isEnabled()
    tab.set_running(False)
    assert tab.start_button.isEnabled()
    assert not tab.stop_button.isEnabled()


def test_start_without_a_device_says_why(tab, tmp_path: Path) -> None:
    tab.set_source(tmp_path / "in.mp4")
    tab.set_devices([])
    tab.start()
    assert tab.status_label.text() == "Select at least one GPU"
    assert tab.start_button.isEnabled(), "a failed start disabled Start"


def test_a_job_stage_reaches_the_log(tab) -> None:
    tab.set_devices([DEVICE_P40])
    tab.handle_event(
        PipelineEvent(
            kind="stage",
            task_id="job",
            stage="upscale",
            message="60 frames, 0 already done, 60 remaining",
        )
    )
    assert (
        "[upscale] 60 frames, 0 already done, 60 remaining"
        in tab.log_view.toPlainText()
    )
    assert tab._total_frames == 60
    assert "upscale:" in tab.status_label.text()


def test_device_progress_fills_the_table_and_the_bar(tab) -> None:
    tab.set_devices([DEVICE_P40])
    tab.handle_event(
        PipelineEvent(
            kind="stage",
            task_id="job",
            stage="upscale",
            message="60 frames, 0 already done, 60 remaining",
        )
    )
    seen: list[int] = []
    for processed in (20, 40, 60):
        tab.handle_event(
            PipelineEvent(
                kind="device_progress",
                task_id="job",
                device_ordinal=0,
                processed=processed,
                total=60,
                fps=12.5,
            )
        )
        seen.append(tab.progress_bar.value())
    assert tab.device_table.item(0, 1).text() == "60"
    assert tab.device_table.item(0, 2).text() == "12.5"
    # Every step must move: QProgressBar ignores a value above its maximum, so
    # a bar whose maximum was the frame count froze at 33% and never advanced.
    assert seen == [33, 66, 100]
    assert "60 / 60 frames" in tab.progress_bar.format()


def test_another_tasks_progress_does_not_move_this_tab(tab) -> None:
    tab.set_devices([DEVICE_P40])
    tab.handle_event(
        PipelineEvent(
            kind="stage",
            task_id="job",
            stage="upscale",
            message="60 frames, 0 already done, 60 remaining",
        )
    )
    tab.handle_event(
        PipelineEvent(
            kind="device_progress",
            task_id="model:realesrgan-x4plus",
            device_ordinal=0,
            processed=5_000_000,
            total=67_000_000,
        )
    )
    assert tab.device_table.item(0, 1).text() == "0"
    assert tab.progress_bar.value() == 0


def test_a_done_event_reenables_start(tab) -> None:
    tab.set_running(True)
    tab.handle_event(PipelineEvent(kind="done", task_id="job", message="/tmp/out.mp4"))
    assert tab.start_button.isEnabled()
    assert not tab.stop_button.isEnabled()
    assert tab.progress_bar.value() == 100
    assert tab.status_label.text() == "Done: /tmp/out.mp4"


def test_an_error_event_reaches_the_log_and_the_status(tab) -> None:
    tab.handle_event(
        PipelineEvent(kind="error", task_id="job", message="Need ~9750 GB for frames")
    )
    assert "Need ~9750 GB for frames" in tab.log_view.toPlainText()
    assert tab.status_label.text() == "Need ~9750 GB for frames"


def test_a_probe_event_fills_the_frame_rate(tab) -> None:
    tab.handle_event(
        PipelineEvent(
            kind="stage",
            task_id="probe:/tmp/in.mkv",
            stage="probe",
            message="384x288, 165303 frames at 25.000 fps, audio yes",
        )
    )
    assert tab.fps_spin.value() == pytest.approx(25.0)
    assert "165303 frames" in tab.status_label.text()


def test_the_log_is_capped(tab) -> None:
    from ui_pyside.upscale_tab import LOG_LINES

    for index in range(LOG_LINES + 250):
        tab.handle_event(
            PipelineEvent(kind="log", task_id="job", message=f"line {index}")
        )
    assert tab.log_view.blockCount() <= LOG_LINES
    assert "line 0\n" not in tab.log_view.toPlainText()
    assert f"line {LOG_LINES + 249}" in tab.log_view.toPlainText()


def test_the_tab_works_without_an_event_pump(tab, tmp_path: Path) -> None:
    """The headless smoke builds the tab with nothing draining the queue."""
    tab.set_devices([DEVICE_P40])
    tab.set_source(tmp_path / "in.mp4")
    time.sleep(0.2)
    assert tab.device_list.count() == 1


def test_progress_lands_on_the_device_that_reported_it(tab) -> None:
    """The CPU is index 0, and so is the first Tesla P40.

    A row keyed on `Device.index` draws the CPU's progress on the first GPU's
    row, which is what a screenshot of a real run caught: the events arrive
    with the worker's ordinal *and* the device's own index, and the ordinal is
    the one that identifies the row.
    """
    tab.set_devices([DEVICE_P40, DEVICE_CPU])
    assert [tab.device_table.item(row, 0).text() for row in range(2)] == [
        "Tesla P40",
        "x86_64",
    ]
    tab.handle_event(
        PipelineEvent(
            kind="stage",
            task_id="job",
            stage="upscale",
            message="60 frames, 0 already done, 60 remaining",
        )
    )
    tab.handle_event(
        PipelineEvent(
            kind="device_progress",
            task_id="job",
            device_index=0,  # the CPU's own index, the same as the first GPU's
            device_ordinal=1,
            processed=17,
            total=60,
            fps=3.5,
        )
    )
    assert tab.device_table.item(0, 1).text() == "0"
    assert tab.device_table.item(1, 1).text() == "17"
    assert tab.device_table.item(1, 2).text() == "3.5"
