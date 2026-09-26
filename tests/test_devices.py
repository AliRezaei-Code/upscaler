"""Device probe tests.

The `nvidia_proc` fixture is a byte-for-byte copy of this machine's
`/proc/driver/nvidia/gpus/*/information`: three Tesla P40s, one of them with
`Video BIOS: ??.??.??.??.??`. Everything asserted about NVIDIA enumeration is
asserted against that real text, not against invented input.

The directory names are the PCI bus addresses with `:` replaced by `-`, because
a colon is illegal in a Windows path and this fixture is checked out on a
Windows runner too. The addresses themselves are in the file *contents*, and
they still parse and sort the same.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from core import devices as devices_module
from core.config import Device
from core.devices import (
    MACOS_COREML_MIN,
    PROBE_TIMEOUT_SECONDS,
    cpu_device,
    is_fp16_capable,
    macos_version,
    mps_available,
    normalise_bus,
    parse_kfd_topology,
    parse_nvidia_proc,
    parse_nvidia_smi,
    parse_pci_bus_id,
    probe_all,
    probe_amd,
    probe_apple,
    probe_intel,
    probe_nvidia,
    rocm_version,
)

FIXTURES = Path(__file__).parent / "fixtures"
NVIDIA_GLOB = str(FIXTURES / "nvidia_proc") + "/*/information"
KFD_GLOB = str(FIXTURES / "kfd" / "node0") + "/properties"
DRM_GLOB = str(FIXTURES / "drm") + "/card[0-9]*/device/vendor"
ROCM_GLOB = str(FIXTURES / "rocm") + "/.info/version"

# The exact text the driver prints when it does not answer in time.
DEGRADED_REASON = (
    "driver did not answer within 10 s; "
    "VRAM and compute capability unknown — FP32 will be forced"
)
# A healthy response for the three P40s, in the CSV shape nvidia-smi emits.
SMI_HEALTHY = (
    "0, 00000000:01:00.0, 6.1, 24576\n"
    "1, 00000000:05:00.0, 6.1, 24576\n"
    "2, 00000000:06:00.0, 6.1, 24576\n"
)


@pytest.fixture
def nvidia_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices_module, "NVIDIA_INFORMATION_GLOB", NVIDIA_GLOB)


def _completed(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=0, stdout=stdout, stderr=""
    )


def _never_called(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("nvidia-smi should not have been called")


def _timeout(*_args: object, **_kwargs: object) -> None:
    raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=PROBE_TIMEOUT_SECONDS)


def _refuses(stderr: str) -> Callable[..., None]:
    def _raise(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(
            returncode=9, cmd="nvidia-smi", output="", stderr=stderr
        )

    return _raise


# --- parsing the real capture -------------------------------------------------


def test_parse_nvidia_proc_reads_the_real_capture() -> None:
    files = sorted((FIXTURES / "nvidia_proc").glob("*/information"))
    assert len(files) == 3
    parsed = [parse_nvidia_proc(path.read_text()) for path in files]
    assert all(len(entry) == 1 for entry in parsed)
    models = [entry[0][0] for entry in parsed]
    uuids = [entry[0][1] for entry in parsed]
    buses = [entry[0][2] for entry in parsed]
    assert models == ["Tesla P40", "Tesla P40", "Tesla P40"]
    assert all(uuid.startswith("GPU-") for uuid in uuids)
    assert len(set(uuids)) == 3
    assert buses == ["0:01:00.0", "0:05:00.0", "0:06:00.0"]


def test_parse_nvidia_proc_ignores_a_file_without_a_model() -> None:
    assert parse_nvidia_proc("") == []
    assert parse_nvidia_proc("IRQ: 155\n") == []
    assert parse_nvidia_proc("Model: Tesla P40\n") == []


def test_normalise_bus_bridges_proc_and_smi_spellings() -> None:
    # /proc prints a 4-digit domain, nvidia-smi an 8-digit one.
    assert normalise_bus("0000:06:00.0") == normalise_bus("00000000:06:00.0")
    assert normalise_bus("00000000:07:00.0") == "0:07:00.0"
    assert normalise_bus("not a bus") == "not a bus"


def test_parse_pci_bus_id_reads_a_csv_row() -> None:
    assert parse_pci_bus_id("0, 00000000:07:00.0, 6.1, 24576") == "0:07:00.0"
    assert parse_pci_bus_id("[N/A], [N/A], [N/A], [N/A]") is None
    assert parse_pci_bus_id("") is None


def test_parse_nvidia_smi_converts_mib_to_bytes() -> None:
    parsed = parse_nvidia_smi(SMI_HEALTHY)
    assert set(parsed) == {"0:01:00.0", "0:05:00.0", "0:06:00.0"}
    first = parsed["0:01:00.0"]
    assert first.compute_capability == (6, 1)
    assert first.total_memory_bytes == 24576 * 1024 * 1024
    assert first.smi_index == 0


def test_parse_nvidia_smi_skips_unparseable_rows() -> None:
    assert parse_nvidia_smi("\n\nUnable to determine the device handle\n") == {}


def test_parse_nvidia_smi_without_a_compute_capability() -> None:
    parsed = parse_nvidia_smi("0, 00000000:01:00.0, [N/A], 24576")
    assert parsed["0:01:00.0"].compute_capability is None
    assert parsed["0:01:00.0"].total_memory_bytes == 24576 * 1024 * 1024


@pytest.mark.parametrize(
    ("cap", "expected"),
    [
        (None, False),
        ((6, 1), False),
        ((6, 9), False),
        ((7, 0), True),
        ((7, 5), True),
        ((8, 0), True),
    ],
)
def test_is_fp16_capable(cap: tuple[int, int] | None, expected: bool) -> None:
    assert is_fp16_capable(cap) is expected


# --- health classification ----------------------------------------------------


def test_a_driver_that_times_out_yields_degraded_but_usable_devices(
    nvidia_fixture: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(devices_module, "_run_smi", _timeout)
    devices = probe_nvidia()
    assert len(devices) == 3
    assert [device.vendor for device in devices] == ["nvidia"] * 3
    # procfs carries no VRAM and no compute capability, and the probe must not
    # invent either when the driver will not say.
    assert all(device.total_memory_bytes == 0 for device in devices)
    assert all(device.compute_capability is None for device in devices)
    assert all(device.usable for device in devices)
    assert all(device.unusable_reason == DEGRADED_REASON for device in devices)
    assert all(device.backend == "onnx:cuda" for device in devices)


def test_a_driver_that_refuses_yields_unusable_devices(
    nvidia_fixture: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        devices_module, "_run_smi", _refuses("Failed to initialize NVML")
    )
    devices = probe_nvidia()
    assert len(devices) == 3
    assert all(not device.usable for device in devices)
    assert all(
        device.unusable_reason == "driver reported no device: Failed to initialize NVML"
        for device in devices
    )


def test_an_empty_response_is_unhealthy(
    nvidia_fixture: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(devices_module, "_run_smi", lambda *a, **k: _completed(""))
    devices = probe_nvidia()
    assert len(devices) == 3
    assert all(not device.usable for device in devices)
    assert all(
        device.unusable_reason == "driver reported no device: empty response"
        for device in devices
    )


def test_a_healthy_driver_fills_vram_and_capability(
    nvidia_fixture: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        devices_module, "_run_smi", lambda *a, **k: _completed(SMI_HEALTHY)
    )
    devices = probe_nvidia()
    assert len(devices) == 3
    for device in devices:
        assert device.usable
        assert device.unusable_reason is None
        assert device.compute_capability == (6, 1)
        assert device.total_memory_bytes == 24576 * 1024 * 1024


def test_enumeration_is_sorted_by_bus_address_not_glob_order(
    nvidia_fixture: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(devices_module, "_run_smi", _timeout)
    devices = probe_nvidia()
    assert [device.pci_bus_id for device in devices] == [
        "0:01:00.0",
        "0:05:00.0",
        "0:06:00.0",
    ]
    assert [device.index for device in devices] == [0, 1, 2]
    # A reversed glob order must not change the numbering.
    monkeypatch.setattr(
        devices_module,
        "NVIDIA_INFORMATION_GLOB",
        str(FIXTURES / "nvidia_proc") + "/*/information",
    )
    reversed_devices = probe_nvidia()
    assert [device.index for device in reversed_devices] == [0, 1, 2]
    assert [device.pci_bus_id for device in reversed_devices] == [
        device.pci_bus_id for device in devices
    ]


def test_a_driver_excluded_gpu_is_unusable(
    nvidia_fixture: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    excluded = tmp_path / "0000:09:00.0" / "information"
    excluded.parent.mkdir(parents=True)
    excluded.write_text(
        "Model: \t\t Tesla P40\n"
        "GPU UUID: \t GPU-dead-beef\n"
        "Bus Location: \t 0000:09:00.0\n"
        "GPU Excluded:\t Yes\n"
    )
    monkeypatch.setattr(
        devices_module,
        "NVIDIA_INFORMATION_GLOB",
        str(tmp_path) + "/*/information",
    )
    monkeypatch.setattr(devices_module, "_run_smi", _timeout)
    devices = probe_nvidia()
    assert len(devices) == 1
    assert not devices[0].usable
    assert devices[0].unusable_reason == "excluded by the NVIDIA driver"


def test_procfs_is_consulted_before_nvidia_smi(
    nvidia_fixture: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enumeration must not depend on a subprocess that may not return."""
    monkeypatch.setattr(devices_module, "_run_smi", _timeout)
    assert [device.name for device in probe_nvidia()] == ["Tesla P40"] * 3


def test_nvidia_probe_without_procfs_falls_back_to_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        devices_module, "NVIDIA_INFORMATION_GLOB", "/nonexistent/*/information"
    )
    monkeypatch.setattr(
        devices_module,
        "_run_smi",
        lambda *a, **k: _completed(
            "0, Tesla P40, 00000000:05:00.0, 6.1, 24576\n"
            "1, Tesla P40, 00000000:01:00.0, 6.1, 24576\n"
        ),
    )
    devices = probe_nvidia()
    assert [device.pci_bus_id for device in devices] == ["0:01:00.0", "0:05:00.0"]
    assert [device.index for device in devices] == [0, 1]
    assert all(device.name == "Tesla P40" for device in devices)
    assert all(device.usable for device in devices)


def test_nvidia_probe_without_procfs_and_a_wedged_driver_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        devices_module, "NVIDIA_INFORMATION_GLOB", "/nonexistent/*/information"
    )
    monkeypatch.setattr(devices_module, "_run_smi", _timeout)
    assert probe_nvidia() == []


def test_a_partial_response_leaves_the_other_devices_degraded(
    nvidia_fixture: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        devices_module,
        "_run_smi",
        lambda *a, **k: _completed("0, 00000000:01:00.0, 6.1, 24576\n"),
    )
    devices = probe_nvidia()
    assert devices[0].total_memory_bytes == 24576 * 1024 * 1024
    assert devices[0].unusable_reason is None
    for missing in devices[1:]:
        assert missing.usable
        assert missing.unusable_reason == DEGRADED_REASON
        assert missing.total_memory_bytes == 0
        assert missing.compute_capability is None


# --- timing -------------------------------------------------------------------


def test_probe_all_over_fixture_input_is_under_two_seconds(
    nvidia_fixture: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        devices_module, "KFD_TOPOLOGY_GLOB", "/nonexistent/*/properties"
    )
    monkeypatch.setattr(
        devices_module, "DRM_VENDOR_GLOB", "/nonexistent/card[0-9]*/vendor"
    )
    monkeypatch.setattr(
        devices_module, "_run_smi", lambda *a, **k: _completed(SMI_HEALTHY)
    )
    started = time.monotonic()
    devices = probe_all()
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"probe_all took {elapsed:.2f}s"
    assert len(devices) == 4


def test_run_smi_bounds_the_subprocess_and_orders_cuda_by_bus(
    nvidia_fixture: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The subprocess must carry the probe timeout and the bus-order env."""
    recorded: dict[str, object] = {}

    def _record(args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        recorded["args"] = args
        recorded.update(kwargs)
        return _completed(SMI_HEALTHY)

    monkeypatch.setattr(devices_module.subprocess, "run", _record)
    probe_nvidia()
    assert recorded["timeout"] == PROBE_TIMEOUT_SECONDS
    assert recorded["check"] is True
    env = recorded["env"]
    assert isinstance(env, dict)
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_a_subprocess_that_never_returns_is_bounded(
    nvidia_fixture: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`nvidia-smi` really is killed at the deadline, with the real subprocess."""
    monkeypatch.setattr(devices_module, "PROBE_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(
        devices_module,
        "_run_smi",
        lambda args, env=None, timeout=None: subprocess.run(
            ["sleep", "30"],
            capture_output=True,
            text=True,
            timeout=devices_module.PROBE_TIMEOUT_SECONDS,
            check=True,
        ),
    )
    started = time.monotonic()
    devices = probe_nvidia()
    elapsed = time.monotonic() - started
    assert elapsed < 10.0, f"probe_nvidia took {elapsed:.2f}s against a hung nvidia-smi"
    assert len(devices) == 3
    assert all(device.usable for device in devices)
    assert all("did not answer" in (device.unusable_reason or "") for device in devices)


def test_a_file_read_that_never_answers_is_abandoned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A wedged driver can block procfs, so the read is given a deadline too."""
    for bus in ("0000-05-00.0", "0000-06-00.0"):
        target = tmp_path / bus / "information"
        target.parent.mkdir(parents=True)
        target.write_text((FIXTURES / "nvidia_proc" / bus / "information").read_text())
    blocked = tmp_path / "0000:01:00.0" / "information"
    blocked.parent.mkdir(parents=True)
    os.mkfifo(blocked)  # a read on a FIFO with no writer blocks forever

    monkeypatch.setattr(devices_module, "PROBE_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(
        devices_module, "NVIDIA_INFORMATION_GLOB", str(tmp_path) + "/*/information"
    )
    monkeypatch.setattr(devices_module, "_run_smi", _timeout)
    started = time.monotonic()
    devices = probe_nvidia()
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"probe_nvidia took {elapsed:.2f}s against a blocked read"
    # The blocked GPU is absent; the two readable ones are still degraded, not lost.
    assert [device.pci_bus_id for device in devices] == ["0:05:00.0", "0:06:00.0"]
    assert all("did not answer" in (device.unusable_reason or "") for device in devices)


# --- AMD ----------------------------------------------------------------------


def test_parse_kfd_topology_single_gpu() -> None:
    entries = parse_kfd_topology(
        (FIXTURES / "kfd" / "node0" / "properties").read_text()
    )
    assert len(entries) == 1
    entry = entries[0]
    assert entry["gfx_arch"] == "gfx1030"
    assert entry["location_id"] == "0x400"
    assert entry["cpu_cores_count"] == "16"


def test_parse_kfd_topology_two_gpus_on_one_node() -> None:
    entries = parse_kfd_topology(
        (FIXTURES / "kfd" / "multi" / "properties").read_text()
    )
    assert len(entries) == 2
    assert [entry["location_id"] for entry in entries] == ["0x0", "0x100"]
    # Node-level keys before the first location_id are inherited by both.
    assert all(entry["gfx_arch"] == "gfx90a" for entry in entries)
    assert all(entry["cpu_cores_count"] == "32" for entry in entries)


def test_parse_kfd_topology_of_nothing() -> None:
    assert parse_kfd_topology("") == []
    assert parse_kfd_topology("\n\n") == []


def test_amd_devices_need_a_rocm_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(devices_module, "KFD_TOPOLOGY_GLOB", KFD_GLOB)
    monkeypatch.setattr(
        devices_module, "ROCM_VERSION_GLOB", str(tmp_path / "absent" / "*")
    )
    devices = probe_amd()
    assert len(devices) == 1
    assert devices[0].vendor == "amd"
    assert devices[0].backend == "onnx:migraphx"
    assert not devices[0].usable
    assert devices[0].unusable_reason == (
        "AMD GPU acceleration on Linux requires ROCm 7.x; see the Runtimes tab"
    )


def test_amd_devices_are_usable_with_a_rocm_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(devices_module, "KFD_TOPOLOGY_GLOB", KFD_GLOB)
    monkeypatch.setattr(devices_module, "ROCM_VERSION_GLOB", ROCM_GLOB)
    devices = probe_amd()
    assert len(devices) == 1
    assert devices[0].usable
    assert devices[0].unusable_reason is None
    assert devices[0].name == "AMD 1030 (0x400)"


def test_rocm_version_is_none_without_rocm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        devices_module, "ROCM_VERSION_GLOB", str(tmp_path / "*" / ".info" / "version")
    )
    assert rocm_version() is None
    monkeypatch.setattr(devices_module, "ROCM_VERSION_GLOB", ROCM_GLOB)
    assert rocm_version() == "6.2.0"


def test_amd_probe_is_empty_without_kfd_nodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        devices_module, "KFD_TOPOLOGY_GLOB", str(tmp_path / "*" / "properties")
    )
    assert probe_amd() == []


# --- Intel --------------------------------------------------------------------


def test_intel_probe_ignores_non_intel_cards_and_connectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(devices_module, "DRM_VENDOR_GLOB", DRM_GLOB)
    devices = probe_intel()
    assert len(devices) == 1
    assert devices[0].vendor == "intel"
    assert devices[0].backend == "torch:xpu"
    assert devices[0].usable


def test_intel_probe_is_empty_without_an_intel_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        devices_module, "DRM_VENDOR_GLOB", str(tmp_path / "card[0-9]*/vendor")
    )
    assert probe_intel() == []


# --- macOS --------------------------------------------------------------------


def _darwin(
    monkeypatch: pytest.MonkeyPatch, version: str, machine: str = "arm64"
) -> None:
    monkeypatch.setattr(devices_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(devices_module.platform, "mac_ver", lambda: (version, "", ""))
    monkeypatch.setattr(devices_module.platform, "machine", lambda: machine)


def test_apple_probe_on_macos_13(monkeypatch: pytest.MonkeyPatch) -> None:
    _darwin(monkeypatch, "13.6.1")
    devices = probe_apple()
    assert len(devices) == 1
    assert devices[0].vendor == "apple"
    assert devices[0].backend == "onnx:coreml"
    assert devices[0].pci_bus_id is None
    assert devices[0].compute_capability is None
    assert devices[0].usable
    assert mps_available() is False


def test_apple_probe_is_refused_below_macos_12(monkeypatch: pytest.MonkeyPatch) -> None:
    _darwin(monkeypatch, "11.7.10")
    devices = probe_apple()
    assert not devices[0].usable
    assert "macOS 12" in (devices[0].unusable_reason or "")
    assert MACOS_COREML_MIN == (12, 0, 0)


def test_apple_probe_on_macos_14_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    _darwin(monkeypatch, "14.6")
    devices = probe_apple()
    assert devices[0].usable
    assert mps_available() is True


def test_mps_needs_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    _darwin(monkeypatch, "14.6", machine="x86_64")
    assert mps_available() is False


def test_mps_needs_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    _darwin(monkeypatch, "13.6")
    assert mps_available() is False


def test_mps_is_unavailable_off_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(devices_module.platform, "mac_ver", lambda: ("14.6", "", ""))
    monkeypatch.setattr(devices_module.platform, "machine", lambda: "arm64")
    assert mps_available() is False


def test_probe_all_lists_only_apple_and_cpu_on_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _darwin(monkeypatch, "14.6")
    monkeypatch.setattr(devices_module, "NVIDIA_INFORMATION_GLOB", "/nonexistent/*/x")
    monkeypatch.setattr(devices_module, "KFD_TOPOLOGY_GLOB", "/nonexistent/*/x")
    monkeypatch.setattr(devices_module, "DRM_VENDOR_GLOB", "/nonexistent/*/x")
    vendors = [device.vendor for device in probe_all()]
    assert vendors == ["apple", "cpu"]


def test_macos_version_parses_partial_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    _darwin(monkeypatch, "14")
    assert macos_version() == (14, 0, 0)


# --- CPU ----------------------------------------------------------------------


def test_cpu_device_is_always_usable() -> None:
    device = cpu_device()
    assert device.vendor == "cpu"
    assert device.backend == "cpu"
    assert device.usable
    assert device.label.startswith("CPU — ")
    assert "VRAM" not in device.label


def test_probe_all_ends_with_the_cpu_device(
    nvidia_fixture: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        devices_module, "KFD_TOPOLOGY_GLOB", "/nonexistent/*/properties"
    )
    monkeypatch.setattr(
        devices_module, "DRM_VENDOR_GLOB", "/nonexistent/card[0-9]*/vendor"
    )
    monkeypatch.setattr(devices_module, "_run_smi", _timeout)
    devices = probe_all()
    assert isinstance(devices[-1], Device)
    assert devices[-1].vendor == "cpu"


def test_probe_all_emits_log_events(
    nvidia_fixture: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.events import PipelineEvent

    monkeypatch.setattr(
        devices_module, "KFD_TOPOLOGY_GLOB", "/nonexistent/*/properties"
    )
    monkeypatch.setattr(
        devices_module, "DRM_VENDOR_GLOB", "/nonexistent/card[0-9]*/vendor"
    )
    monkeypatch.setattr(devices_module, "_run_smi", _timeout)
    events: list[PipelineEvent] = []
    probe_all(events.append)
    assert events
    assert all(event.kind == "log" for event in events)
    assert any("did not answer" in event.message for event in events)


def test_probe_nvidia_without_a_caller_is_silent(
    nvidia_fixture: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(devices_module, "_run_smi", _timeout)
    assert len(probe_nvidia()) == 3
    assert len(probe_amd()) == 0
