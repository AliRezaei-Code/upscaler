"""Device enumeration that cannot hang and cannot lie.

Two properties matter more than completeness here:

1. **It must not block.** On a machine whose driver has wedged, `nvidia-smi`
   does not return — it was measured at over 90 seconds on the reference
   machine, and `torch.cuda.is_available()` took longer than 120. Every
   subprocess runs under `PROBE_TIMEOUT_SECONDS`, and every procfs/sysfs read
   shares one deadline, because a wedged driver can block a
   `/proc/driver/nvidia` read just as happily as it blocks `nvidia-smi`.
2. **It must not claim a GPU works when it does not.** A GPU whose driver
   never answers is reported `DEGRADED` — present, usable, but with unknown
   VRAM and forced FP32 — and a GPU the driver refuses outright is reported
   `UNHEALTHY` and refused by `JobConfig.validate`. Reporting either as healthy
   would start a 55,000-frame-per-device job that can never finish.

This module is the single enumerator. No backend gets a `probe()` of its own,
because a second implementation is how the device list the UI shows and the
list `select_backend` receives drift apart.
"""

from __future__ import annotations

import glob
import os
import platform
import re
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

from .config import Device
from .events import PipelineEvent

PROBE_TIMEOUT_SECONDS = 10.0
HEALTHY = "healthy"
DEGRADED = "degraded"
UNHEALTHY = "unhealthy"

# Enumeration sources. Module-level so a test can point them at a fixture.
NVIDIA_INFORMATION_GLOB = "/proc/driver/nvidia/gpus/*/information"
KFD_TOPOLOGY_GLOB = "/sys/class/kfd/kfd/topology/node*/properties"
DRM_VENDOR_GLOB = "/sys/class/drm/card[0-9]*/device/vendor"
ROCM_VERSION_GLOB = "/opt/rocm*/.info/version"

INTEL_PCI_VENDOR = "0x8086"
MACOS_COREML_MIN = (12, 0, 0)
MACOS_MPS_MIN = (14, 0, 0)

_SMI_QUERY_FIELDS = "index,pci.bus_id,compute_cap,memory.total"
_SMI_QUERY_FIELDS_WITH_NAME = "index,name,pci.bus_id,compute_cap,memory.total"

# `nvidia-smi` prints an 8-digit PCI domain, /proc prints 4. Compare the
# canonical form so the two sources join up.
_BUS_RE = re.compile(
    r"^(?:([0-9a-fA-F]{1,8}):)?([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])$"
)


def _degraded_reason() -> str:
    """The exact wording shown for a driver that did not answer in time."""
    return (
        f"driver did not answer within {PROBE_TIMEOUT_SECONDS:.0f} s; "
        "VRAM and compute capability unknown — FP32 will be forced"
    )


def is_fp16_capable(cap: tuple[int, int] | None) -> bool:
    """Whether FP16 tensor cores are available at this compute capability.

    `None` means the driver never told us, and the answer is False: the
    reference workload measured FP16 as *slower* than FP32 on Pascal, so the
    conservative branch is also the faster one.
    """
    return cap is not None and cap[0] >= 7


def normalise_bus(bus: str) -> str:
    """Canonicalise a PCI address to `domain:bus:device.function`, hex, no padding."""
    match = _BUS_RE.match(bus.strip())
    if match is None:
        return bus.strip().lower()
    domain, bus_no, device_no, function = match.groups()
    prefix = f"{int(domain, 16):x}" if domain else "0"
    return f"{prefix}:{bus_no.lower()}:{device_no.lower()}.{function}"


def parse_pci_bus_id(csv_line: str) -> str | None:
    """Pull the PCI bus address out of one `nvidia-smi --format=csv` row.

    The address is the first field after the index that looks like a bus
    address; `None` when the row has none, which is how a driver that answers
    with an error banner rather than data is detected.
    """
    parts = [part.strip() for part in csv_line.split(",")]
    for part in parts[1:]:
        if _BUS_RE.match(part):
            return normalise_bus(part)
    return None


def parse_nvidia_proc(text: str) -> list[tuple[str, str, str]]:
    """Parse one `/proc/driver/nvidia/gpus/*/information` file.

    Returns `(model, uuid, bus)` triples: normally one, none when the file
    has no model or no bus address. Note what the file does **not** carry —
    no memory size and no compute capability — which is why a probe that reads
    only this file must report `total_memory_bytes == 0` rather than a guess.
    """
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip().lower()] = value.strip()
    model = fields.get("model", "")
    bus = fields.get("bus location", "")
    if not model or not bus:
        return []
    return [(model, fields.get("gpu uuid", ""), normalise_bus(bus))]


def _nvidia_excluded(text: str) -> bool:
    """Whether the driver has excluded this GPU from use."""
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() == "gpu excluded":
            return value.strip().lower() in {"yes", "true", "1"}
    return False


class SmiInfo(NamedTuple):
    """One `nvidia-smi` row, converted to the units the app uses.

    `smi_index` is the driver's own ordinal, which is **not** the app's: the
    app numbers devices by PCI bus order, and the name avoids colliding with
    `tuple.index`.
    """

    smi_index: int
    compute_capability: tuple[int, int] | None
    total_memory_bytes: int


def parse_nvidia_smi(stdout: str) -> dict[str, SmiInfo]:
    """Parse `nvidia-smi` CSV output, keyed by canonical bus address.

    `memory.total` is reported by `nvidia-smi` in **MiB** (24576 for a 24 GB
    card) and is converted to bytes here, once, so no other module has to know.
    Rows that do not parse are skipped rather than guessed at.
    """
    result: dict[str, SmiInfo] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        bus = parse_pci_bus_id(line)
        if bus is None:
            continue
        parts = [part.strip() for part in line.split(",")]
        try:
            index = int(parts[0])
            memory_mib = int(parts[-1])
        except ValueError:
            continue
        capability: tuple[int, int] | None = None
        if len(parts) >= 4:
            raw = parts[-2]
            if "." in raw:
                major, _, minor = raw.partition(".")
                if major.isdigit() and minor.isdigit():
                    capability = (int(major), int(minor))
        result[bus] = SmiInfo(
            smi_index=index,
            compute_capability=capability,
            total_memory_bytes=memory_mib * 1024 * 1024,
        )
    return result


def parse_kfd_topology(text: str) -> list[dict[str, str]]:
    """Parse an AMD KFD `properties` file into one dict per GPU.

    The file is `key value` lines. Keys that appear before the first
    `location_id` describe the node and are inherited by every GPU on it; each
    `location_id` starts a new GPU, because a node with two GPUs emits two of
    them. A file with no `location_id` yields a single dict anyway — the
    caller can still see `gfx_arch`.
    """
    preamble: dict[str, str] = {}
    blocks: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key, _, value = stripped.partition(" ")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if key == "location_id":
            blocks.append(dict(preamble))
        if not value:
            preamble[key] = ""
            continue
        if blocks:
            blocks[-1][key] = value
        else:
            preamble[key] = value
    if blocks:
        return blocks
    if preamble:
        return [preamble]
    return []


def rocm_version() -> str | None:
    """The host ROCm version, or `None` when ROCm is not installed."""
    paths = sorted(glob.glob(ROCM_VERSION_GLOB))
    if not paths:
        return None
    try:
        return Path(paths[0]).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def macos_version() -> tuple[int, int, int]:
    """The running macOS version as `(major, minor, patch)`; zeros elsewhere."""
    release = platform.mac_ver()[0]
    parts: list[int] = []
    for chunk in release.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def mps_available() -> bool:
    """Whether torch's `mps` backend can be used: macOS 14+ on Apple silicon."""
    return (
        platform.system() == "Darwin"
        and macos_version() >= MACOS_MPS_MIN
        and (platform.machine() == "arm64")
    )


def _read_text_files(
    paths: Sequence[str], timeout: float | None = None
) -> dict[str, str | None]:
    """Read several small procfs/sysfs files under **one shared deadline.

    A wedged driver can block a `/proc/driver/nvidia` read indefinitely, so
    every read runs on its own daemon thread and the batch is abandoned at
    `timeout` rather than waited on file by file — one stuck file must not cost
    one timeout per GPU.
    """
    # Resolved here, not as a default, so a test that shortens
    # PROBE_TIMEOUT_SECONDS shortens the real deadline.
    limit = PROBE_TIMEOUT_SECONDS if timeout is None else timeout
    results: dict[str, str | None] = dict.fromkeys(paths)
    threads: list[tuple[str, threading.Thread]] = []

    def _worker(path: str) -> None:
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                results[path] = handle.read()
        except OSError:
            results[path] = None

    for path in paths:
        thread = threading.Thread(target=_worker, args=(path,), daemon=True)
        thread.start()
        threads.append((path, thread))

    deadline = time.monotonic() + limit
    for path, thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            results[path] = None
    return results


def _run_smi(
    args: Sequence[str],
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run `nvidia-smi` under a timeout, raising on a non-zero exit.

    The timeout is resolved from the module global at call time for the same
    reason as `_read_text_files`.
    """
    return subprocess.run(
        ["nvidia-smi", *args],
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_SECONDS if timeout is None else timeout,
        env=env,
        check=True,
    )


def _smi_env() -> dict[str, str]:
    """A child environment whose CUDA ordinals follow PCI bus order.

    Without this, CUDA's device numbering and the app's bus-ordered numbering
    can disagree, and device 1 in the UI would be a different physical GPU
    from device 1 in the process.
    """
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def _call_smi(fields: str, emit: Callable[[str], None]) -> tuple[dict[str, str], str]:
    """Query `nvidia-smi`, returning `(rows-by-bus, raw stdout)`.

    Raises `TimeoutExpired` and `CalledProcessError` to the caller, which owns
    the classification: a timeout and a refusal are different facts.
    """
    try:
        completed = _run_smi(
            [
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            env=_smi_env(),
        )
    except subprocess.TimeoutExpired:
        emit(_degraded_reason())
        raise
    rows: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        bus = parse_pci_bus_id(line)
        if bus is not None:
            rows[bus] = line
    return rows, completed.stdout


def probe_nvidia(emit: Callable[[str], None] | None = None) -> list[Device]:
    """Enumerate NVIDIA GPUs and classify each one's health.

    Enumeration is a glob of `/proc/driver/nvidia/gpus/*/information`, sorted
    by bus address so the numbering is stable across runs — glob order is not
    bus order. `nvidia-smi` is consulted only for the fields procfs cannot
    supply: VRAM and compute capability.
    """
    log = emit or (lambda _message: None)
    info_paths = sorted(glob.glob(NVIDIA_INFORMATION_GLOB))
    texts = _read_text_files(info_paths)

    entries: list[tuple[str, str, str, bool]] = []
    for path in info_paths:
        text = texts.get(path)
        if text is None:
            log(f"could not read {path} within {PROBE_TIMEOUT_SECONDS:.0f} s")
            continue
        for model, uuid, bus in parse_nvidia_proc(text):
            entries.append((model, uuid, bus, _nvidia_excluded(text)))
    entries.sort(key=lambda entry: _bus_sort_key(entry[2]))

    if not entries:
        return _probe_nvidia_without_procfs(log)

    try:
        rows, stdout = _call_smi(_SMI_QUERY_FIELDS, log)
    except subprocess.TimeoutExpired:
        rows = {}
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()[:200]
        return [
            _nvidia_device(
                index=index,
                model=model,
                bus=bus,
                excluded=excluded,
                info=None,
                reason=f"driver reported no device: {stderr}",
                health=UNHEALTHY,
            )
            for index, (model, _uuid, bus, excluded) in enumerate(entries)
        ]
    else:
        if not rows and not stdout.strip():
            return [
                _nvidia_device(
                    index=index,
                    model=model,
                    bus=bus,
                    excluded=excluded,
                    info=None,
                    reason="driver reported no device: empty response",
                    health=UNHEALTHY,
                )
                for index, (model, _uuid, bus, excluded) in enumerate(entries)
            ]

    parsed = parse_nvidia_smi("\n".join(rows.values()))
    devices: list[Device] = []
    for index, (model, _uuid, bus, excluded) in enumerate(entries):
        info = parsed.get(bus)
        if excluded:
            devices.append(
                _nvidia_device(
                    index=index,
                    model=model,
                    bus=bus,
                    excluded=True,
                    info=info,
                    reason="excluded by the NVIDIA driver",
                    health=UNHEALTHY,
                )
            )
        elif info is None:
            devices.append(
                _nvidia_device(
                    index=index,
                    model=model,
                    bus=bus,
                    excluded=False,
                    info=None,
                    reason=_degraded_reason(),
                    health=DEGRADED,
                )
            )
        else:
            devices.append(
                _nvidia_device(
                    index=index,
                    model=model,
                    bus=bus,
                    excluded=False,
                    info=info,
                    reason=None,
                    health=HEALTHY,
                )
            )
    if any(device.usable is False for device in devices):
        log("at least one NVIDIA device is unusable and will be refused")
    return devices


def _bus_sort_key(bus: str) -> tuple[int, int, int, int]:
    """Sort key that orders bus addresses the way the hardware does."""
    match = _BUS_RE.match(bus)
    if match is None:
        return (1 << 30, 0, 0, 0)
    domain, bus_no, device_no, function = match.groups()
    return (
        int(domain, 16) if domain else 0,
        int(bus_no, 16),
        int(device_no, 16),
        int(function),
    )


def _nvidia_device(
    *,
    index: int,
    model: str,
    bus: str,
    excluded: bool,
    info: SmiInfo | None,
    reason: str | None,
    health: str,
) -> Device:
    """Build one NVIDIA `Device` from procfs and SMI data.

    A `DEGRADED` device stays usable: it is present, the app just cannot know
    how much VRAM it has, so it runs FP32 — the same conclusion the reference
    workload reached by measurement.
    """
    return Device(
        index=index,
        name=model,
        vendor="nvidia",
        backend="onnx:cuda",
        total_memory_bytes=info.total_memory_bytes if info else 0,
        pci_bus_id=bus,
        compute_capability=info.compute_capability if info else None,
        usable=health != UNHEALTHY,
        unusable_reason=reason,
    )


def _probe_nvidia_without_procfs(log: Callable[[str], None]) -> list[Device]:
    """Fall back to `nvidia-smi` alone when procfs is unreadable.

    Containers and hardened kernels can hide `/proc/driver/nvidia`. Answering
    "no GPUs" there would be wrong; a name and bus address from SMI is enough
    to list the device honestly.
    """
    try:
        rows, _stdout = _call_smi(_SMI_QUERY_FIELDS_WITH_NAME, log)
    except subprocess.TimeoutExpired:
        log("nvidia-smi timed out and no NVIDIA devices are visible in procfs")
        return []
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()[:200]
        log(f"nvidia-smi failed and no NVIDIA devices are visible in procfs: {stderr}")
        return []

    parsed = parse_nvidia_smi("\n".join(rows.values()))
    devices: list[Device] = []
    for bus, line in rows.items():
        parts = [part.strip() for part in line.split(",")]
        name = parts[1] if len(parts) > 1 else "NVIDIA GPU"
        info = parsed.get(bus)
        devices.append(
            Device(
                index=len(devices),
                name=name,
                vendor="nvidia",
                backend="onnx:cuda",
                total_memory_bytes=info.total_memory_bytes if info else 0,
                pci_bus_id=bus,
                compute_capability=info.compute_capability if info else None,
                usable=True,
                unusable_reason=None,
            )
        )
    devices.sort(key=lambda device: _bus_sort_key(device.pci_bus_id or ""))
    for position, device in enumerate(devices):
        devices[position] = Device(**{**device.__dict__, "index": position})
    return devices


def probe_amd(emit: Callable[[str], None] | None = None) -> list[Device]:
    """Enumerate AMD GPUs from KFD topology.

    The backend is `onnx:migraphx`: `ROCmExecutionProvider` was removed in
    ONNX Runtime 1.23+, so MIGraphX is the only remaining path — and it needs
    a host ROCm 7.x install plus Python 3.12 for its wheel. Without ROCm the
    devices are listed and refused, with the reason stated, rather than
    quietly absent.
    """
    log = emit or (lambda _message: None)
    node_paths = sorted(glob.glob(KFD_TOPOLOGY_GLOB))
    if not node_paths:
        return []
    texts = _read_text_files(node_paths)
    entries: list[dict[str, str]] = []
    for path in node_paths:
        text = texts.get(path)
        if text is None:
            log(f"could not read {path} within {PROBE_TIMEOUT_SECONDS:.0f} s")
            continue
        entries.extend(parse_kfd_topology(text))
    if not entries:
        return []

    version = rocm_version()
    usable = version is not None
    reason = (
        None
        if usable
        else "AMD GPU acceleration on Linux requires ROCm 7.x; see the Runtimes tab"
    )
    if not usable:
        log(reason or "")
    devices = [
        Device(
            index=index,
            name=_amd_name(entry),
            vendor="amd",
            backend="onnx:migraphx",
            total_memory_bytes=0,
            pci_bus_id=entry.get("location_id"),
            compute_capability=None,
            usable=usable,
            unusable_reason=reason,
        )
        for index, entry in enumerate(entries)
    ]
    return devices


def _amd_name(entry: dict[str, str]) -> str:
    """A readable AMD name, from `gfx_arch` when the file has one."""
    arch = entry.get("gfx_arch", "").replace("gfx", "")
    location = entry.get("location_id", "")
    if arch and location:
        return f"AMD {arch} ({location})"
    if arch:
        return f"AMD {arch}"
    return f"AMD GPU ({location})" if location else "AMD GPU"


def probe_intel(emit: Callable[[str], None] | None = None) -> list[Device]:
    """Enumerate Intel integrated GPUs from DRM.

    The backend is `torch.xpu`: there is no ONNX Runtime Intel EP, and the
    Intel Extension for PyTorch is archived and end-of-life, so it is never
    used. The glob is `card[0-9]*` and not `card*` because connector
    directories such as `card2-HDMI-A-1` also expose `device/vendor` and would
    otherwise report the same GPU three times.
    """
    del emit  # nothing to log: DRM is a directory read with no subprocess
    vendor_paths = sorted(glob.glob(DRM_VENDOR_GLOB))
    if not vendor_paths:
        return []
    texts = _read_text_files(vendor_paths)
    devices: list[Device] = []
    for path in vendor_paths:
        text = texts.get(path)
        if text is None or text.strip().lower() != INTEL_PCI_VENDOR:
            continue
        # `Path.parts`, not `path.split("/")`: sysfs is a POSIX path on
        # Linux and a backslash-separated one on Windows, and the literal
        # split raised IndexError there rather than returning a card name.
        card = Path(path).parts[-3]
        devices.append(
            Device(
                index=len(devices),
                name=f"Intel GPU ({card})",
                vendor="intel",
                backend="torch:xpu",
                total_memory_bytes=0,
                pci_bus_id=None,
                compute_capability=None,
                usable=True,
                unusable_reason=None,
            )
        )
    return devices


def probe_apple(emit: Callable[[str], None] | None = None) -> list[Device]:
    """The single Apple accelerator, served by CoreML.

    `compute_capability` and `pci_bus_id` are `None` because CoreML exposes
    neither; the app must not invent them. The device is refused below macOS 12
    because `ModelFormat: MLProgram` is refused there.
    """
    log = emit or (lambda _message: None)
    version = macos_version()
    machine = platform.machine() or "unknown"
    usable = version >= MACOS_COREML_MIN
    reason = None
    if not usable:
        reason = (
            "CoreML MLProgram requires macOS 12 or newer; "
            f"this is macOS {version[0]}.{version[1]}"
        )
        log(reason)
    return [
        Device(
            index=0,
            name=f"Apple GPU ({machine})",
            vendor="apple",
            backend="onnx:coreml",
            total_memory_bytes=0,
            pci_bus_id=None,
            compute_capability=None,
            usable=usable,
            unusable_reason=reason,
        )
    ]


def cpu_device() -> Device:
    """The always-present CPU fallback, so a GPU-less machine is still runnable."""
    name = platform.processor() or platform.machine() or "CPU"
    return Device(
        index=0,
        name=name,
        vendor="cpu",
        backend="cpu",
        total_memory_bytes=0,
        pci_bus_id=None,
        compute_capability=None,
        usable=True,
        unusable_reason=None,
    )


def probe_all(emit: Callable[[PipelineEvent], None] | None = None) -> list[Device]:
    """Enumerate every device this machine can run on, CPU last.

    `emit` receives one `log` event per probe outcome — a driver that did not
    answer, a vendor whose runtime is missing — so a user who sees three
    devices marked FP32 learns why without opening a log file.
    """
    if emit is None:

        def emit(event: PipelineEvent) -> None:
            del event

    def log(message: str) -> None:
        if message:
            emit(PipelineEvent(kind="log", message=message))

    devices: list[Device] = []
    if platform.system() == "Darwin":
        devices.extend(probe_apple(log))
    else:
        devices.extend(probe_nvidia(log))
        devices.extend(probe_amd(log))
        devices.extend(probe_intel(log))
    devices.append(cpu_device())
    return devices
