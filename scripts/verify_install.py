#!/usr/bin/env python3
"""Prove a built bundle is not broken, on a machine with no display and no GPU.

Every packaging job in CI ends by running this against the artefact it just
built, and the same command is what a maintainer runs before publishing. It is
the gate between "PyInstaller exited 0" and "there is a working application on a
user's machine", which are not the same thing.

Three checks, in the order a failure becomes expensive:

1. **The bundle launches and opens exactly one window.** The frozen executable
   is started with `QT_QPA_PLATFORM=offscreen` against a throwaway `$HOME`, and
   is expected to still be running when the observation window closes. The
   window count is read from the application's own log: `ui_pyside.runtimes_tab`
   writes one `resolved onnxruntime:` line per `RuntimsTab`, and a `RuntimesTab`
   is only ever constructed by a `MainWindow`, which only exists once a
   `QApplication` has. A second line therefore *is* a second window — which is
   precisely the failure `multiprocessing.freeze_support()` prevents, where every
   spawned pool worker re-executes the entry point and opens its own window. On
   Linux the count is corroborated by reading `/proc` for further processes
   running the same executable, and by `xdotool` when a window server and that
   tool are both present. Every skipped observable is reported as skipped, never
   silently.

2. **A real job runs on every device the app can see.** A 1-frame clip with an
   audio track is encoded through the real pipeline — extract, upscale, the
   completeness check, encode — on a throwaway work directory, once per probed
   device, each in its own process so a wedged accelerator cannot hang the run.
   A device the app lists as usable but that cannot reach an execution provider
   is a failure, because that is the lie the probe exists not to tell. Pass
   `--allow-unreachable-devices` to downgrade that to a report, which is what a
   maintainer does on a machine whose GPUs are known to be down.

3. **The bundle ships the execution providers it claims to.** ONNX Runtime's
   GPU providers are separate shared objects under `onnxruntime/capi/`, so their
   presence in the artefact — not in the build environment — is what says whether
   an accelerator can work at all.

Nothing here requires a GPU, and nothing touches the user's real data directory:
`HOME`, `XDG_DATA_HOME`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME`, `APPDATA` and
`LOCALAPPDATA` are all redirected into a temporary directory that is removed on
the way out unless `--keep-work-dir` says otherwise.

Exit codes: 0 every check passed, 1 a check failed (each failure is named on
stderr), 2 the command line was wrong.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()

#: The marker the job child prints its result on. A sentinel rather than the
#: last line of stdout, because ONNX Runtime writes warnings to stdout that have
#: nothing to do with the answer.
JOB_RESULT_MARKER = "VERIFY-INSTALL-JOB "

#: The log line `ui_pyside.runtimes_tab.log_resolved_runtimes` writes exactly
#: once per `RuntimesTab`, and therefore once per window.
WINDOW_MARKER = "resolved onnxruntime:"

#: `imageio_ffmpeg` ships a static 7.0.2 binary and `find_ffmpeg()` prefers
#: `PATH`; this smoke has to build a clip before it can run a job, so it needs an
#: ffmpeg even on a runner that has none on `PATH`.
MINIMUM_CLIP_FRAMES = 1
DEFAULT_CLIP_FRAMES = 10
DEFAULT_CLIP_SIZE = (320, 240)

#: The rate the generated clip is built at. It is a constant rather than a flag
#: because the clip's duration is what bounds it, and a rate nobody chose would
#: make the frame count a lie.
CLIP_FRAME_RATE = 10

#: `core.models.MIN_MODEL_BYTES`. The generated fixture has to clear it, or
#: `verify_model` rejects it as a failed download — which is correct behaviour
#: and not what this check is about.
MIN_MODEL_BYTES = 1_048_576

#: The 2x the generated graph upscales by, and the size its convolutions use.
SMOKE_SCALE = 2
SMOKE_CHANNELS = 3
SMOKE_KERNEL = 3

#: Seconds the launched application is given to reach its first log line, and
#: the total time it is then watched before it is stopped. A frozen Qt app that
#: is going to fail does so in the first second or two; the rest is margin for a
#: loaded CI runner importing torch.
LAUNCH_READY_SECONDS = 20.0
LAUNCH_OBSERVE_SECONDS = 5.0

#: How long one device's whole job may take. Generous, because the first ORT
#: session on a machine that has never run one is slow, and bounded, because a
#: wedged GPU otherwise never returns.
DEFAULT_DEVICE_TIMEOUT = 300.0


class VerificationError(Exception):
    """One named check failed. The message is what the reader is shown."""


# --------------------------------------------------------------------------- #
# locating the bundle
# --------------------------------------------------------------------------- #


def _executable_names() -> tuple[str, str]:
    """The frozen executable's name, with and without a Windows extension."""
    return ("upscaler", "upscaler.exe")


def resolve_bundle(app_dir: Path, executable: Path | None) -> tuple[Path, Path]:
    """Find the frozen executable and the directory it keeps its payload in.

    `app_dir` may be a PyInstaller `--onedir` bundle (`dist/upscaler/`), a
    `dpkg-deb -x` root (`/usr/`), or a macOS `.app`. `executable` overrides the
    search entirely, and may itself name a `.app` directory.

    Returns the executable to run and the directory whose subdirectories hold the
    bundled libraries and data.
    """
    if executable is not None:
        target = executable.resolve()
        if target.is_dir() and target.suffix == ".app":
            inner = target / "Contents" / "MacOS" / "upscaler"
            if not inner.is_file():
                raise VerificationError(
                    f"{target} is a bundle with no Contents/MacOS/upscaler in it"
                )
            return inner, target / "Contents" / "Frameworks"
        if not target.is_file():
            raise VerificationError(f"no such executable: {target}")
        if not os.access(target, os.X_OK):
            raise VerificationError(f"{target} is not executable")
        # The packaged launcher lives in .../bin and its payload in
        # .../lib/upscaler, so "the directory it keeps its payload in" is not its
        # own parent. Getting this wrong would point the provider check at a
        # directory that never has anything in it.
        if target.parent.name == "bin":
            sibling = target.parent.parent / "lib" / "upscaler"
            if (sibling / target.name).is_file():
                return target, sibling
        return target, target.parent

    if app_dir is None:
        raise VerificationError("one of --app-dir or --executable is required")

    root = app_dir.resolve()
    if not root.is_dir():
        raise VerificationError(f"--app-dir {root} is not a directory")

    candidates: list[tuple[Path, Path, str]] = [
        (root, root, "a PyInstaller --onedir bundle"),
        # The directory PyInstaller's --distpath produces is *named* upscaler, so
        # pointing --app-dir at the dist directory rather than at the bundle
        # inside it is the obvious thing to do and has to work.
        (root / "upscaler", root / "upscaler", "a dist directory holding the bundle"),
        (
            root / "usr" / "lib" / "upscaler",
            root / "usr" / "lib" / "upscaler",
            "a dpkg-deb -x root",
        ),
        (root / "Contents", root / "Contents" / "Frameworks", "a macOS .app bundle"),
    ]
    launcher = root / "usr" / "bin" / "upscaler"
    tried: list[str] = []
    for exe_name in _executable_names():
        for payload, runtime, description in candidates:
            executable_path = payload / exe_name
            tried.append(str(executable_path))
            if executable_path.is_file() and os.access(executable_path, os.X_OK):
                note(f"bundle      : {executable_path} ({description})")
                return executable_path, runtime
    if launcher.is_file():
        # Preferred over the bare binary: a user runs the launcher, and it is
        # what puts the bundle's libraries ahead of the host's.
        note(f"bundle      : {launcher} (the packaged launcher)")
        return launcher, root / "usr" / "lib" / "upscaler"
    tried.append(str(launcher))

    raise VerificationError(
        f"no frozen executable under {root}. Looked for:\n  "
        + "\n  ".join(tried)
        + "\nPass --executable to name it directly."
    )


def note(message: str) -> None:
    """A line of progress. stderr, so the summary can be piped on its own."""
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# environments
# --------------------------------------------------------------------------- #


def isolated_environment(root: Path, bundle_runtime: Path | None) -> dict[str, str]:
    """The child's whole environment: a throwaway home, plus the bundle's.

    `root` is the smoke's own scratch directory. Every variable the app reads to
    find a writable home is redirected into it, so a run on a developer's machine
    cannot touch the models, runtimes and logs they actually use.
    """
    env = dict(os.environ)
    home = root / "home"
    for relative in ("data", "config", "cache", "roaming", "local"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["XDG_DATA_HOME"] = str(root / "data")
    env["XDG_CONFIG_HOME"] = str(root / "config")
    env["XDG_CACHE_HOME"] = str(root / "cache")
    env["APPDATA"] = str(root / "roaming")
    env["LOCALAPPDATA"] = str(root / "local")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), *([str(bundle_runtime)] if bundle_runtime else [])]
    )
    if bundle_runtime is not None:
        env.update(_runtime_path_entries(bundle_runtime))
    return env


def _runtime_path_entries(bundle_runtime: Path) -> dict[str, str]:
    """Put the bundle's libraries and binaries ahead of the host's.

    A frozen executable gets this from its own `RPATH` and from the launcher
    script; a plain interpreter does not, and the difference shows up as a
    `libopenblas-*.so: cannot open shared object file` from a vendored OpenCV.
    """
    entries = {
        "linux": ("LD_LIBRARY_PATH", bundle_runtime),
        "darwin": ("DYLD_LIBRARY_PATH", bundle_runtime),
        "win32": ("PATH", bundle_runtime),
    }
    variable, directory = entries[sys.platform]
    existing = os.environ.get(variable, "")
    joined = str(directory) + (os.pathsep + existing if existing else "")
    result = {variable: joined}
    if sys.platform == "darwin":
        # A .app keeps executables in Contents/MacOS, but this smoke's job phase
        # runs the host interpreter, so the bundle's Resources/bin is what has to
        # come first when the bundle brought an ffmpeg.
        resources = bundle_runtime.parent / "Resources" / "bin"
        if resources.is_dir():
            result["PATH"] = f"{resources}{os.pathsep}{os.environ.get('PATH', '')}"
    elif sys.platform != "win32":
        binaries = bundle_runtime / "bin"
        if binaries.is_dir():
            result["PATH"] = f"{binaries}{os.pathsep}{os.environ.get('PATH', '')}"
    return result


# --------------------------------------------------------------------------- #
# the source clip
# --------------------------------------------------------------------------- #


def find_ffmpeg() -> str:
    """An ffmpeg to build the clip with: `PATH`, then imageio-ffmpeg's own."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        raise VerificationError(
            "no ffmpeg: put one on PATH or install imageio-ffmpeg, because the "
            "smoke has to build a source clip before it can run a job"
        ) from None
    return str(imageio_ffmpeg.get_ffmpeg_exe())


def make_clip(path: Path, *, frames: int, width: int, height: int) -> Path:
    """Build a real clip: `testsrc2` video, a sine audio track, `frames` frames.

    The audio is not decoration. Both scripts this application replaced dropped
    every audio stream, so a silent output is the exact failure it exists to
    prevent, and a smoke built on a silent clip cannot see it.

    The output is bounded with `-t` rather than `-frames:v`, which is a measured
    requirement and not a style choice: `-frames:v` makes the muxer stop as soon
    as the last video packet is written, which is before the AAC encoder has
    emitted anything, so a one-frame clip built that way has no audio stream at
    all. `-t frames/rate` gives the same number of frames and keeps the audio.
    """
    if frames < MINIMUM_CLIP_FRAMES:
        raise VerificationError(
            f"--frames must be at least {MINIMUM_CLIP_FRAMES}, got {frames}"
        )
    seconds = frames / CLIP_FRAME_RATE
    command = [
        find_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={width}x{height}:rate={CLIP_FRAME_RATE}",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000",
        "-t",
        f"{seconds:.4f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not path.is_file():
        raise VerificationError(
            "could not build a source clip with ffmpeg: "
            + (completed.stderr.strip()[-500:] or "ffmpeg printed nothing")
        )
    if not _describe_output(path)[0]:
        raise VerificationError(
            f"the source clip ffmpeg just built has no audio stream: {path}. A "
            "clip without audio cannot prove the encoder keeps audio, which is "
            "the failure this check exists for."
        )
    return path


# --------------------------------------------------------------------------- #
# the generated model
# --------------------------------------------------------------------------- #


def make_smoke_model(path: Path, *, height: int, width: int) -> Path:
    """Write a real ONNX graph that upscales 2x, for a self-contained smoke.

    `Conv -> LeakyRelu -> Resize` is the smallest graph that exercises every
    provider: a convolution, a nonlinearity and an upsample, all three of which
    the CoreML and DirectML operation tables actually list. The weights make it a
    3x3 box mean per colour channel, so a constant frame comes back constant and
    a channel swap is visible.

    A second initializer carries the model past `MIN_MODEL_BYTES`. It is
    referenced by no node, which ONNX permits: the size floor is there to catch a
    truncated download, and this fixture is not a download. Every other part of
    the file is a graph the runtime really executes.
    """
    if height < 1 or width < 1:
        raise VerificationError(
            f"the generated model needs positive dimensions, got {width}x{height}"
        )
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    shape = [1, SMOKE_CHANNELS, height, width]
    kernel = np.full(
        (SMOKE_CHANNELS, 1, SMOKE_KERNEL, SMOKE_KERNEL),
        1.0 / (SMOKE_KERNEL * SMOKE_KERNEL),
        dtype=np.float32,
    )
    scales = np.array(
        [1.0, 1.0, float(SMOKE_SCALE), float(SMOKE_SCALE)], dtype=np.float32
    )
    # `Resize` at opset 13 takes either `scales` or `sizes`, never both, and
    # shape inference treats an empty `sizes` input as "both were provided" — so
    # `sizes` is left off the node entirely and only the scales carry the
    # geometry. `roi` is required to be present for the input positions to line
    # up, and is an empty float tensor when the transformation is asymmetric.
    empty_roi = np.array([], dtype=np.float32)
    padding = np.zeros(max(0, MIN_MODEL_BYTES), dtype=np.uint8)

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "box"],
            ["blurred"],
            kernel_shape=[SMOKE_KERNEL, SMOKE_KERNEL],
            pads=[1, 1, 1, 1],
            group=SMOKE_CHANNELS,
            name="box_mean",
        ),
        helper.make_node(
            "LeakyRelu", ["blurred"], ["activated"], alpha=0.1, name="leaky"
        ),
        helper.make_node(
            "Resize",
            ["activated", "roi", "scales"],
            ["big"],
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
            name="upscale",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "upscaler-smoke",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [
            helper.make_tensor_value_info(
                "big",
                TensorProto.FLOAT,
                [1, SMOKE_CHANNELS, height * SMOKE_SCALE, width * SMOKE_SCALE],
            )
        ],
        [
            numpy_helper.from_array(kernel, "box"),
            numpy_helper.from_array(scales, "scales"),
            # `roi` is empty for asymmetric coordinate transformation and
            # `sizes` is empty when the scales are given; the schema still
            # requires both.
            numpy_helper.from_array(empty_roi, "roi"),
            numpy_helper.from_array(padding, "smoke_padding_to_clear_min_bytes"),
        ],
    )
    # The IR version has to be one the *runtime* accepts, not one the `onnx`
    # package writes by default: onnx 1.23 emits IR 14 and ONNX Runtime 1.22
    # refuses anything above 10 with "Unsupported model IR version". The
    # runtime's ceiling moves with the release, so it is measured by loading
    # the graph rather than hard-coded — and the version that worked is
    # reported, because a smoke that quietly pins an old IR would hide the
    # moment the runtime dropped it.
    import onnxruntime

    candidates = sorted({min(onnx.IR_VERSION, 10), onnx.IR_VERSION, 9, 8}, reverse=True)
    last_error = ""
    for ir_version in candidates:
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17)],
            producer_name="scripts/verify_install.py",
        )
        model.ir_version = ir_version
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
        try:
            onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as exc:  # the message names the version it refused
            last_error = str(exc)
            continue
        note(
            f"model ir    : version {ir_version}, the highest this "
            f"onnxruntime build accepts"
        )
        break
    else:
        raise VerificationError(
            f"the generated model does not load at IR version {candidates[0]}: "
            f"{last_error}"
        )
    if path.stat().st_size < MIN_MODEL_BYTES:
        raise VerificationError(
            f"the generated model is {path.stat().st_size} bytes, under the "
            f"{MIN_MODEL_BYTES}-byte floor core.models.verify_model enforces"
        )
    return path


# --------------------------------------------------------------------------- #
# check 1: the bundle launches, once
# --------------------------------------------------------------------------- #


@dataclass
class LaunchOutcome:
    """What the launched application did, as observed from outside."""

    survived: bool
    window_lines: int
    extra_processes: int
    x11_windows: int | None
    log_path: Path | None
    resolved: dict[str, str]
    stderr_tail: str
    returncode: int | None


def _find_log(root: Path) -> Path | None:
    """The application's own log, wherever the platform puts it."""
    matches = sorted(root.glob("**/upscaler.log"))
    return matches[0] if matches else None


def _read_resolved(log: Path) -> tuple[dict[str, str], int]:
    """The two resolution lines, and how many windows wrote them.

    `ui_pyside.runtimes_tab.log_resolved_runtimes` logs one `resolved
    <module>: <path>` line per module per construction of the tab, and the tab
    is only built by a `MainWindow`, which only exists after a `QApplication`.
    Counting those lines is therefore counting windows — which is the whole
    point, because a spawn worker that re-runs the entry point builds a second
    tab, writes a second pair of lines and shows a second window.
    """
    text = log.read_text(encoding="utf-8", errors="replace")
    resolved: dict[str, str] = {}
    for line in text.splitlines():
        if "resolved " not in line:
            continue
        _, _, rest = line.partition("resolved ")
        name, _, path = rest.partition(": ")
        if name in ("onnxruntime", "torch"):
            resolved.setdefault(name, path.strip())
    return resolved, text.count(WINDOW_MARKER)


def _same_executable_processes(pid: int, target: Path) -> int:
    """How many live processes are running `target`, other than `pid` itself.

    Linux only, because it is the only platform this runs on without a
    dependency: `/proc/<pid>/exe` is a symlink to the running image, so it
    identifies a re-executed frozen child exactly. A broken freeze guard does
    not show up as a second *window* in one process — it shows up as a second
    process running the same binary, which is what this counts.
    """
    proc = Path("/proc")
    if not proc.is_dir():
        return 0
    try:
        mine = target.stat()
    except OSError:
        return 0
    found = 0
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            executable = (entry / "exe").resolve()
            info = executable.stat()
        except OSError:
            continue
        if info.st_ino == mine.st_ino and info.st_dev == mine.st_dev:
            found += 1
    return max(0, found - 1)


def _x11_window_count(pid: int) -> int | None:
    """Top-level windows owned by `pid`, or `None` when that cannot be asked.

    `xdotool` is the only window-counting tool in common enough use to be worth
    shelling out to, and it is not installed everywhere. Returning `None` means
    "not measured", and the caller says so rather than passing silently.
    """
    if sys.platform != "linux" or not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return None
    tool = shutil.which("xdotool")
    if tool is None:
        return None
    completed = subprocess.run(
        [tool, "search", "--pid", str(pid), "--onlyvisible", "--name", ".+"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return len([line for line in completed.stdout.splitlines() if line.strip()])


def check_launch(
    executable: Path,
    env: dict[str, str],
    root: Path,
    *,
    ready_seconds: float,
    observe_seconds: float,
) -> LaunchOutcome:
    """Start the frozen application, watch it, and stop it.

    The application is a GUI: it is expected to be running when the observation
    window closes and to be killed afterwards, so a non-zero exit *after* the
    log appears is expected and a non-zero exit *before* it is a crash.
    """
    launch_env = dict(env)
    launch_env["QT_QPA_PLATFORM"] = "offscreen"
    started = time.monotonic()
    process = subprocess.Popen(
        [str(executable)],
        cwd=str(root),
        env=launch_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    log = None
    resolved: dict[str, str] = {}
    window_lines = 0
    extra = 0
    x11: int | None = None
    try:
        while time.monotonic() - started < ready_seconds:
            if process.poll() is not None:
                break
            candidate = _find_log(root)
            if candidate is not None:
                found, lines = _read_resolved(candidate)
                if lines:
                    log, resolved, window_lines = candidate, found, lines
                    break
            time.sleep(0.2)

        deadline = time.monotonic() + observe_seconds
        while time.monotonic() < deadline:
            extra = max(extra, _same_executable_processes(process.pid, executable))
            if x11 is None:
                x11 = _x11_window_count(process.pid)
            if process.poll() is not None:
                break
            time.sleep(0.25)
        survived = process.poll() is None
    finally:
        if process.poll() is None:
            _terminate(process)

    stderr_tail = ""
    if process.stderr is not None:
        try:
            stderr_tail = process.stderr.read()[-2000:]
        except ValueError:  # the pipe was closed by the kill above
            stderr_tail = ""
    return LaunchOutcome(
        survived=survived,
        window_lines=window_lines,
        extra_processes=extra,
        x11_windows=x11,
        log_path=log,
        resolved=resolved,
        stderr_tail=stderr_tail,
        returncode=process.returncode,
    )


def _terminate(process: subprocess.Popen[str]) -> None:
    """Stop the application and reap it, escalating if it ignores SIGTERM."""
    process.terminate()
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    if sys.platform == "win32":
        process.kill()
    else:
        # The whole session, because a frozen app that already spawned pool
        # workers leaves them behind and they would hold the work directory.
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except (ProcessLookupError, PermissionError):
            process.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)


def report_launch(outcome: LaunchOutcome, executable: Path) -> list[str]:
    """Describe the launch, and return the failures found in it."""
    note(f"launch      : {executable}")
    failures: list[str] = []
    if not outcome.survived:
        detail = outcome.stderr_tail.strip()[-600:]
        failures.append(
            f"the frozen application exited with {outcome.returncode} instead of "
            f"staying up.\n{detail or '(it printed nothing to stderr)'}"
        )
        note(f"launch      : FAILED, exit {outcome.returncode}")
        return failures
    note("launch      : stayed up for the whole observation window")

    if outcome.log_path is None:
        failures.append(
            "the frozen application never wrote upscaler.log, so it never got as "
            "far as building its tabs. The bundle's Qt plugins or its data "
            "directory are not usable."
        )
        note("launch      : FAILED, no log written")
        return failures
    note(f"log         : {outcome.log_path}")

    for module, path in sorted(outcome.resolved.items()):
        note(f"resolved    : {module} = {path}")

    if outcome.window_lines == 1:
        note("windows     : exactly 1 (one RuntimesTab constructed, one window)")
    else:
        note(f"windows     : {outcome.window_lines}")
        failures.append(
            f"the application built {outcome.window_lines} windows where exactly "
            "one is correct. Each 'resolved onnxruntime:' line is one MainWindow, "
            "and more than one means a spawned worker re-ran the entry point — "
            "the failure multiprocessing.freeze_support() prevents."
        )
    if outcome.extra_processes:
        note(
            f"processes   : {outcome.extra_processes} extra copy/copies of the "
            "frozen executable"
        )
        failures.append(
            f"{outcome.extra_processes} further processes are running "
            f"{executable.name}. A spawn worker that re-executes the frozen binary "
            "opens a window of its own; the freeze guard is not holding."
        )
    else:
        note("processes   : 1 copy of the frozen executable (from /proc)")
    if outcome.window_lines and outcome.x11_windows == 1:
        note("x11 windows : exactly 1 (xdotool, by pid)")
    elif outcome.x11_windows is None:
        note("x11 windows : not measured (no xdotool, or no window server)")
    else:
        note(f"x11 windows : {outcome.x11_windows}")
    if outcome.resolved.get("onnxruntime", "").endswith("not installed"):
        note("resolved    : onnxruntime is not installed — this is the slim build")
    return failures


# --------------------------------------------------------------------------- #
# check 2: a job on every probed device
# --------------------------------------------------------------------------- #


@dataclass
class DeviceJob:
    label: str
    ok: bool
    unreachable: bool
    message: str
    output: Path | None
    output_bytes: int
    has_audio: bool
    dimensions: str
    providers: str
    seconds: float


def _interpret_job(
    label: str,
    completed: subprocess.CompletedProcess[str],
    output: Path,
    seconds: float,
    timed_out: bool,
) -> DeviceJob:
    """Turn the job child's result line into a verdict."""
    if timed_out:
        return DeviceJob(
            label,
            False,
            True,
            f"timed out after {seconds:.0f}s; the device never answered",
            None,
            0,
            False,
            "",
            "",
            seconds,
        )
    payload: dict[str, object] | None = None
    for line in completed.stdout.splitlines():
        if line.startswith(JOB_RESULT_MARKER):
            payload = json.loads(line[len(JOB_RESULT_MARKER) :])
    if payload is None:
        detail = (completed.stderr.strip() or completed.stdout.strip())[-600:]
        return DeviceJob(
            label,
            False,
            False,
            f"the job process produced no result line (exit "
            f"{completed.returncode}).\n" + (detail or "(it printed nothing)"),
            None,
            0,
            False,
            "",
            "",
            seconds,
        )
    if not payload.get("ok"):
        return DeviceJob(
            label,
            False,
            bool(payload.get("unreachable", False)),
            str(payload.get("error", "unknown error")),
            None,
            0,
            False,
            "",
            str(payload.get("providers", "")),
            seconds,
        )
    assert output.is_file(), f"the job reported success but {output} is missing"
    size = output.stat().st_size
    return DeviceJob(
        label,
        size > 0,
        False,
        str(payload.get("message", "")) if size else "the output file is empty",
        output,
        size,
        bool(payload.get("has_audio", False)),
        str(payload.get("dimensions", "")),
        str(payload.get("providers", "")),
        seconds,
    )


def check_jobs(
    clip: Path,
    model: Path,
    root: Path,
    env: dict[str, str],
    *,
    timeout: float,
    allow_unreachable: bool,
    only: str | None,
) -> tuple[list[DeviceJob], list[str]]:
    """Run the smoke job once per usable device, each in its own process."""
    sys.path.insert(0, str(REPO_ROOT))
    from core.devices import probe_all

    failures: list[str] = []
    wanted: set[int] | None = None
    if only:
        wanted = {int(part) for part in only.split(",") if part.strip()}

    probed = probe_all()
    if not probed:
        failures.append("core.devices.probe_all() returned no device at all")
        return [], failures

    note(
        "devices     : "
        + ", ".join(
            f"[{d.index}] {d.name} ({d.backend}, usable={d.usable})" for d in probed
        )
    )
    if all(d.vendor == "cpu" for d in probed):
        note(
            "devices     : no accelerator was found; the job runs on the CPU "
            "provider, which is the documented behaviour on a machine with no GPU"
        )

    targets = [d for d in probed if d.usable and (wanted is None or d.index in wanted)]
    if not targets:
        names = ", ".join(f"[{d.index}] {d.name}: {d.unusable_reason}" for d in probed)
        failures.append(f"no usable device to run the job on. The probe found: {names}")
        return [], failures

    results: list[DeviceJob] = []
    for device in targets:
        spec = root / "devices" / f"device-{device.index}.json"
        spec.parent.mkdir(parents=True, exist_ok=True)
        spec.write_text(
            json.dumps(
                {
                    "index": device.index,
                    "name": device.name,
                    "vendor": device.vendor,
                    "backend": device.backend,
                    "usable": device.usable,
                    "unusable_reason": device.unusable_reason,
                    "compute_capability": device.compute_capability,
                }
            ),
            encoding="utf-8",
        )
        work = root / "work" / f"device-{device.index}"
        output = root / "out" / f"device-{device.index}.mp4"
        work.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(SCRIPT_PATH),
            "--run-job",
            "--job-input",
            str(clip),
            "--job-model",
            str(model),
            "--job-work-dir",
            str(work),
            "--job-output",
            str(output),
            "--job-device",
            str(spec),
        ]
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
            timed_out = False
        except subprocess.TimeoutExpired:
            completed = subprocess.CompletedProcess(command, 124, "", "")
            timed_out = True
        elapsed = time.monotonic() - started
        results.append(
            _interpret_job(device.label, completed, output, elapsed, timed_out)
        )

    note("")
    for job in results:
        if job.ok:
            verdict = "ok"
        elif job.unreachable:
            verdict = "UNREACHABLE"
        else:
            verdict = "FAILED"
        note(f"job [{job.label}]: {verdict} in {job.seconds:.1f}s — {job.message}")
        if job.output is not None and job.output_bytes:
            note(
                f"              output {job.output.name}: "
                f"{job.output_bytes} bytes, {job.dimensions}, "
                f"audio={'yes' if job.has_audio else 'NO'}"
            )
        if job.providers:
            note(f"              providers {job.providers}")

    for job in results:
        if job.ok:
            continue
        if job.unreachable and allow_unreachable:
            note(
                f"job [{job.label}]: unreachable, tolerated by "
                "--allow-unreachable-devices"
            )
            continue
        kind = "was unreachable" if job.unreachable else "failed"
        failures.append(f"the job {kind} on {job.label}: {job.message}")
    return results, failures


def run_job_role(args: argparse.Namespace) -> int:
    """The other half of this script: run one job on one device, print JSON.

    It is a separate process because the whole point of `--device-timeout` is to
    be able to kill a device that will not answer, and a wedged accelerator
    cannot be killed from inside its own process.
    """
    sys.path.insert(0, str(REPO_ROOT))
    import multiprocessing as mp

    from core.config import Device, JobConfig
    from core.pipeline import run_job

    spec = json.loads(Path(args.job_device).read_text(encoding="utf-8"))
    capability = spec.get("compute_capability")
    device = Device(
        index=int(spec["index"]),
        name=str(spec["name"]),
        vendor=spec["vendor"],
        backend=str(spec["backend"]),
        total_memory_bytes=0,
        pci_bus_id=None,
        compute_capability=tuple(capability) if capability else None,
        usable=bool(spec["usable"]),
        unusable_reason=spec.get("unusable_reason"),
    )
    output = Path(args.job_output)
    config = JobConfig(
        input_path=Path(args.job_input),
        output_path=output,
        model_path=Path(args.job_model),
        work_dir=Path(args.job_work_dir),
        devices=(device,),
    )
    # Whether the interpreter this job would run in can offer the execution
    # provider the device needs. Decided *before* the job runs, because a
    # missing provider is a property of the build, not a fault in the pipeline,
    # and the two have to be reported differently: the fat CPU build on a
    # machine with an NVIDIA card is correct and cannot reach the GPU, while a
    # job that fails *with* the provider present is a real defect.
    providers, wanted, missing = _provider_state(device)
    if missing:
        print(
            JOB_RESULT_MARKER
            + json.dumps(
                {
                    "ok": False,
                    "unreachable": True,
                    "error": (
                        f"{device.name} cannot be reached by this build. "
                        f"{missing} It wanted: {wanted or 'nothing'}. "
                        f"Available: {providers or 'none'}."
                    ),
                    "providers": providers,
                    "onnxruntime_file": _origin("onnxruntime"),
                }
            ),
            flush=True,
        )
        return 1

    messages: list[str] = []
    error: str | None = None

    def emit(event: object) -> None:
        kind = getattr(event, "kind", "")
        message = str(getattr(event, "message", ""))
        if kind == "error":
            nonlocal error
            error = error or message
        elif message:
            messages.append(message)

    try:
        run_job(config, emit, mp.get_context("spawn").Event())
    except Exception as exc:
        error = error or f"{type(exc).__name__}: {exc}"

    has_audio, dimensions = _describe_output(output)
    source_has_audio = _describe_output(Path(args.job_input))[0]
    if error is None and source_has_audio and not has_audio:
        # The whole reason this project exists: both scripts it replaced dropped
        # every audio stream, and an output that is silently truncated in a way
        # nobody notices until they play it is the worst kind of wrong.
        error = (
            f"the source {Path(args.job_input).name} has an audio stream and "
            f"{output.name} does not. The encoder dropped it."
        )
    ok = error is None and output.is_file() and output.stat().st_size > 0
    result = {
        "ok": ok,
        "unreachable": False,
        "error": error,
        "providers": providers,
        "onnxruntime_file": _origin("onnxruntime"),
        "has_audio": has_audio,
        "source_has_audio": source_has_audio,
        "dimensions": dimensions,
        "message": messages[-1] if messages else "",
    }
    print(JOB_RESULT_MARKER + json.dumps(result), flush=True)
    return 0 if ok else 1


def _provider_state(device: object) -> tuple[str, str, str]:
    """What this interpreter can offer a device, and what it cannot.

    Returns `(available, wanted, missing)`. `core.backends.onnx_backend.providers_for`
    already refuses a device whose provider this build does not carry, with a
    message naming it, so that refusal *is* the answer and is passed on rather
    than caught and discarded.
    """
    try:
        import onnxruntime
    except ImportError as exc:
        return f"onnxruntime could not be imported: {exc}", "", ""
    from core.backends.onnx_backend import providers_for

    available = list(onnxruntime.get_available_providers())
    try:
        wanted = providers_for(device)
    except Exception as exc:
        return ", ".join(available), "not listed; the backend refused", str(exc)
    missing = [name for name in wanted if name not in available]
    return ", ".join(available), ", ".join(wanted), ", ".join(missing)


def _origin(module: str) -> str:
    """Where a module was imported from, without importing it twice."""
    import importlib.util

    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return "not importable"
    return spec.origin or "builtin" if spec else "not installed"


def _describe_output(path: Path) -> tuple[bool, str]:
    """Does the output have an audio stream, and what size is its video?

    Read back with the app's own `core.ffmpeg.probe`, so a file that is present
    but truncated is caught here rather than by a user.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False, "no output"
    sys.path.insert(0, str(REPO_ROOT))
    from core.ffmpeg import probe

    info = probe(path)
    return info.has_audio, f"{info.width}x{info.height}"


# --------------------------------------------------------------------------- #
# check 3: which execution providers the artefact actually carries
# --------------------------------------------------------------------------- #


def check_providers(bundle_runtime: Path | None) -> list[str]:
    """List the ONNX Runtime provider shared objects inside the artefact.

    The Python-level `get_available_providers()` answers for the interpreter,
    not for the package. The GPU execution providers are separate shared objects
    under `onnxruntime/capi/`, so whether one is in the bundle is the fact that
    decides whether an accelerator can work at all — and the fat CPU build
    carries none, which is exactly what the Runtimes tab is for.
    """
    failures: list[str] = []
    if bundle_runtime is None:
        note("providers   : no bundle directory to inspect")
        return failures
    candidates = [
        bundle_runtime / "onnxruntime" / "capi",
        bundle_runtime / "_internal" / "onnxruntime" / "capi",
    ]
    capi = next((path for path in candidates if path.is_dir()), None)
    if capi is None:
        note(
            f"providers   : this build carries no onnxruntime (looked in "
            f"{', '.join(str(path) for path in candidates)}) — the slim build, "
            "where the Runtimes tab installs one"
        )
        return failures
    found = sorted(
        item.name
        for item in capi.iterdir()
        if item.name.startswith("libonnxruntime_providers_")
    )
    accelerators = [name for name in found if "shared" not in name]
    if found:
        note(f"providers   : {', '.join(found)}")
    if accelerators:
        note("             : accelerator execution providers are bundled")
    else:
        note(
            "             : no accelerator execution provider is bundled, so "
            "this build runs on the CPU provider until a runtime is installed"
        )
    return failures


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """The full command line, with a real description of every flag."""
    parser = argparse.ArgumentParser(
        prog="verify_install.py",
        description=(
            "Headless smoke test for a built Upscaler bundle: launch the frozen "
            "application and prove it opens exactly one window, run a real "
            "one-frame job on every device the probe can see, and report which "
            "ONNX Runtime execution providers the artefact actually carries. "
            "Needs no display and no GPU, and never touches the real data "
            "directory."
        ),
        epilog=(
            "Exit status: 0 every check passed, 1 a check failed with the reason "
            "on stderr, 2 the command line was wrong."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = parser.add_argument_group("what to test")
    target.add_argument(
        "--app-dir",
        type=Path,
        metavar="DIR",
        help=(
            "an unpacked bundle: a PyInstaller --onedir directory, a tree from "
            "`dpkg-deb -x package.deb /somewhere`, or a macOS .app. The frozen "
            "executable is located inside it."
        ),
    )
    target.add_argument(
        "--executable",
        type=Path,
        metavar="PATH",
        help=(
            "the frozen executable itself, or a macOS .app directory. Overrides "
            "--app-dir; use it when the bundle has been moved or renamed."
        ),
    )
    target.add_argument(
        "--source",
        type=Path,
        metavar="CLIP",
        help=(
            "a real video to use as the job's input. When omitted, one is built "
            "with ffmpeg: testsrc2 video, a sine audio track, "
            f"{DEFAULT_CLIP_FRAMES} frames. The audio is deliberate — a silent "
            "output is the bug this application was written to fix."
        ),
    )
    target.add_argument(
        "--model",
        type=Path,
        metavar="PATH",
        help=(
            "a .onnx or .pth checkpoint for the job. When omitted, a real 2x "
            "ONNX graph (Conv, LeakyRelu, Resize) is written into the work "
            "directory, and the run says so. The generated file is a fixture, "
            "not a super-resolution model: it proves the engine runs, not that "
            "it sharpens anything."
        ),
    )
    timing = parser.add_argument_group("how long to wait")
    timing.add_argument(
        "--launch-ready-timeout",
        type=float,
        default=LAUNCH_READY_SECONDS,
        metavar="SECONDS",
        help=(
            "how long the application is given to write its first log line "
            f"(default: {LAUNCH_READY_SECONDS:.0f}). A frozen Qt app that is "
            "going to fail does so in the first second or two."
        ),
    )
    timing.add_argument(
        "--launch-observe",
        type=float,
        default=LAUNCH_OBSERVE_SECONDS,
        metavar="SECONDS",
        help=(
            "how long the application is watched, once it is up, before it is "
            f"stopped (default: {LAUNCH_OBSERVE_SECONDS:.0f}). The window and "
            "process counts are sampled throughout this window."
        ),
    )
    timing.add_argument(
        "--device-timeout",
        type=float,
        default=DEFAULT_DEVICE_TIMEOUT,
        metavar="SECONDS",
        help=(
            "how long one device's whole job may take before it is declared "
            f"unreachable (default: {DEFAULT_DEVICE_TIMEOUT:.0f}). Each device "
            "runs in its own process, so a wedged accelerator is killed rather "
            "than waited on."
        ),
    )
    scope = parser.add_argument_group("scope")
    scope.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_CLIP_FRAMES,
        metavar="N",
        help=(
            f"frames in the generated clip (default: {DEFAULT_CLIP_FRAMES}). Use "
            "--frames 1 for the smallest possible job. Ignored when --source is "
            "given."
        ),
    )
    scope.add_argument(
        "--devices",
        metavar="LIST",
        help=(
            "comma-separated device indices to run the job on. The default is "
            "every device the probe reports as usable, which on a machine with no "
            "accelerator is the CPU."
        ),
    )
    scope.add_argument(
        "--allow-unreachable-devices",
        action="store_true",
        help=(
            "report a device that cannot be reached instead of failing. Use it "
            "on a machine whose GPUs are known to be down; never use it to make "
            "a broken bundle pass."
        ),
    )
    skip = parser.add_argument_group("skip a phase")
    skip.add_argument(
        "--skip-launch",
        action="store_true",
        help="do not launch the frozen application; run the job checks only.",
    )
    skip.add_argument(
        "--skip-job",
        action="store_true",
        help="do not run a job; launch the application and report providers only.",
    )
    work = parser.add_argument_group("work directory")
    work.add_argument(
        "--work-dir",
        type=Path,
        metavar="DIR",
        help=(
            "where the clip, the generated model, the throwaway data "
            "directory and the per-device work directories go. Default: a "
            "temporary directory, removed on the way out."
        ),
    )
    work.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="keep the work directory after the run and print its path.",
    )
    parser.add_argument(
        "--run-job",
        action="store_true",
        help=argparse.SUPPRESS,  # the child role; not part of the public CLI
    )
    job = parser.add_argument_group("arguments for the internal --run-job role")
    job.add_argument("--job-input", type=Path, help=argparse.SUPPRESS)
    job.add_argument("--job-model", type=Path, help=argparse.SUPPRESS)
    job.add_argument("--job-work-dir", type=Path, help=argparse.SUPPRESS)
    job.add_argument("--job-output", type=Path, help=argparse.SUPPRESS)
    job.add_argument("--job-device", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run every check. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run_job:
        return run_job_role(args)
    if (args.app_dir is None) == (args.executable is None):
        parser.error("give exactly one of --app-dir and --executable")

    root = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else Path(tempfile.mkdtemp(prefix="upscaler-verify-"))
    )
    root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    try:
        executable, bundle_runtime = resolve_bundle(args.app_dir, args.executable)
        note(f"smoke root  : {root}")
        note(f"platform    : {sys.platform}, python {sys.version.split()[0]}")

        env = isolated_environment(root, bundle_runtime)
        note(
            f"isolation   : HOME and the data/config/cache directories are "
            f"redirected under {root}"
        )

        clip = args.source or make_clip(
            root / "source.mp4",
            frames=args.frames,
            width=DEFAULT_CLIP_SIZE[0],
            height=DEFAULT_CLIP_SIZE[1],
        )
        note(f"source clip : {clip} ({clip.stat().st_size} bytes)")

        model = args.model or make_smoke_model(
            root / "smoke-2x.onnx",
            height=DEFAULT_CLIP_SIZE[1],
            width=DEFAULT_CLIP_SIZE[0],
        )
        if args.model is None:
            note(
                f"model       : {model} — a generated 2x ONNX graph, not a real "
                "super-resolution model"
            )
        else:
            note(f"model       : {model}")

        if not args.skip_launch:
            outcome = check_launch(
                executable,
                env,
                root,
                ready_seconds=args.launch_ready_timeout,
                observe_seconds=args.launch_observe,
            )
            failures.extend(report_launch(outcome, executable))
        else:
            note("launch      : skipped (--skip-launch)")

        if not args.skip_job:
            note("")
            _, job_failures = check_jobs(
                clip,
                model,
                root,
                env,
                timeout=args.device_timeout,
                allow_unreachable=args.allow_unreachable_devices,
                only=args.devices,
            )
            failures.extend(job_failures)
        else:
            note("job         : skipped (--skip-job)")

        note("")
        check_providers(bundle_runtime)
    except VerificationError as exc:
        failures.append(str(exc))
    except KeyboardInterrupt:
        print("verify_install.py: interrupted", file=sys.stderr)
        return 1
    finally:
        if args.keep_work_dir:
            note(f"work dir    : kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)

    if failures:
        print("", file=sys.stderr)
        print(f"verify_install.py: {len(failures)} check(s) FAILED", file=sys.stderr)
        for number, failure in enumerate(failures, start=1):
            print(f"  {number}. {failure}", file=sys.stderr)
        return 1
    print("verify_install.py: every check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
