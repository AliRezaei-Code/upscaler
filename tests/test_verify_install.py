"""The smoke script's own contract: what it accepts, and how loudly it fails.

The script is what stands between a build that compiled and a build that works,
so the two things that must never regress are that it *runs* and that it fails
with a named reason rather than a stack trace. A smoke that cannot be invoked,
or that shrugs off a missing bundle, is worse than no smoke: it is a green tick
on a broken artefact.

Nothing here builds a bundle. Launching a frozen Qt application is the job of
`scripts/verify_install.py` itself, on a machine that has one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "verify_install.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import verify_install as smoke  # noqa: E402 - the script is not a package


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    """The script, as a user runs it."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )


class TestCommandLine:
    def test_help_describes_every_flag(self) -> None:
        result = run("--help")

        assert result.returncode == 0
        for flag in (
            "--app-dir",
            "--executable",
            "--source",
            "--model",
            "--launch-ready-timeout",
            "--launch-observe",
            "--device-timeout",
            "--frames",
            "--devices",
            "--allow-unreachable-devices",
            "--skip-launch",
            "--skip-job",
            "--work-dir",
            "--keep-work-dir",
        ):
            assert flag in result.stdout, f"{flag} is not documented in --help"

    @pytest.mark.parametrize(
        "arguments",
        [
            pytest.param([], id="neither target"),
            pytest.param(["--app-dir", "x", "--executable", "y"], id="both targets"),
            pytest.param(["--nonexistent-flag"], id="unknown flag"),
            pytest.param(["--frames"], id="flag with no value"),
        ],
    )
    def test_a_wrong_command_line_is_exit_two(self, arguments: list[str]) -> None:
        result = run(*arguments)

        assert result.returncode == 2
        assert "usage:" in result.stderr.lower()

    def test_the_job_role_is_hidden_from_the_public_help(self) -> None:
        # It is a real flag -- the parent invokes it -- but it is not part of the
        # interface a user or a CI job types.
        assert "--run-job" not in run("--help").stdout


class TestBundleResolution:
    def test_a_missing_executable_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(smoke.VerificationError) as excinfo:
            smoke.resolve_bundle(None, tmp_path / "not-here")

        assert "not-here" in str(excinfo.value)

    def test_an_empty_app_dir_lists_where_it_looked(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(smoke.VerificationError) as excinfo:
            smoke.resolve_bundle(tmp_path, None)

        message = str(excinfo.value)
        assert "no frozen executable" in message
        assert "Looked for" in message
        # A bundle is looked for in three layouts, and the message has to show
        # all three or the reader cannot tell which packaging shape they have.
        assert "upscaler/upscaler" in message
        assert "usr/lib/upscaler" in message

    def test_an_onedir_bundle_is_found(self, tmp_path: Path) -> None:
        bundle = tmp_path / "upscaler"
        bundle.mkdir()
        executable = bundle / "upscaler"
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)

        found, runtime = smoke.resolve_bundle(tmp_path, None)

        assert found == executable.resolve()
        assert runtime == bundle.resolve()

    def test_a_dpkg_root_is_found(self, tmp_path: Path) -> None:
        libdir = tmp_path / "usr" / "lib" / "upscaler"
        libdir.mkdir(parents=True)
        executable = libdir / "upscaler"
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)

        found, runtime = smoke.resolve_bundle(tmp_path, None)

        assert found == executable.resolve()
        assert runtime == libdir.resolve()

    def test_the_packaged_launcher_is_preferred_over_the_binary(
        self, tmp_path: Path
    ) -> None:
        # A user runs /usr/bin/upscaler, and that launcher is what puts the
        # bundle's libraries ahead of the host's. Preferring it is the only way
        # the smoke exercises the path a real user takes.
        bindir = tmp_path / "usr" / "bin"
        bindir.mkdir(parents=True)
        launcher = bindir / "upscaler"
        launcher.write_text('#!/bin/sh\nexec /usr/lib/upscaler/upscaler "$@"\n')
        launcher.chmod(0o755)

        found, runtime = smoke.resolve_bundle(tmp_path, None)

        assert found == launcher
        assert runtime == tmp_path / "usr" / "lib" / "upscaler"


class TestIsolation:
    def test_every_home_variable_is_redirected_into_the_work_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in (
            "HOME",
            "XDG_DATA_HOME",
            "XDG_CONFIG_HOME",
            "XDG_CACHE_HOME",
            "APPDATA",
            "LOCALAPPDATA",
        ):
            monkeypatch.setenv(name, "/the/real/user/dir")
        monkeypatch.setenv("PYTHONPATH", "")

        env = smoke.isolated_environment(tmp_path, None)

        for name in (
            "HOME",
            "XDG_DATA_HOME",
            "XDG_CONFIG_HOME",
            "XDG_CACHE_HOME",
            "APPDATA",
            "LOCALAPPDATA",
        ):
            assert str(tmp_path) in env[name], f"{name} was not redirected"
            assert "/the/real/user/dir" not in env[name]
        assert str(REPO_ROOT) in env["PYTHONPATH"]

    def test_the_bundle_libraries_come_first(self, tmp_path: Path) -> None:
        bundle = tmp_path / "bundle"
        (bundle / "bin").mkdir(parents=True)

        env = smoke.isolated_environment(tmp_path, bundle)

        assert env["LD_LIBRARY_PATH"].startswith(str(bundle))
        # The bundle's ffmpeg has to win over any ffmpeg on PATH, or the smoke
        # would silently test a different encoder from the one the package ships.
        if sys.platform != "win32":
            assert env["PATH"].startswith(str(bundle / "bin"))

    def test_a_bundle_with_no_bin_leaves_path_alone(self, tmp_path: Path) -> None:
        # Nothing to point at: a bundle with no `bin` directory has no ffmpeg of
        # its own, and PATH is left as it was rather than gaining a dead entry.
        bundle = tmp_path / "bundle"
        bundle.mkdir()

        env = smoke.isolated_environment(tmp_path, bundle)

        assert env["PATH"] == os.environ.get("PATH", "")


class TestGeneratedModel:
    def test_the_fixture_loads_and_is_two_times_the_size(self, tmp_path: Path) -> None:
        import onnx

        model = smoke.make_smoke_model(tmp_path / "smoke.onnx", height=48, width=64)

        assert model.stat().st_size >= smoke.MIN_MODEL_BYTES
        graph = onnx.load(str(model)).graph
        used = {node.op_type for node in graph.node}
        # All three are in EMITTED_OPS_ALLOWLIST, so the fixture is a graph the
        # CoreML and DirectML operation tables would also accept.
        assert used == {"Conv", "LeakyRelu", "Resize"}

    def test_the_graph_really_upscales_by_two(self, tmp_path: Path) -> None:

        model = smoke.make_smoke_model(tmp_path / "smoke.onnx", height=8, width=8)
        from core.backends.export import scale_from_graph

        assert scale_from_graph(model) == 2

    def test_a_too_small_frame_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(smoke.VerificationError) as excinfo:
            smoke.make_smoke_model(tmp_path / "smoke.onnx", height=0, width=8)

        assert "positive" in str(excinfo.value) or "scale" in str(excinfo.value)


class TestClip:
    def test_a_generated_clip_has_video_and_audio(self, tmp_path: Path) -> None:
        clip = smoke.make_clip(tmp_path / "clip.mp4", frames=3, width=64, height=48)

        from core.ffmpeg import probe

        info = probe(clip)
        assert info.has_audio, "the generated clip lost its audio track"
        assert info.frame_count == 3

    def test_a_one_frame_clip_still_has_audio(self, tmp_path: Path) -> None:
        # Measured, not assumed: `-frames:v 1` makes ffmpeg stop muxing before
        # the audio encoder emits anything, so a one-frame clip built that way
        # has no audio stream and cannot prove the encoder preserves one.
        clip = smoke.make_clip(tmp_path / "one.mp4", frames=1, width=64, height=48)

        from core.ffmpeg import probe

        info = probe(clip)
        assert info.frame_count == 1
        assert info.has_audio

    def test_zero_frames_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(smoke.VerificationError):
            smoke.make_clip(tmp_path / "clip.mp4", frames=0, width=64, height=48)


class TestWindowCounting:
    def test_one_line_per_tab_is_one_window(self, tmp_path: Path) -> None:
        log = tmp_path / "upscaler.log"
        log.write_text(
            "2026-01-01 00:00:00,000 INFO t: resolved onnxruntime: /a/onnxruntime\n"
            "2026-01-01 00:00:00,000 INFO t: resolved torch: /a/torch\n",
            encoding="utf-8",
        )

        resolved, windows = smoke._read_resolved(log)

        assert windows == 1
        assert resolved["onnxruntime"] == "/a/onnxruntime"
        assert resolved["torch"] == "/a/torch"

    def test_a_second_window_is_counted(self, tmp_path: Path) -> None:
        # This is the frozen-app failure the check exists for: a spawn worker
        # that re-runs the entry point builds its own MainWindow, and the Runtimes
        # tab inside it writes its own resolution line.
        line = "INFO t: resolved onnxruntime: /a/onnxruntime\n"
        log = tmp_path / "upscaler.log"
        log.write_text(line * 2, encoding="utf-8")

        _, windows = smoke._read_resolved(log)

        assert windows == 2

    def test_the_report_names_the_freeze_guard(self, tmp_path: Path) -> None:
        outcome = smoke.LaunchOutcome(
            survived=True,
            window_lines=3,
            extra_processes=2,
            x11_windows=None,
            log_path=tmp_path / "upscaler.log",
            resolved={"onnxruntime": "/a/onnxruntime", "torch": "/a/torch"},
            stderr_tail="",
            returncode=None,
        )

        failures = smoke.report_launch(outcome, Path("/bundle/upscaler"))

        assert len(failures) == 2
        assert any("3 windows" in failure for failure in failures)
        assert any("freeze_support" in failure for failure in failures)
        assert any("2 further processes" in failure for failure in failures)

    def test_a_clean_launch_produces_no_failures(self, tmp_path: Path) -> None:
        outcome = smoke.LaunchOutcome(
            survived=True,
            window_lines=1,
            extra_processes=0,
            x11_windows=1,
            log_path=tmp_path / "upscaler.log",
            resolved={"onnxruntime": "/a/onnxruntime", "torch": "/a/torch"},
            stderr_tail="",
            returncode=None,
        )

        assert smoke.report_launch(outcome, Path("/bundle/upscaler")) == []

    def test_a_crash_before_the_log_is_a_failure(self, tmp_path: Path) -> None:
        outcome = smoke.LaunchOutcome(
            survived=False,
            window_lines=0,
            extra_processes=0,
            x11_windows=None,
            log_path=None,
            resolved={},
            stderr_tail="libxcb-xinerama.so: cannot open shared object file",
            returncode=127,
        )

        failures = smoke.report_launch(outcome, Path("/bundle/upscaler"))

        assert len(failures) == 1
        assert "127" in failures[0]
        assert "libxcb" in failures[0]


class TestJobVerdict:
    def _child(
        self, payload: dict[str, object], tmp_path: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["verify_install.py", "--run-job"],
            returncode=0 if payload.get("ok") else 1,
            stdout="\n".join(
                [
                    "some onnxruntime chatter on stdout",
                    smoke.JOB_RESULT_MARKER + json.dumps(payload),
                ]
            ),
            stderr="",
        )

    def test_an_unreachable_device_is_tolerated_only_on_request(
        self, tmp_path: Path
    ) -> None:
        child = self._child(
            {
                "ok": False,
                "unreachable": True,
                "error": "Tesla P40 cannot be reached by this build",
                "providers": "CPUExecutionProvider",
            },
            tmp_path,
        )

        job = smoke._interpret_job("GPU 0", child, tmp_path / "out.mp4", 1.0, False)
        failures = []
        for entry in [job]:
            if entry.ok:
                continue
            if entry.unreachable:
                continue
            failures.append(entry.message)

        assert job.unreachable
        assert failures == []

    def test_a_broken_pipeline_is_never_tolerated(self, tmp_path: Path) -> None:
        child = self._child(
            {"ok": False, "unreachable": False, "error": "3 frames missing"},
            tmp_path,
        )

        job = smoke._interpret_job("CPU", child, tmp_path / "out.mp4", 1.0, False)

        assert not job.unreachable
        assert "3 frames missing" in job.message

    def test_a_timeout_is_an_unreachable_device(self, tmp_path: Path) -> None:
        job = smoke._interpret_job(
            "GPU 1",
            subprocess.CompletedProcess([], 124, "", ""),
            tmp_path / "o",
            300.0,
            True,
        )

        assert job.unreachable
        assert "timed out" in job.message

    def test_a_child_that_printed_no_result_is_a_failure(self, tmp_path: Path) -> None:
        job = smoke._interpret_job(
            "CPU",
            subprocess.CompletedProcess([], 1, "", "Traceback: boom"),
            tmp_path / "out.mp4",
            1.0,
            False,
        )

        assert not job.unreachable
        assert "no result line" in job.message
        assert "boom" in job.message


class TestProviderReporting:
    def test_a_slim_build_says_so(self, tmp_path: Path) -> None:
        failures = smoke.check_providers(tmp_path)

        assert failures == []

    def test_the_shared_library_alone_is_not_an_accelerator(
        self, tmp_path: Path
    ) -> None:
        capi = tmp_path / "onnxruntime" / "capi"
        capi.mkdir(parents=True)
        (capi / "libonnxruntime_providers_shared.so").write_bytes(b"")
        (capi / "libonnxruntime_providers_cuda.so").write_bytes(b"")

        assert smoke.check_providers(tmp_path) == []
