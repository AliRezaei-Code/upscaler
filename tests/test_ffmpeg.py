from __future__ import annotations

import errno
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import core.ffmpeg as ffmpeg_module
from core.errors import UpscalerError
from core.ffmpeg import (
    ENCODE_TMP_SUFFIX,
    FFMPEG_IMAGE_PATTERN,
    _parse_rate,
    _verify_encoded_video,
    count_frames,
    encode_video,
    extract_frames,
    find_ffmpeg,
    find_ffprobe,
    probe,
)

# Small enough that every test in this file is a fraction of a second, and
# large enough that libx264 still takes the paths a real file takes.
WIDTH, HEIGHT, RATE = 64, 48, 25
FRAMES = 25


def _build_clip(
    path: Path, *, frames: int = FRAMES, rate: int = RATE, with_audio: bool = False
) -> Path:
    """Encode a real clip with ffmpeg itself, so nothing here is a fixture."""
    seconds = frames / rate
    argv = [
        find_ffmpeg(),
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={WIDTH}x{HEIGHT}:rate={rate}:duration={seconds}",
    ]
    if with_audio:
        argv += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=44100:duration={seconds}",
        ]
    argv += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if with_audio:
        argv += ["-c:a", "aac", "-shortest"]
    argv.append(str(path))
    subprocess.run(argv, check=True, capture_output=True)
    return path


def _raw_streams(path: Path) -> list[dict[str, Any]]:
    """ffprobe's own view of a file, so a test can assert on its premise."""
    found = find_ffprobe()
    if found is None:
        pytest.skip("no ffprobe on this host: the premise cannot be shown")
    completed = subprocess.run(
        [
            found,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload: dict[str, Any] = json.loads(completed.stdout)
    streams: list[dict[str, Any]] = payload["streams"]
    return streams


def _staging_path(out_path: Path) -> Path:
    """The staging path the pipeline computes, spelled the same way here."""
    return out_path.with_suffix(out_path.suffix + ENCODE_TMP_SUFFIX)


_A_VIDEO_STREAM_LIST = json.dumps(
    {"streams": [{"codec_type": "video", "width": WIDTH, "height": HEIGHT}]}
)


def _write_frame(
    frames_dir: Path,
    index: int = 0,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> Path:
    """A frame carrying a real PNG IHDR and nothing after it.

    The stubbed runner never decodes anything; `encode_video` reads the
    33-byte header of every frame to learn the size the output must have, and
    a file that were not a PNG would fail there loudly rather than quietly.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / f"frame_{index:08d}.png"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )
    return path


def _stub_run(
    *,
    payload: bytes | None,
    failure: str | None = None,
    streams: str = _A_VIDEO_STREAM_LIST,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """A stand-in for `_run` that writes to the output path it is handed.

    `failure` makes it raise the way the real `_run` does on a non-zero exit,
    which is the only way an encode fails: the staging file ffmpeg itself
    created is on disk, and the caller's cleanup is what has to remove it.
    `streams` is what the read-back ffprobe call is answered with, so a test
    can say what ffmpeg "wrote" without an encoder.
    """

    def run(
        argv: list[str],
        *,
        timeout: float = ffmpeg_module.FFMPEG_TIMEOUT_SECONDS,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        del timeout, check
        if argv[0] == find_ffprobe():
            return subprocess.CompletedProcess(argv, 0, streams, "")
        if payload is not None:
            Path(argv[-1]).write_bytes(payload)
        if failure is not None:
            raise UpscalerError(failure)
        return subprocess.CompletedProcess(argv, 0, "", "")

    return run


# --- find_ffmpeg / find_ffprobe ----------------------------------------------


def test_find_ffmpeg_returns_something_executable() -> None:
    found = find_ffmpeg()
    assert Path(found).is_file()
    assert os.access(found, os.X_OK)


def test_find_ffmpeg_asks_the_path_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def counting_which(name: str) -> str:
        calls.append(name)
        return str(tmp_path / "ffmpeg")

    monkeypatch.setattr(ffmpeg_module, "_ffmpeg_cache", None)
    monkeypatch.setattr(shutil, "which", counting_which)

    assert find_ffmpeg() == find_ffmpeg() == str(tmp_path / "ffmpeg")
    assert calls == ["ffmpeg"]


def test_find_ffprobe_runs_when_it_reports_one() -> None:
    found = find_ffprobe()
    if found is None:
        pytest.skip("this host has no ffprobe at all, which is the fallback case")
    version = subprocess.run(
        [found, "-version"], check=True, capture_output=True, text=True
    )
    assert version.stdout.startswith("ffprobe version")


def test_find_ffprobe_is_none_when_neither_path_nor_ffmpeg_has_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reachable case: imageio-ffmpeg's static ffmpeg ships with no ffprobe
    # beside it, so the ffmpeg-only fallback is not hypothetical.
    lone_ffmpeg = tmp_path / "bin" / "ffmpeg"
    lone_ffmpeg.parent.mkdir()
    lone_ffmpeg.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(ffmpeg_module, "_ffmpeg_cache", str(lone_ffmpeg))
    assert find_ffprobe() is None


def test_find_ffprobe_falls_back_to_the_ffmpeg_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A static build that ships both tools, installed outside PATH.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        sibling = bindir / name
        sibling.write_text("#!/bin/sh\n", encoding="utf-8")
        sibling.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(ffmpeg_module, "_ffmpeg_cache", str(bindir / "ffmpeg"))
    assert find_ffprobe() == str(bindir / "ffprobe")


# --- the frame-rate fraction parser ------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("25/1", 25.0), ("30000/1001", 29.97), ("24", 24.0)],
)
def test_a_fractional_rate_is_read_as_a_number(text: str, expected: float) -> None:
    assert _parse_rate(text) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("text", ["0/0", "N/A", "", "0/1", "25/x", None, 25.0])
def test_an_unusable_rate_falls_through_instead_of_dividing_by_zero(
    text: str | float | None,
) -> None:
    # "0/0" is what an unknown rate looks like; dividing by it would take the
    # whole probe down on a file ffprobe read perfectly well.
    assert _parse_rate(text) is None


# --- probe --------------------------------------------------------------------


def test_probe_reads_the_dimensions_frame_count_and_rate_of_a_clip(
    tmp_path: Path,
) -> None:
    clip = _build_clip(tmp_path / "clip.mp4")
    info = probe(clip)
    assert (info.width, info.height) == (WIDTH, HEIGHT)
    assert info.frame_count == FRAMES
    assert info.fps == float(RATE)
    assert info.duration == pytest.approx(FRAMES / RATE, abs=0.05)


def test_probe_reports_a_silent_clip_as_having_no_audio(tmp_path: Path) -> None:
    assert probe(_build_clip(tmp_path / "silent.mp4")).has_audio is False


def test_probe_reports_the_audio_track_of_a_clip_with_one(tmp_path: Path) -> None:
    clip = _build_clip(tmp_path / "tone.mp4", with_audio=True)
    assert probe(clip).has_audio is True


def test_probe_counts_the_frames_of_a_matroska_that_answers_n_a(
    tmp_path: Path,
) -> None:
    # The regression guard for the ported `int(stream["nb_frames"])`: Matroska
    # does not carry the field, so the count has to come from a real decode.
    matroska = _build_clip(tmp_path / "clip.mkv")
    assert not _raw_streams(matroska)[0].get("nb_frames")

    info = probe(matroska)
    assert info.frame_count == FRAMES
    assert (info.width, info.height) == (WIDTH, HEIGHT)


def test_probe_agrees_with_itself_when_ffprobe_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _build_clip(tmp_path / "clip.mp4", with_audio=True)
    with_ffprobe = probe(clip)
    monkeypatch.setattr(ffmpeg_module, "find_ffprobe", lambda: None)
    without_ffprobe = probe(clip)

    assert without_ffprobe.width == with_ffprobe.width
    assert without_ffprobe.height == with_ffprobe.height
    assert without_ffprobe.frame_count == with_ffprobe.frame_count
    assert without_ffprobe.fps == with_ffprobe.fps
    assert without_ffprobe.has_audio == with_ffprobe.has_audio


def test_probe_rejects_a_file_that_is_not_a_video(tmp_path: Path) -> None:
    not_a_video = tmp_path / "notes.txt"
    not_a_video.write_text("this is not a video", encoding="utf-8")
    with pytest.raises(UpscalerError, match=r"notes\.txt"):
        probe(not_a_video)


def test_probe_rejects_a_file_that_is_not_there(tmp_path: Path) -> None:
    with pytest.raises(UpscalerError, match=r"absent\.mp4"):
        probe(tmp_path / "absent.mp4")


# --- extract_frames -----------------------------------------------------------


def test_extract_frames_numbers_from_zero_and_returns_the_count(
    tmp_path: Path,
) -> None:
    clip = _build_clip(tmp_path / "clip.mp4")
    frames_dir = tmp_path / "frames_in"
    written = extract_frames(clip, frames_dir)

    assert written == FRAMES
    assert (frames_dir / "frame_00000000.png").is_file()
    assert (frames_dir / f"frame_{FRAMES - 1:08d}.png").is_file()
    assert not (frames_dir / f"frame_{FRAMES:08d}.png").exists()
    assert count_frames(frames_dir) == written


def test_extract_frames_keeps_the_sources_own_rate_when_no_fps_is_given(
    tmp_path: Path,
) -> None:
    # 20 frames at 10 fps: a build that quietly substituted a 25 fps default
    # would write 50 and the encode would run at the wrong speed.
    clip = _build_clip(tmp_path / "clip.mp4", frames=20, rate=10)
    assert probe(clip).fps == 10.0
    assert extract_frames(clip, tmp_path / "frames_in") == 20


def test_extract_frames_resamples_when_a_rate_is_given(tmp_path: Path) -> None:
    clip = _build_clip(tmp_path / "clip.mp4", frames=20, rate=10)
    frames_dir = tmp_path / "frames_in"
    assert extract_frames(clip, frames_dir, fps=20) == 40


def test_extract_frames_reports_ffmpegs_own_words_when_the_file_is_absent(
    tmp_path: Path,
) -> None:
    with pytest.raises(UpscalerError) as caught:
        extract_frames(tmp_path / "absent.mp4", tmp_path / "frames_in")
    assert "absent.mp4" in str(caught.value)
    assert "No such file or directory" in str(caught.value)


def test_count_frames_ignores_everything_that_is_not_a_frame(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames_out"
    frames_dir.mkdir()
    for name in ("frame_00000000.png", "frame_00000001.png", "notes.txt", "x.jpg"):
        (frames_dir / name).write_bytes(b"")
    assert count_frames(frames_dir) == 2


# --- encode_video -------------------------------------------------------------


def test_encoding_the_extracted_frames_back_gives_the_same_video(
    tmp_path: Path,
) -> None:
    clip = _build_clip(tmp_path / "clip.mp4")
    frames_dir = tmp_path / "frames_out"
    extract_frames(clip, frames_dir)
    out_path = tmp_path / "out.mp4"

    returned = encode_video(
        frames_dir, _staging_path(out_path), out_path, RATE, 18, None
    )

    assert returned == out_path
    assert not _staging_path(out_path).exists()
    encoded = probe(out_path)
    assert (encoded.width, encoded.height) == (WIDTH, HEIGHT)
    assert encoded.frame_count == FRAMES
    assert encoded.fps == float(RATE)


@pytest.mark.parametrize("crf", [0, 51])
def test_both_ends_of_the_crf_range_encode(tmp_path: Path, crf: int) -> None:
    # CRF 0 is lossless and CRF 51 is the worst legal quality: both are valid
    # x264 settings, and neither is an error for a range check to catch.
    clip = _build_clip(tmp_path / "clip.mp4", frames=10, rate=10)
    frames_dir = tmp_path / "frames_out"
    extract_frames(clip, frames_dir)
    out_path = tmp_path / f"crf{crf}.mp4"

    encode_video(frames_dir, _staging_path(out_path), out_path, 10, crf, None)

    assert probe(out_path).frame_count == 10


def test_frames_that_are_not_all_one_size_are_caught_before_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The case every name-based check in the pipeline misses: a work directory
    # holding frames of two different videos that happen to have the same
    # count passes the resume set, the staleness guard and the completeness
    # check, and the `image2` demuxer would mux the two into one file that
    # changes resolution part way through.
    frames_dir = tmp_path / "frames_out"
    _write_frame(frames_dir, index=0)
    _write_frame(frames_dir, index=1, width=WIDTH * 4, height=HEIGHT * 4)
    _write_frame(frames_dir, index=2)
    out_path = tmp_path / "out.mp4"
    calls: list[list[str]] = []
    monkeypatch.setattr(ffmpeg_module, "_run", _stub_run(payload=b"never written"))

    with pytest.raises(UpscalerError) as caught:
        encode_video(frames_dir, _staging_path(out_path), out_path, RATE, 18, None)

    message = str(caught.value)
    assert "frame_00000001.png" in message
    assert f"{WIDTH * 4}x{HEIGHT * 4}" in message
    assert f"{WIDTH}x{HEIGHT}" in message
    assert calls == [], "ffmpeg was run before the frames were checked"
    assert not out_path.exists()


def test_a_frames_directory_with_nothing_in_it_is_reported(tmp_path: Path) -> None:
    empty_frames = tmp_path / "frames_out"
    empty_frames.mkdir()
    out_path = tmp_path / "out.mp4"

    with pytest.raises(UpscalerError, match="no frames to encode"):
        encode_video(empty_frames, _staging_path(out_path), out_path, RATE, 18, None)

    assert not out_path.exists()


def test_a_partial_staging_file_is_removed_when_ffmpeg_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The case a real failure after a real write produces: ffmpeg got far
    # enough to create the file, then died. The half-written `.part` must not
    # be left where the next run would read it as finished work.
    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(b"previous")
    frames_dir = tmp_path / "frames_out"
    _write_frame(frames_dir)
    monkeypatch.setattr(
        ffmpeg_module,
        "_run",
        _stub_run(payload=b"half an mp4", failure="muxer did not like it"),
    )

    with pytest.raises(UpscalerError, match="muxer did not like it"):
        encode_video(frames_dir, _staging_path(out_path), out_path, RATE, 18, None)

    assert out_path.read_bytes() == b"previous"
    assert not _staging_path(out_path).exists()


def test_a_cross_device_rename_falls_back_to_a_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Never expected — the staging file is required to sit in the output's own
    # directory — but if the publish does fall back, the caller has to hear
    # that the rename was not atomic.
    out_path = tmp_path / "out.mp4"
    messages: list[str] = []
    frames_dir = tmp_path / "frames_out"
    _write_frame(frames_dir)
    monkeypatch.setattr(ffmpeg_module, "_run", _stub_run(payload=b"an encoded video"))
    monkeypatch.setattr(ffmpeg_module, "os", _CrossDeviceOs())

    encode_video(
        frames_dir,
        _staging_path(out_path),
        out_path,
        RATE,
        18,
        None,
        messages.append,
    )

    assert messages == ["os.replace raised EXDEV; fell back to shutil.move"]
    assert out_path.read_bytes() == b"an encoded video"
    assert not _staging_path(out_path).exists()


def test_a_failed_encode_leaves_the_previous_file_untouched(tmp_path: Path) -> None:
    # A frames directory whose first frame is not a decodable image: ffmpeg
    # really runs, really fails, and the output the user already had is still
    # the output they have afterwards.
    previous = b"the output of the run before this one"
    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(previous)
    frames_dir = tmp_path / "frames_out"
    _write_frame(frames_dir)

    with pytest.raises(UpscalerError):
        encode_video(frames_dir, _staging_path(out_path), out_path, RATE, 18, None)

    assert out_path.read_bytes() == previous
    assert not _staging_path(out_path).exists()
    assert not list(tmp_path.glob(f"*{ENCODE_TMP_SUFFIX}"))


@pytest.mark.parametrize(
    ("streams", "message"),
    [
        ('{"streams": []}', "no video stream"),
        (
            json.dumps({"streams": [{"codec_type": "audio"}]}),
            "no video stream",
        ),
        (
            json.dumps(
                {"streams": [{"codec_type": "video", "width": 32, "height": 24}]}
            ),
            "came out 32x24 but the frames are 64x48",
        ),
    ],
)
def test_an_output_that_is_not_the_encoded_frames_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, streams: str, message: str
) -> None:
    # ffmpeg exiting 0 is not proof of a video. This is the check that turns a
    # picture-less or wrong-sized output into a loud failure with the previous
    # file still in place. The read-back needs an ffprobe; the host without one
    # is covered by `test_the_read_back_is_skipped_when_there_is_no_ffprobe`.
    if find_ffprobe() is None:
        pytest.skip("no ffprobe on this host: the read-back is skipped by design")
    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(b"the output of the run before this one")
    frames_dir = tmp_path / "frames_out"
    _write_frame(frames_dir)
    monkeypatch.setattr(
        ffmpeg_module, "_run", _stub_run(payload=b"whatever", streams=streams)
    )

    with pytest.raises(UpscalerError, match=re.escape(message)):
        encode_video(frames_dir, _staging_path(out_path), out_path, RATE, 18, None)

    assert out_path.read_bytes() == b"the output of the run before this one"
    assert not _staging_path(out_path).exists()


def test_the_read_back_is_skipped_when_there_is_no_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The ffmpeg-only fallback would have to decode the file to learn what is
    # in it, which is the one cost the read-back exists to avoid, so on such a
    # host the encode still publishes and exactly one process is run.
    out_path = tmp_path / "out.mp4"
    frames_dir = tmp_path / "frames_out"
    _write_frame(frames_dir)
    calls: list[list[str]] = []
    stub = _stub_run(payload=b"a video")

    def counting_run(
        argv: list[str],
        *,
        timeout: float = ffmpeg_module.FFMPEG_TIMEOUT_SECONDS,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return stub(argv, timeout=timeout, check=check)

    monkeypatch.setattr(ffmpeg_module, "find_ffprobe", lambda: None)
    monkeypatch.setattr(ffmpeg_module, "_run", counting_run)

    encode_video(frames_dir, _staging_path(out_path), out_path, RATE, 18, None)

    assert len(calls) == 1
    assert calls[0][0] == find_ffmpeg()
    assert out_path.read_bytes() == b"a video"


class _CrossDeviceOs:
    """`os` with `replace` rigged to fail the way a cross-device rename does."""

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)

    @staticmethod
    def replace(src: str, dst: str) -> None:
        del src, dst
        raise OSError(errno.EXDEV, "Invalid cross-device link")


def test_the_image_pattern_is_what_both_sides_agree_on() -> None:
    # The extraction naming and the encode's input pattern are one constant on
    # purpose: two spellings would give a silent, zero-frame encode.
    assert FFMPEG_IMAGE_PATTERN == "frame_%08d.png"
    assert ENCODE_TMP_SUFFIX == ".part.mp4"


def test_a_short_encode_is_refused(tmp_path: Path) -> None:
    """The residual ffmpeg failure: exit 0, a playable file, too few frames.

    ffmpeg's `image2` demuxer skips files it cannot read and keeps going, so a
    directory of five frames with unreadable ones yields a shorter video and no
    error at all. `_checked_frame_size` refuses the directory before the encode
    starts; this covers the other route to the same outcome, where the frames
    are fine and the demuxer still loses some.
    """
    from core.ffmpeg import find_ffprobe
    from tests.support import write_frame

    if find_ffprobe() is None:
        # The read-back is skipped without ffprobe, because the alternative is
        # a full decode of the output - which is the one cost it exists to
        # avoid. imageio-ffmpeg ships ffmpeg and no ffprobe, so this is the
        # Windows runner's situation, and the test has no premise there.
        pytest.skip("no ffprobe on this host, so the encode read-back is skipped")

    frames = tmp_path / "frames"
    write_frame(frames / "frame_00000000.png")
    write_frame(frames / "frame_00000001.png")
    write_frame(frames / "frame_00000002.png")
    out = encode_video(
        frames, tmp_path / "out.part.mp4", tmp_path / "out.mp4", 10, 30, None
    )

    # Three frames of video where five were expected: a playable file, the
    # right dimensions, and no error from ffmpeg.
    with pytest.raises(UpscalerError, match="fewer frames"):
        _verify_encoded_video(out, (96, 64), 5, 10)


def test_a_correct_encode_passes_the_same_check(tmp_path: Path) -> None:
    from core.ffmpeg import find_ffprobe
    from tests.support import write_frame

    if find_ffprobe() is None:
        pytest.skip("no ffprobe on this host, so the encode read-back is skipped")

    frames = tmp_path / "frames"
    for index in range(4):
        write_frame(frames / f"frame_{index:08d}.png")
    out = encode_video(
        frames, tmp_path / "out.part.mp4", tmp_path / "out.mp4", 10, 30, None
    )
    _verify_encoded_video(out, (96, 64), 4, 10)
