from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import core.ffmpeg as ffmpeg_module
from core.errors import UpscalerError
from core.ffmpeg import (
    ENCODE_TMP_SUFFIX,
    FFMPEG_IMAGE_PATTERN,
    encode_video,
    extract_frames,
    find_ffmpeg,
    find_ffprobe,
    probe,
)

WIDTH, HEIGHT, RATE, FRAMES = 64, 48, 25, 25
STAGED = b"what the encoder wrote"


def _staging_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ENCODE_TMP_SUFFIX)


def _ffmpeg_major_version() -> int:
    """The major version of the ffmpeg this run will actually shell out to."""
    banner = subprocess.run(
        [find_ffmpeg(), "-version"], check=True, capture_output=True, text=True
    ).stdout
    return int(banner.split()[2].split(".")[0])


# The fraction of the video's length that must survive the trim, keyed on
# whether the ffmpeg is 5 or newer. Measured on the same 1 s clip: 0.77 s of
# audio on 4.4.2 and 0.91 s on 7.0.2. These are those fractions with a margin,
# not the measurements, because a test that pinned the exact number would fail
# on the next ffmpeg release over a difference nobody acts on.
AUDIO_SURVIVES = {True: 0.85, False: 0.70}


def _streams(path: Path) -> list[dict[str, str]]:
    """What `ffprobe` says a muxed file contains, read without a decode.

    `probe()` insists on a video stream because the pipeline needs one, so it
    cannot be used to look at an output that a broken ffmpeg left without one.
    """
    ffprobe = find_ffprobe()
    if ffprobe is None:
        pytest.skip("no ffprobe on this host: the muxed streams cannot be listed")
    listed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload: dict[str, Any] = json.loads(listed.stdout)
    return list(payload["streams"])


def _stream_types(path: Path) -> set[str]:
    return {stream["codec_type"] for stream in _streams(path)}


def _stream_seconds(path: Path, codec_type: str) -> float:
    for stream in _streams(path):
        if stream["codec_type"] == codec_type:
            return float(stream["duration"])
    raise AssertionError(f"{path.name} has no {codec_type} stream")


def _write_frame(frames_dir: Path) -> Path:
    """A frame carrying a real PNG IHDR and nothing after it.

    The recorder below never runs an encoder, so a whole PNG would be dead
    weight; `encode_video` reads the 24-byte header to learn the size the
    output must have, and a file that were not a PNG would fail there loudly.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / "frame_00000000.png"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + WIDTH.to_bytes(4, "big")
        + HEIGHT.to_bytes(4, "big")
    )
    return path


class _ArgvRecorder:
    """Replaces the ffmpeg runner and keeps every argv it was handed.

    `encode_video` owns the publish, so the argv is the contract: what the
    caller can rely on is which flags ffmpeg is given, and a wrong `-map` is
    how the ported scripts lost the audio of a 165,303-frame render. The
    read-back ffprobe call is answered too, or the publish would never happen.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self,
        argv: list[str],
        *,
        timeout: float = ffmpeg_module.FFMPEG_TIMEOUT_SECONDS,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        del timeout, check
        self.calls.append(argv)
        if argv[0] == find_ffprobe():
            return subprocess.CompletedProcess(argv, 0, self.streams_json(), "")
        Path(argv[-1]).write_bytes(STAGED)
        return subprocess.CompletedProcess(argv, 0, "", "")

    @staticmethod
    def streams_json() -> str:
        return json.dumps(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "width": str(WIDTH),
                        "height": str(HEIGHT),
                    }
                ]
            }
        )

    @property
    def argv(self) -> list[str]:
        """The ffmpeg call, not the read-back ffprobe one."""
        return next(call for call in self.calls if call[0] == find_ffmpeg())


def _build_tone_clip(path: Path) -> Path:
    """A real clip with a real audio stream, encoded by ffmpeg itself."""
    seconds = FRAMES / RATE
    subprocess.run(
        [
            find_ffmpeg(),
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={WIDTH}x{HEIGHT}:rate={RATE}:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=44100:duration={seconds}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _build_silent_clip(path: Path) -> Path:
    """The same clip with no audio track at all."""
    seconds = FRAMES / RATE
    subprocess.run(
        [
            find_ffmpeg(),
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={WIDTH}x{HEIGHT}:rate={RATE}:duration={seconds}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


# --- what the encode command looks like ---------------------------------------


def test_an_encode_with_an_audio_source_maps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames_dir = tmp_path / "frames_out"
    _write_frame(frames_dir)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"the original video")
    out_path = tmp_path / "out.mp4"
    recorder = _ArgvRecorder()
    monkeypatch.setattr(ffmpeg_module, "_run", recorder)

    returned = encode_video(
        frames_dir, _staging_path(out_path), out_path, RATE, 18, source
    )

    assert recorder.argv == [
        find_ffmpeg(),
        "-nostdin",
        "-y",
        "-framerate",
        str(RATE),
        "-i",
        str(frames_dir / FFMPEG_IMAGE_PATTERN),
        "-i",
        str(source),
        "-map",
        "0:v",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-fflags",
        "+shortest",
        str(_staging_path(out_path)),
    ]
    assert returned == out_path
    assert out_path.read_bytes() == STAGED
    assert not _staging_path(out_path).exists()


def test_an_encode_without_an_audio_source_names_only_the_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames_dir = tmp_path / "frames_out"
    _write_frame(frames_dir)
    out_path = tmp_path / "out.mp4"
    recorder = _ArgvRecorder()
    monkeypatch.setattr(ffmpeg_module, "_run", recorder)

    encode_video(frames_dir, _staging_path(out_path), out_path, RATE, 18, None)

    argv = recorder.argv
    assert argv.count("-i") == 1
    assert "1:a?" not in argv
    assert "-c:a" not in argv
    assert argv[argv.index("-map") + 1] == "0:v"
    assert out_path.read_bytes() == STAGED


@pytest.mark.parametrize("fps", [23.976, 25.0, 59.94])
@pytest.mark.parametrize("crf", [0, 18, 51])
def test_the_rate_and_the_crf_reach_ffmpeg_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fps: float, crf: int
) -> None:
    frames_dir = tmp_path / "frames_out"
    _write_frame(frames_dir)
    out_path = tmp_path / "out.mp4"
    recorder = _ArgvRecorder()
    monkeypatch.setattr(ffmpeg_module, "_run", recorder)

    encode_video(frames_dir, _staging_path(out_path), out_path, fps, crf, None)

    argv = recorder.argv
    assert argv[argv.index("-framerate") + 1] == f"{fps:g}"
    assert argv[argv.index("-crf") + 1] == str(crf)
    assert argv[argv.index("-c:v") + 1] == "libx264"
    assert argv[argv.index("-pix_fmt") + 1] == "yuv420p"


def test_ffmpeg_writes_to_the_staging_path_and_the_output_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames_dir = tmp_path / "frames_out"
    _write_frame(frames_dir)
    out_path = tmp_path / "out.mp4"
    staging = _staging_path(out_path)
    recorder = _ArgvRecorder()
    monkeypatch.setattr(ffmpeg_module, "_run", recorder)

    encode_video(frames_dir, staging, out_path, RATE, 18, None)

    assert recorder.argv[-1] == str(staging)
    assert out_path.read_bytes() == STAGED
    assert not staging.exists()


def test_the_read_back_does_not_decode_the_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `-count_frames` here would be a second full decode of a 165,303-frame
    # video; the pipeline has already proved the frame directory complete.
    if find_ffprobe() is None:
        pytest.skip("no ffprobe on this host: there is no read-back to inspect")
    frames_dir = tmp_path / "frames_out"
    _write_frame(frames_dir)
    out_path = tmp_path / "out.mp4"
    recorder = _ArgvRecorder()
    monkeypatch.setattr(ffmpeg_module, "_run", recorder)

    encode_video(frames_dir, _staging_path(out_path), out_path, RATE, 18, None)

    read_back = [call for call in recorder.calls if call[0] == find_ffprobe()]
    assert len(read_back) == 1
    assert "-count_frames" not in read_back[0]


# --- and that it survives a real encode ---------------------------------------


def test_a_real_encode_keeps_the_sources_audio(tmp_path: Path) -> None:
    # The end-to-end form of the same claim. Both ported scripts passed one
    # input and mapped nothing else, so the 165,303-frame result was silent;
    # nothing about an argv assertion alone would have caught that.
    source = _build_tone_clip(tmp_path / "source.mp4")
    assert "audio" in _stream_types(source)
    frames_dir = tmp_path / "frames_out"
    extract_frames(source, frames_dir)
    out_path = tmp_path / "out.mp4"

    encode_video(frames_dir, _staging_path(out_path), out_path, RATE, 18, source)

    assert "audio" in _stream_types(out_path)


def test_the_encoded_video_still_has_its_frames(tmp_path: Path) -> None:
    # Unconditional on purpose: `-fflags +shortest` keeps the picture on every
    # ffmpeg measured, and the read-back in `encode_video` turns a picture-less
    # output into a loud failure rather than a published file.
    source = _build_tone_clip(tmp_path / "source.mp4")
    frames_dir = tmp_path / "frames_out"
    extract_frames(source, frames_dir)
    out_path = tmp_path / "out.mp4"

    encode_video(frames_dir, _staging_path(out_path), out_path, RATE, 18, source)

    encoded = probe(out_path)
    assert (encoded.width, encoded.height) == (WIDTH, HEIGHT)
    assert encoded.frame_count == FRAMES
    assert encoded.has_audio is True


def test_the_audio_is_trimmed_to_the_video_rather_than_padded_past_it(
    tmp_path: Path,
) -> None:
    # `-fflags +shortest` ends the audio with the video. How much of the tail
    # it removes is the one thing that legitimately differs between ffmpeg
    # versions — the copied audio is flushed at a granularity that is not the
    # same in each — so the expectation below is keyed on the version and the
    # two constants are the measured fractions with a margin, not the
    # measurements themselves. What holds on every version is that the track
    # survives, is not padded past the picture, and is not gutted.
    source = _build_tone_clip(tmp_path / "source.mp4")
    frames_dir = tmp_path / "frames_out"
    extract_frames(source, frames_dir)
    out_path = tmp_path / "out.mp4"

    encode_video(frames_dir, _staging_path(out_path), out_path, RATE, 18, source)

    video_seconds = _stream_seconds(out_path, "video")
    audio_seconds = _stream_seconds(out_path, "audio")
    floor = AUDIO_SURVIVES[_ffmpeg_major_version() >= 5]
    assert audio_seconds <= video_seconds + 1 / RATE, "the audio outran the video"
    assert audio_seconds >= video_seconds * floor, (
        f"only {audio_seconds:.3f}s of a {video_seconds:.3f}s video kept its audio; "
        f"expected at least {floor:.0%}"
    )


def test_a_source_with_no_audio_track_does_not_fail_the_encode(
    tmp_path: Path,
) -> None:
    # `-map 1:a?` carries the trailing `?` for this: a source that turns out to
    # have no track must not fail the encode, it must produce a silent one.
    silent = _build_silent_clip(tmp_path / "silent.mp4")
    assert "audio" not in _stream_types(silent)
    frames_dir = tmp_path / "frames_out"
    extract_frames(silent, frames_dir)
    out_path = tmp_path / "out.mp4"

    encode_video(frames_dir, _staging_path(out_path), out_path, RATE, 18, silent)

    assert out_path.is_file()
    assert "audio" not in _stream_types(out_path)
    assert not _staging_path(out_path).exists()


def test_an_encode_that_lost_its_picture_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What ffmpeg 4.4.2 actually did with `-shortest`, turned into the check
    # that stops it: exit 0, a file on disk, and no video in it.
    if find_ffprobe() is None:
        pytest.skip("no ffprobe on this host: the read-back is skipped by design")
    frames_dir = tmp_path / "frames_out"
    _write_frame(frames_dir)
    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(b"the output of the run before this one")
    recorder = _ArgvRecorder()
    monkeypatch.setattr(ffmpeg_module, "_run", recorder)
    monkeypatch.setattr(
        _ArgvRecorder, "streams_json", staticmethod(lambda: '{"streams": []}')
    )

    with pytest.raises(UpscalerError, match="no video stream"):
        encode_video(frames_dir, _staging_path(out_path), out_path, RATE, 18, None)

    assert out_path.read_bytes() == b"the output of the run before this one"
    assert not _staging_path(out_path).exists()
