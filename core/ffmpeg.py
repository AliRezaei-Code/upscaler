"""Finding, probing and running ffmpeg.

Three behaviours in this module exist because the two hand-written scripts this
application replaces got them wrong, and each mistake cost a long run:

* `probe` counts frames by **decoding** the file. `nb_frames` is `"N/A"` for
  Matroska and for many MP4s, and the ported `get_video_info` did
  `int(video_stream.get("nb_frames", 0))`, which raises `ValueError` on
  `"N/A"` and yields `0` when the key is absent. A frame count of `0` makes the
  pipeline extract nothing and then encode nothing, silently.
* `encode_video` maps the source's audio. Both ported scripts passed a single
  `-i` and mapped nothing else, so the 165,303-frame result was a silent video.
* `encode_video` publishes atomically. An encode that dies half way through
  165,303 frames must leave the previous output exactly as it was, not a
  truncated one.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import UpscalerError

FFMPEG_IMAGE_PATTERN = "frame_%08d.png"
ENCODE_TMP_SUFFIX = ".part.mp4"

# Long on purpose: a 165,303-frame source takes hours to decode or encode, and
# the point of the ceiling is to stop a wedged binary from hanging the app
# forever rather than to police normal work.
FFMPEG_TIMEOUT_SECONDS = 12 * 3600.0
# The post-encode read-back is a header parse, not a decode, so a minute is
# already generous; anything slower is a hung ffprobe and should be reported.
STRUCTURAL_CHECK_TIMEOUT_SECONDS = 60.0
# The tail of a failure is what actually says what went wrong; ffmpeg's banner
# and input listing push the real error off the end of the screen.
STDERR_TAIL_CHARS = 500
# A PNG's dimensions are in its IHDR, which ends 33 bytes in: 8 of signature,
# 4 of length, 4 of type, 4 of width, 4 of height, 9 of the rest of the chunk.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_HEADER_BYTES = 33

# The `-count_frames` flag is what makes the count exact, and it costs a full
# decode pass. So does the ffmpeg-only fallback below: the number is trusted
# precisely because nothing cheap was trusted instead.
_PROBE_ARGS = (
    "-v",
    "error",
    "-count_frames",
    "-show_entries",
    "stream=codec_type,width,height,nb_read_frames,nb_frames,"
    "avg_frame_rate,r_frame_rate",
    "-show_entries",
    "stream_tags=duration",
    "-show_entries",
    "format=duration",
    "-of",
    "json",
)

_FRAME_COUNT_RE = re.compile(r"\bframe=\s*(\d+)")
_DIMENSIONS_RE = re.compile(r"(?<![\w.])(\d{2,5})x(\d{2,5})(?![\w.])")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([kM]?)\s*fps\b")
_TBR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([kM]?)\s*tbr\b")
_MULTIPLIERS = {"": 1.0, "k": 1_000.0, "M": 1_000_000.0}


@dataclass(frozen=True)
class VideoInfo:
    """What the pipeline needs to know about a source before touching a frame."""

    width: int
    height: int
    frame_count: int
    fps: float
    duration: float
    has_audio: bool


_ffmpeg_cache: str | None = None


def find_ffmpeg() -> str:
    """The ffmpeg to run, preferring the one on `PATH`.

    `imageio-ffmpeg`'s bundled static build is the fallback so a fresh checkout
    with no system ffmpeg can still run a job. The answer is cached after the
    first success because a job shells out to ffmpeg thousands of times.
    """
    global _ffmpeg_cache
    if _ffmpeg_cache is not None:
        return _ffmpeg_cache

    on_path = shutil.which("ffmpeg")
    if on_path is not None:
        _ffmpeg_cache = on_path
        return on_path

    # Imported here, not at module scope, so that a checkout missing
    # imageio-ffmpeg still imports `core.ffmpeg` and fails with a message
    # naming the package rather than an ImportError during start-up.
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - imageio-ffmpeg is declared
        raise UpscalerError(
            "ffmpeg not found on PATH and no bundled copy is available: "
            "install ffmpeg or the imageio-ffmpeg package"
        ) from exc

    bundled = str(imageio_ffmpeg.get_ffmpeg_exe())
    if not Path(bundled).is_file():
        raise UpscalerError(
            f"ffmpeg not found on PATH and the bundled copy at {bundled} is missing"
        )
    _ffmpeg_cache = bundled
    return bundled


def find_ffprobe() -> str | None:
    """The ffprobe to run, or `None` when there is none.

    This returns `None` rather than raising on purpose. `probe()` has a real
    fallback that decodes the file with ffmpeg instead, so a missing ffprobe
    costs one extra decode pass and some parsing, not a refusal to open a file.
    That fallback is reachable: `imageio-ffmpeg`'s bundled ffmpeg ships with no
    ffprobe beside it.

    Not cached, unlike `find_ffmpeg`: resolving it is two stat calls, and a
    cached `None` would hide a runtime that gets installed later in the session.
    """
    on_path = shutil.which("ffprobe")
    if on_path is not None:
        return on_path
    sibling = Path(find_ffmpeg()).with_name("ffprobe")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return None


def probe(path: Path) -> VideoInfo:
    """Read a source video's dimensions, frame count, rate and audio flag.

    This always decodes the whole file: `ffprobe -count_frames` on the ffprobe
    path, and a `ffmpeg -f null -` decode pass on the fallback. That is the
    price of an exact frame count, and it is worth it — the frame count drives
    the resume set, the disk preflight and the completeness check, and every
    cheaper source for it is either absent or `"N/A"` in exactly the containers
    a user is likely to open.
    """
    ffprobe = find_ffprobe()
    if ffprobe is None:
        return _probe_with_ffmpeg(path)
    return _probe_with_ffprobe(ffprobe, path)


def extract_frames(video: Path, frames_dir: Path, *, fps: float | None = None) -> int:
    """Decode `video` into `frames_dir` as `frame_%08d.png`, numbered from 0.

    `fps=None` adds no `-r`, so the frames carry the source's own rate; the
    encode must then be given that same rate, which is why `probe` reports it.

    Returns the number of PNGs in the directory afterwards. Re-running over a
    directory that already holds frames is the caller's business: the count is
    a directory count, so a stale directory over-reports rather than
    under-reports, and the pipeline's completeness check is what catches a
    mismatch.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    argv = [find_ffmpeg(), "-nostdin", "-y", "-i", str(video)]
    if fps is not None:
        argv += ["-r", f"{fps:g}"]
    argv += ["-start_number", "0", str(frames_dir / FFMPEG_IMAGE_PATTERN)]
    _run(argv)

    first = frames_dir / "frame_00000000.png"
    if not first.is_file():
        raise UpscalerError(
            f"ffmpeg reported success but wrote no frames starting at {first}; "
            f"the directory does not look like an extraction of {video.name}"
        )
    return count_frames(frames_dir)


def encode_video(
    frames_dir: Path,
    tmp_path: Path,
    out_path: Path,
    fps: float,
    crf: int,
    audio_source: Path | None,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Encode the frames to `out_path`, preserving `audio_source`'s audio.

    The caller supplies **both** `tmp_path` and `out_path`; this function owns
    the whole publish. `tmp_path` is required to sit in the output's own
    directory — the pipeline computes it as
    `out_path.with_suffix(out_path.suffix + ENCODE_TMP_SUFFIX)` — so the final
    rename is a same-filesystem `os.replace` and `EXDEV` cannot happen. Keeping
    the decision with the caller and the durability with this function is what
    means the fsync exists exactly once.

    The audio block is present only when `audio_source` is given. Both ported
    scripts passed a single input and mapped nothing else, so a 165,303-frame
    result was a silent video; `-map 1:a?` with its trailing `?` is what keeps
    the track without failing when the source turns out to have none.

    The audio is trimmed to the video with `-fflags +shortest`, and that is
    **not** the more obvious `-shortest`. Measured on this host's ffmpeg 4.4.2,
    10 frames at 10 fps against a 1 s sine, the same command three ways:

    * `-shortest`         -> `audio,45`  — exit 0, **no video track at all**
    * `-fflags +shortest` -> `video,10 audio,33`
    * no flag             -> `video,10 audio,45`

    The output flag races the stream-copied audio and the video loses, and it
    loses differently depending on how fast the encoder runs: the same 4.4.2
    kept 5 of 10 frames at `-preset ultrafast` and 49 of 50 at 2 s. 7.0.2
    happens to keep the picture either way, so the bug only bites on the
    version that happens to be installed. The input flag stops at the end of
    the shorter *stream* without tearing the video out, on both, and the audio
    tail it trims is the right semantic anyway: the frames are the authority
    on how long the result is. Do not "simplify" this back to `-shortest`.
    """
    argv = [
        find_ffmpeg(),
        "-nostdin",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-i",
        str(frames_dir / FFMPEG_IMAGE_PATTERN),
    ]
    if audio_source is not None:
        argv += ["-i", str(audio_source)]
    argv += ["-map", "0:v"]
    if audio_source is not None:
        argv += ["-map", "1:a?"]
    argv += [
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
    ]
    if audio_source is not None:
        argv += ["-c:a", "copy"]
    argv += ["-fflags", "+shortest"]

    # Before the encode, not after: a directory that is not all one size is
    # worth finding out about before spending hours of encoding on it.
    expected = _checked_frame_size(frames_dir)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    argv.append(str(tmp_path))
    try:
        _run(argv)
        if not tmp_path.is_file():
            raise UpscalerError(
                f"ffmpeg exited 0 but wrote no output at {tmp_path}; "
                f"the encode of {frames_dir} produced nothing"
            )
        # ffmpeg exiting 0 is not the same as having written a video — see the
        # measurement at the top of this docstring — so the staged file is
        # read back before it is allowed to become the user's output.
        _verify_encoded_video(tmp_path, expected, count_frames(frames_dir), fps)
    except UpscalerError:
        # A `.part` left behind is indistinguishable from a completed encode on
        # the next run, and a half-written one can be mistaken for a good
        # output. `out_path` is untouched either way, because nothing has
        # written to it yet.
        tmp_path.unlink(missing_ok=True)
        raise

    try:
        os.replace(tmp_path, out_path)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        # Not expected — see the docstring — but `shutil.move` degrades to a
        # copy, which is better than losing the encode, and the caller should
        # hear that the durability guarantee is weaker.
        shutil.move(str(tmp_path), str(out_path))
        _report(log, "os.replace raised EXDEV; fell back to shutil.move")
    _fsync_directory(out_path.parent)
    return out_path


def count_frames(frames_dir: Path) -> int:
    """How many `frame_*.png` files the directory holds."""
    return sum(1 for _ in frames_dir.glob("frame_*.png"))


def _checked_frame_size(frames_dir: Path) -> tuple[int, int]:
    """The pixel size the frames in `frames_dir` are, and proof they all agree.

    Read from the files rather than taken as a parameter, because the
    directory is the only statement the encode has of the output it must
    produce.

    Every frame is checked, not just the first, because that is the case the
    rest of the pipeline cannot see. The resume set and the staleness guard
    both compare *names*, so a work directory holding 500 frames of one video
    and 500 of another — same count, same `frame_0000000x.png` spelling —
    passes both, and the entry count matches too. ffmpeg's `image2` demuxer
    reads what it can, so the result would be one file that changes
    resolution part way through.

    Affordable at 165,000 frames because a PNG header is 33 bytes at a fixed
    offset: one small read per file and no decode of any of them. Measured at
    20 microseconds a frame on this filesystem, so 3.2 s for a 165,303-frame
    job, against the hours the encode itself takes.
    """
    first = frames_dir / "frame_00000000.png"
    if not first.is_file():
        remaining = sorted(frames_dir.glob("frame_*.png"))
        if not remaining:
            raise UpscalerError(f"{frames_dir} holds no frames to encode")
        first = remaining[0]
    expected = _png_dimensions(first)
    for frame in frames_dir.glob("frame_*.png"):
        if frame == first:
            continue
        found = _png_dimensions(frame)
        if found != expected:
            raise UpscalerError(
                f"{frame} is {found[0]}x{found[1]} but {first.name} is "
                f"{expected[0]}x{expected[1]}: the frames in {frames_dir} are not "
                f"all one size, and ffmpeg would mux them into a single file"
            )
    return expected


def _png_dimensions(path: Path) -> tuple[int, int]:
    """The width and height in a PNG's IHDR, from its first 33 bytes."""
    with path.open("rb") as handle:
        header = handle.read(_PNG_HEADER_BYTES)
    if header[:8] != _PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise UpscalerError(
            f"{path} is not a PNG, so the size the encode must produce is unknown"
        )
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


#: How far the encoded duration may differ from `frames / fps` before the
#: output is called short. One frame plus a margin for container timestamp
#: rounding; anything more is a real truncation.
DURATION_TOLERANCE_SECONDS = 0.05


def _verify_encoded_video(
    output: Path,
    expected: tuple[int, int],
    expected_frames: int,
    fps: float,
) -> None:
    """Refuse to publish a file ffmpeg called a success that is not a video.

    Two cheap structural checks, neither of which decodes the output:

    * the output has a video stream, and its dimensions are the frames' own —
      ffmpeg 4.4.2 with `-shortest` and `-c:a copy` exits **0** having written
      an audio-only file, which is a silent success and the reason this
      function exists at all;
    * its duration is `expected_frames / fps`, to within one frame. This is
      what catches ffmpeg's `image2` demuxer quietly skipping files: measured
      here, a directory of five frames where three are unreadable produces a
      two-frame video and exit code 0. `-count_frames` would give the number
      exactly, but it is a full decode of a 165,303-frame video, and the
      container duration answers the same question in one header.

    Skipped when there is no ffprobe, because the ffmpeg-only fallback would
    have to decode the file to learn what is in it, which is the one cost this
    check exists to avoid. That host has already paid a decode in `probe()`.
    """
    ffprobe = find_ffprobe()
    if ffprobe is None:
        return
    completed = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,duration",
            "-of",
            "json",
            str(output),
        ],
        timeout=STRUCTURAL_CHECK_TIMEOUT_SECONDS,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise UpscalerError(
            f"{output.name} is not a file ffprobe can read: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise UpscalerError(
            f"{output.name} made ffprobe return {type(payload).__name__}, "
            f"not a stream list"
        )
    video = _first_stream(payload.get("streams"), "video")
    if video is None:
        raise UpscalerError(
            f"{output.name} has no video stream: ffmpeg exited 0 but wrote a "
            f"file that cannot be played, and it was not published"
        )
    size = (_positive_int(video.get("width")), _positive_int(video.get("height")))
    if size != expected:
        raise UpscalerError(
            f"{output.name} came out {size[0]}x{size[1]} but the frames are "
            f"{expected[0]}x{expected[1]}"
        )
    duration = _float_or_none(video.get("duration"))
    if duration is None or fps <= 0:
        return
    wanted = expected_frames / fps
    if abs(duration - wanted) > 1.0 / fps + DURATION_TOLERANCE_SECONDS:
        raise UpscalerError(
            f"{output.name} runs {duration:.3f}s but {expected_frames} frames at "
            f"{fps:.3f} fps is {wanted:.3f}s; ffmpeg published fewer frames than "
            f"the directory held and it was not written to {output}"
        )


# --- the ffprobe path ---------------------------------------------------------


def _probe_with_ffprobe(ffprobe: str, path: Path) -> VideoInfo:
    completed = _run([ffprobe, *_PROBE_ARGS, str(path)])
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise UpscalerError(
            f"ffprobe did not return JSON for {path.name}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise UpscalerError(
            f"ffprobe returned {type(payload).__name__}, not an object, for {path.name}"
        )

    streams = payload.get("streams")
    video = _first_stream(streams, "video")
    if video is None:
        raise UpscalerError(f"{path.name} has no video stream to upscale")

    width = _positive_int(video.get("width"))
    height = _positive_int(video.get("height"))
    if width is None or height is None:
        raise UpscalerError(
            f"ffprobe reported no usable dimensions for {path.name}: "
            f"width={video.get('width')!r} height={video.get('height')!r}"
        )

    fps = _frame_rate(video) or 0.0
    duration = _container_duration(video, payload)
    frame_count = _frame_count(video, fps, duration, path)
    return VideoInfo(
        width=width,
        height=height,
        frame_count=frame_count,
        fps=fps,
        duration=duration,
        has_audio=_first_stream(streams, "audio") is not None,
    )


def _frame_count(video: dict[str, Any], fps: float, duration: float, path: Path) -> int:
    # `nb_read_frames` is what `-count_frames` just counted by decoding the
    # file, so it is exact. `nb_frames` is a container field that is absent or
    # `"N/A"` often enough that a bare `int()` on it is a crash or a zero.
    for key in ("nb_read_frames", "nb_frames"):
        counted = _positive_int(video.get(key))
        if counted is not None:
            return counted
    if duration > 0.0 and fps > 0.0:
        return round(duration * fps)
    raise UpscalerError(
        f"Could not determine the frame count of {path.name}: "
        f"ffprobe reported nb_read_frames={video.get('nb_read_frames')!r} "
        f"nb_frames={video.get('nb_frames')!r} duration={duration!r} fps={fps!r}"
    )


# --- the ffmpeg-only fallback -------------------------------------------------


def _probe_with_ffmpeg(path: Path) -> VideoInfo:
    """Probe by decoding with ffmpeg, for a host with no ffprobe anywhere.

    Both halves are attempted even when the first one fails, so a file this
    cannot read produces one error naming both failures rather than the first
    of them.
    """
    failures: list[str] = []

    # `ffmpeg -i` with no output always exits non-zero, so its exit code says
    # nothing; only the stream listing on stderr is read.
    listing = _run([find_ffmpeg(), "-nostdin", "-i", str(path)], check=False)
    video_line = _video_stream_line(listing.stderr)
    size: tuple[int, int] | None = None
    fps = 0.0
    if video_line is None:
        failures.append(f"ffmpeg printed no video stream line: {_tail(listing.stderr)}")
    else:
        width, height = _dimensions_from(video_line)
        fps = _fps_from(video_line)
        if width is None or height is None:
            failures.append(
                f"ffmpeg's stream line {video_line.strip()!r} carries no usable "
                f"dimensions"
            )
        else:
            size = (width, height)

    frames = 0
    try:
        decode = _run(
            [
                find_ffmpeg(),
                "-nostdin",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ]
        )
        counts = _FRAME_COUNT_RE.findall(decode.stderr)
        if counts:
            frames = int(counts[-1])
        else:
            failures.append(
                f"the decode pass printed no frame count: {_tail(decode.stderr)}"
            )
    except UpscalerError as exc:
        failures.append(f"the decode pass failed: {exc}")

    if size is None or frames <= 0:
        raise UpscalerError(
            f"Could not read {path.name} without ffprobe: {'; '.join(failures)}"
        )

    return VideoInfo(
        width=size[0],
        height=size[1],
        frame_count=frames,
        fps=fps,
        duration=frames / fps if fps > 0.0 else 0.0,
        has_audio=any(": Audio:" in line for line in listing.stderr.splitlines()),
    )


def _video_stream_line(stderr: str) -> str | None:
    for line in stderr.splitlines():
        if ": Video:" in line and line.lstrip().startswith("Stream"):
            return line
    return None


def _dimensions_from(line: str) -> tuple[int | None, int | None]:
    for match in _DIMENSIONS_RE.finditer(line):
        width, height = int(match.group(1)), int(match.group(2))
        if 1 <= width <= 16_384 and 1 <= height <= 16_384:
            return width, height
    return None, None


def _fps_from(line: str) -> float:
    # `fps` is the rate the stream claims; `tbr` is what the timebase implies.
    # The 4:3 and 1:1 in a DAR tag are already excluded by the pattern.
    for pattern in (_FPS_RE, _TBR_RE):
        match = pattern.search(line)
        if match is not None:
            return float(match.group(1)) * _MULTIPLIERS[match.group(2)]
    return 0.0


# --- shared parsing -----------------------------------------------------------


def _first_stream(streams: Any, codec_type: str) -> dict[str, Any] | None:
    if not isinstance(streams, list):
        return None
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == codec_type:
            return stream
    return None


def _float_or_none(value: Any) -> float | None:
    """A number from ffprobe's JSON, or `None` for `"N/A"` and friends."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        number = int(value)
    except ValueError:
        return None
    return number if number > 0 else None


def _frame_rate(video: dict[str, Any]) -> float | None:
    """The stream's rate, preferring the average over the container's nominal."""
    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = _parse_rate(video.get(key))
        if rate is not None:
            return rate
    return None


def _parse_rate(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return _positive_float(numerator)
    top = _positive_float(numerator)
    bottom = _positive_float(denominator)
    if top is None or bottom is None or bottom <= 0.0:
        # "0/0" is what an unknown rate looks like; divide by it and the whole
        # probe dies on a video that is perfectly readable.
        return None
    return top / bottom


def _positive_float(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if number > 0.0 else None


def _container_duration(video: dict[str, Any], payload: dict[str, Any]) -> float:
    fmt = payload.get("format")
    if isinstance(fmt, dict):
        duration = _positive_float(fmt.get("duration", ""))
        if duration is not None:
            return duration
    tags = video.get("tags")
    if isinstance(tags, dict):
        # Matroska puts the duration in a timecode tag, not in the format.
        for key in ("DURATION", "duration"):
            raw = tags.get(key)
            if isinstance(raw, str):
                seconds = _timecode_seconds(raw)
                if seconds is not None:
                    return seconds
    return 0.0


def _timecode_seconds(text: str) -> float | None:
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (float(part) for part in parts)
    except ValueError:
        return None
    return hours * 3600.0 + minutes * 60.0 + seconds


# --- process plumbing ---------------------------------------------------------


def _run(
    argv: list[str],
    *,
    timeout: float = FFMPEG_TIMEOUT_SECONDS,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg/ffprobe argv, capturing both streams.

    The exit status is inspected here rather than with `check=True` so that the
    one caller which expects a non-zero status (`ffmpeg -i` with no output, for
    the stream listing) can ask for `check=False` and still get the same error
    text for the callers that do want a failure.
    """
    tool = Path(argv[0]).name
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise UpscalerError(
            f"{tool} did not finish within {timeout:.0f}s: {shlex.join(argv)}\n"
            f"{_tail(_as_text(exc.stderr))}"
        ) from exc
    except FileNotFoundError as exc:
        raise UpscalerError(f"{tool} is not executable: {argv[0]}") from exc
    if check and completed.returncode != 0:
        raise UpscalerError(
            f"{tool} exited {completed.returncode}: {shlex.join(argv)}\n"
            f"{_tail(completed.stderr)}"
        )
    return completed


def _as_text(value: str | bytes | None) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _tail(stream: str | None) -> str:
    if not stream:
        return "(ffmpeg printed nothing)"
    return stream[-STDERR_TAIL_CHARS:].strip()


def _report(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)


def _fsync_directory(directory: Path) -> None:
    """Make a rename in `directory` survive a crash.

    The data is already fsynced by ffmpeg closing the file; without this the
    directory entry for the new name can still be lost on a power cut, leaving
    a file the pipeline recorded as finished and that is not there.
    """
    if os.name == "nt":
        # Windows has no way to open a directory for fsync; the rename is
        # ordered by the filesystem and this is the documented limit there.
        return
    handle = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)
