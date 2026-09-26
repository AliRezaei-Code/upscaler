"""The model downloader: resumable, verified, and never half-committed.

A model is tens to hundreds of megabytes, and the two hand-written scripts this
application replaces both re-downloaded from zero on every failure. Three
properties fix that:

1. **A half-written file is never mistaken for a model.** Every byte lands in
   a `.part` file beside `dest` — `RealESRGAN_x4plus.pth.part` — and the
   rename to `dest` happens only after verification, so `models_dir()` can
   never contain a truncated `.pth`.
2. **A dropped connection costs a range, not the whole file.** The `.part`
   size becomes the `Range` offset of the next attempt, so a retry resumes
   where the last one stopped instead of re-fetching what it already has. A
   server that answers `200` to a `Range` request is ignoring the range, and
   appending to that response would produce two copies of the body, so the
   transfer restarts from zero instead.
3. **A finished download is proven before it is published.** A wrong size or a
   wrong SHA-256 raises `DownloadError`; a hash mismatch deletes the `.part`,
   because a corrupt prefix cannot be repaired by resuming onto it.

`retries` counts the attempts *after* the first, so the default of 3 means at
most four requests, spaced by `BACKOFF_BASE_SECONDS` doubling from 0.5 s.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

from .errors import DownloadError
from .events import PipelineEvent
from .models import hash_file

# One place decides how this application introduces itself to a model host.
USER_AGENT = "upscaler/0.1.0 (+https://github.com/AliRezaei-Code/upscaler)"

BACKOFF_BASE_SECONDS = 0.5
BACKOFF_MAX_SECONDS = 8.0
PROGRESS_INTERVAL_SECONDS = 0.25

# A host that is having a bad day is worth another attempt; a host saying the
# resource does not exist is not, and retrying it only delays the error.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class _IncompleteBody(Exception):
    """The connection ended before the promised number of bytes arrived."""


class _RetryableStatus(Exception):
    """The server answered with a status that is worth another attempt."""


@dataclass(frozen=True)
class DownloadResult:
    """What one call to `download_file` actually transferred.

    `resumed_from + bytes_written` is always the size of `path`. `resumed_from`
    is the offset the transfer that produced the result began at, so it is `0`
    both for a fresh download and for one where the server ignored the `Range`
    request and the existing `.part` had to be discarded.
    """

    path: Path
    bytes_written: int
    resumed_from: int


def part_path(dest: Path) -> Path:
    """Where `dest` is written before it is verified.

    The suffix is appended rather than replaced, so `RealESRGAN_x4plus.pth`
    stages as `RealESRGAN_x4plus.pth.part` and sorts next to the model it will
    become. Note that the project's `.gitignore` rule is the literal
    `models/.part`, which does not match this name: a `.part` file that outlives
    its download is untracked only if the rule is widened to `models/*.part`.
    """
    return dest.with_suffix(dest.suffix + ".part")


def _emit(emit: Callable[[PipelineEvent], None] | None, event: PipelineEvent) -> None:
    """Send one event, if the caller wants events at all."""
    if emit is not None:
        emit(event)


def _expected_total(
    expected_size: int | None, content_length: str | None, resumed_from: int
) -> int | None:
    """Total size the finished file must have, or `None` when nobody knows.

    `Content-Length` on a `206` counts only the bytes still to come, so the
    offset has to be added back; on a `200` the offset is already `0`.
    """
    if expected_size is not None:
        return expected_size
    if content_length is None:
        return None
    return resumed_from + int(content_length)


def _transfer(
    url: str,
    part: Path,
    task_id: str,
    emit: Callable[[PipelineEvent], None] | None,
    chunk_size: int,
    timeout: float,
    expected_size: int | None,
) -> tuple[int, int]:
    """One HTTP attempt. Returns `(resumed_from, bytes_written)`.

    The `.part` is left in place whatever happens, which is what makes the next
    attempt resume rather than restart.
    """
    existing = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    with requests.Session() as session:
        session.headers["User-Agent"] = USER_AGENT
        with session.get(url, stream=True, timeout=timeout, headers=headers) as resp:
            status = resp.status_code
            if status == 416:
                # The part file already covers the whole resource: there is
                # nothing left to ask for, so let the caller verify it.
                return existing, 0
            if status in _RETRYABLE_STATUS:
                raise _RetryableStatus(f"{url} answered HTTP {status}")
            if status == 206:
                mode = "ab"
            elif status == 200:
                # A 200 in reply to a Range means the offset was ignored. Keep
                # the bytes and the second copy of the body and the model is
                # silently doubled, so throw the prefix away and start again.
                mode = "wb"
                existing = 0
            else:
                raise DownloadError(f"{url} answered HTTP {status}")

            total = _expected_total(
                expected_size, resp.headers.get("Content-Length"), existing
            )
            started = time.monotonic()
            last_emit: float | None = None
            written = 0

            def report() -> None:
                """Send this attempt's progress; closes over the loop's counters."""
                # Windows' `time.monotonic` has about 15 ms of resolution, so a
                # fast local transfer can start and finish inside one tick and
                # report 0 B/s - a progress bar that stops at the beginning
                # rather than at the end. A millisecond floor is short enough
                # to be meaningless and long enough to be a rate.
                elapsed = max(time.monotonic() - started, 1e-3)
                rate = written / elapsed
                done = existing + written
                _emit(
                    emit,
                    PipelineEvent(
                        kind="device_progress",
                        task_id=task_id,
                        processed=done,
                        # total stays 0 when nobody knows the size, and a
                        # front-end has to render that bar as indeterminate.
                        total=total if total is not None else 0,
                        # For a download `fps` carries bytes per second; the
                        # field is shared with the frame counters of a job.
                        fps=rate,
                        eta_seconds=(total - done) / rate if total and rate else 0.0,
                    ),
                )

            with part.open(mode) as handle:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
                    if (
                        last_emit is None
                        or time.monotonic() - last_emit >= PROGRESS_INTERVAL_SECONDS
                    ):
                        report()
                        last_emit = time.monotonic()

    # Throttling can skip the closing chunk, so the final update is always
    # sent; a progress bar must never stop short of the end.
    if written:
        report()
    if total is not None and existing + written < total:
        raise _IncompleteBody(
            f"{url} sent {existing + written} of {total} bytes before the "
            "connection ended"
        )
    return existing, written


def _verify(
    part: Path,
    dest: Path,
    expected_size: int | None,
    expected_sha256: str | None,
) -> None:
    """Prove the staged file is the model that was asked for, or raise.

    A size mismatch keeps the `.part`: the body was complete, so a later run
    with a corrected expectation can still verify it. A hash mismatch deletes
    it, because the bytes already on disk are the wrong bytes and resuming onto
    them cannot produce a good file.
    """
    actual_size = part.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise DownloadError(
            f"{dest.name} is {actual_size} bytes, expected {expected_size}"
        )
    if expected_sha256 is None:
        return
    actual = hash_file(part)
    if actual != expected_sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"{dest.name} has sha256 {actual}, expected {expected_sha256}"
        )


def download_file(
    url: str,
    dest: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    emit: Callable[[PipelineEvent], None] | None = None,
    retries: int = 3,
    chunk_size: int = 1024 * 1024,
    timeout: float = 30.0,
) -> DownloadResult:
    """Fetch `url` to `dest`, resuming a previous attempt and verifying the result.

    The `.part` file is kept across attempts and the rename to `dest` happens
    only once `expected_size` and `expected_sha256`, if given, both agree with
    what arrived. A `retries`-many set of recoverable failures — a dropped
    connection, a truncated body, a server answering 5xx — is retried with
    doubling backoff; a 404 or any other 4xx fails at once, because repeating
    it cannot help.

    Progress is reported as `device_progress` events on `task_id`
    `"model:<dest stem>"`, with `processed` and `total` in bytes and `fps` in
    bytes per second. `total` is `0` when neither `expected_size` nor the
    server's `Content-Length` says how big the file is, so a front-end must
    render that bar as indeterminate rather than divide by it. The first and
    the final event are always sent; the ones in between are throttled to
    `PROGRESS_INTERVAL_SECONDS`.
    """
    part = part_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Read once for the log line only; _transfer re-reads it per attempt
    # because a retry legitimately starts further along.
    staged = part.stat().st_size if part.exists() else 0
    size_text = f"{expected_size:,} bytes" if expected_size else "unknown size"
    message = f"Downloading {dest.name} ({size_text})"
    if staged:
        message = f"Resuming {dest.name} at {staged:,} bytes of {size_text}"
    _emit(
        emit,
        PipelineEvent(
            kind="stage",
            task_id=f"model:{dest.stem}",
            stage="download",
            message=message,
        ),
    )

    failure: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resumed_from, written = _transfer(
                url,
                part,
                f"model:{dest.stem}",
                emit,
                chunk_size,
                timeout,
                expected_size,
            )
        except (requests.RequestException, _IncompleteBody, _RetryableStatus) as exc:
            failure = exc
            if attempt == retries:
                break
            time.sleep(min(BACKOFF_BASE_SECONDS * 2**attempt, BACKOFF_MAX_SECONDS))
            continue
        _verify(part, dest, expected_size, expected_sha256)
        os.replace(part, dest)
        return DownloadResult(
            path=dest, bytes_written=written, resumed_from=resumed_from
        )

    raise DownloadError(f"{url} failed after {retries + 1} attempts: {failure}")
