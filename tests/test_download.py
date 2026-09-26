"""The downloader, against a real socket.

A fake `requests` proves nothing about resuming: whether the retry asks for the
right offset is a fact about the bytes on the wire, so every test here runs an
in-process `ThreadingHTTPServer` on 127.0.0.1 and records the `Range` header
each request carried. The connection is broken for real — the handler writes
part of the body, promises the full length and then drops the socket — so the
client genuinely sees a short read rather than a short file.
"""

from __future__ import annotations

import hashlib
import socket
import struct
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.download import download_file, part_path
from core.errors import DownloadError
from core.events import PipelineEvent

BODY = bytes(range(256)) * 6000
BODY_SIZE = len(BODY)
BODY_SHA256 = hashlib.sha256(BODY).hexdigest()
DROP_AFTER = 512 * 1024
# Small enough that a broken connection is repaired within the test's patience.
CHUNK = 64 * 1024
TIMEOUT = 10.0


@dataclass
class _Host:
    """What the fake origin should do, and what it was asked for."""

    body: bytes
    #: Break the first transfer after this many bytes.
    drop_first_after: int | None = None
    #: Break it with a TCP reset rather than a clean close.
    drop_with_reset: bool = False
    #: Answer `200` to a `Range` request, as a server ignoring ranges does.
    ignore_range: bool = False
    #: Answer every request with this status.
    always_status: int | None = None
    seen: list[str | None] = field(default_factory=list)


@dataclass(frozen=True)
class _Served:
    url: str
    host: _Host


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        """Stay quiet: the transfers under test are not meant to be read."""

    @property
    def _script(self) -> _Host:
        server = self.server
        assert isinstance(server, _Server)
        return server.script

    def _respond(self, body: bytes, start: int, total: int, drop: bool) -> None:
        """Send `body` as part of a resource of `total` bytes, then maybe drop."""
        if start:
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{total - 1}/{total}")
            self.send_header("Content-Length", str(len(body)))
        else:
            self.send_response(200)
            # The full length is promised even when the body is cut short:
            # that mismatch is exactly what makes a truncated transfer visible.
            self.send_header("Content-Length", str(total))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        if drop:
            self.close_connection = True
            if self._script.drop_with_reset:
                # SO_LINGER with a zero timeout makes close() send RST, which
                # is how a proxy or a mobile network really cuts a transfer.
                self.connection.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                )
            self.connection.close()

    def do_GET(self) -> None:
        script = self._script
        script.seen.append(self.headers.get("Range"))
        total = len(script.body)

        if script.always_status is not None:
            self.send_response(script.always_status)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        start = 0
        range_header = script.seen[-1]
        if range_header and not script.ignore_range:
            start = int(range_header.removeprefix("bytes=").removesuffix("-"))
            if start >= total:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{total}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

        first = len(script.seen) == 1
        if first and script.drop_first_after is not None:
            self._respond(script.body[: script.drop_first_after], start, total, True)
            return
        self._respond(script.body[start:], start, total, False)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    script: _Host

    def __init__(self, script: _Host) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.script = script


@pytest.fixture
def serve() -> Iterator[Callable[[_Host], _Served]]:
    """Start a throwaway origin on 127.0.0.1 and shut it down afterwards."""
    servers: list[_Server] = []

    def start(script: _Host) -> _Served:
        server = _Server(script)
        # The default 0.5 s poll interval would otherwise dominate a suite
        # that starts nine of these.
        threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()
        servers.append(server)
        return _Served(
            url=f"http://127.0.0.1:{server.server_port}/model.pth", host=script
        )

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def test_resume_asks_for_the_bytes_already_on_disk(
    serve: Callable[[_Host], _Served], tmp_path: Path
) -> None:
    """A dropped connection costs a range request, not the whole file."""
    served = serve(_Host(body=BODY, drop_first_after=DROP_AFTER))
    dest = tmp_path / "RealESRGAN_x4plus.pth"

    result = download_file(
        served.url,
        dest,
        expected_size=BODY_SIZE,
        expected_sha256=BODY_SHA256,
        chunk_size=CHUNK,
        timeout=TIMEOUT,
    )

    assert served.host.seen == [None, f"bytes={DROP_AFTER}-"]
    assert dest.read_bytes() == BODY
    assert result.resumed_from + result.bytes_written == BODY_SIZE
    assert not part_path(dest).exists()


def test_resume_survives_a_connection_reset(
    serve: Callable[[_Host], _Served], tmp_path: Path
) -> None:
    """An RST is repaired the same way, and still produces the exact file."""
    served = serve(_Host(body=BODY, drop_first_after=DROP_AFTER, drop_with_reset=True))
    dest = tmp_path / "model.pth"

    download_file(
        served.url,
        dest,
        expected_size=BODY_SIZE,
        expected_sha256=BODY_SHA256,
        chunk_size=CHUNK,
        timeout=TIMEOUT,
    )

    assert served.host.seen[0] is None
    for header in served.host.seen[1:]:
        assert header is not None and header.startswith("bytes=")
        assert int(header.removeprefix("bytes=").removesuffix("-")) <= BODY_SIZE
    assert dest.read_bytes() == BODY


def test_range_ignored_by_the_server_restarts_from_zero(
    serve: Callable[[_Host], _Served], tmp_path: Path
) -> None:
    """A `200` to a `Range` must not be appended to the staged bytes."""
    served = serve(_Host(body=BODY, ignore_range=True))
    dest = tmp_path / "model.pth"
    part_path(dest).write_bytes(b"\x00" * 1000)

    result = download_file(served.url, dest, chunk_size=CHUNK, timeout=TIMEOUT)

    assert served.host.seen == ["bytes=1000-"]
    assert dest.read_bytes() == BODY
    assert (result.resumed_from, result.bytes_written) == (0, BODY_SIZE)


def test_a_complete_part_file_is_renamed_not_refetched(
    serve: Callable[[_Host], _Served], tmp_path: Path
) -> None:
    """A `416` means the staged bytes already cover the whole resource."""
    served = serve(_Host(body=BODY))
    dest = tmp_path / "model.pth"
    part_path(dest).write_bytes(BODY)

    result = download_file(
        served.url,
        dest,
        expected_size=BODY_SIZE,
        expected_sha256=BODY_SHA256,
        timeout=TIMEOUT,
    )

    assert served.host.seen == [f"bytes={BODY_SIZE}-"]
    assert (result.resumed_from, result.bytes_written) == (BODY_SIZE, 0)
    assert dest.read_bytes() == BODY
    assert not part_path(dest).exists()


def test_size_mismatch_names_both_numbers(
    serve: Callable[[_Host], _Served], tmp_path: Path
) -> None:
    """A truncated or padded model is refused before it reaches the app."""
    served = serve(_Host(body=BODY))
    dest = tmp_path / "model.pth"

    with pytest.raises(DownloadError) as failure:
        download_file(
            served.url,
            dest,
            expected_size=BODY_SIZE + 7,
            chunk_size=CHUNK,
            timeout=TIMEOUT,
        )

    assert str(BODY_SIZE) in str(failure.value)
    assert str(BODY_SIZE + 7) in str(failure.value)
    assert not dest.exists()


def test_hash_mismatch_deletes_the_staged_file(
    serve: Callable[[_Host], _Served], tmp_path: Path
) -> None:
    """Wrong bytes cannot be repaired by resuming, so they are thrown away."""
    served = serve(_Host(body=BODY))
    dest = tmp_path / "model.pth"

    with pytest.raises(DownloadError) as failure:
        download_file(
            served.url,
            dest,
            expected_sha256="0" * 64,
            chunk_size=CHUNK,
            timeout=TIMEOUT,
        )

    assert "0" * 64 in str(failure.value)
    assert not part_path(dest).exists()
    assert not dest.exists()


def test_progress_is_reported_in_bytes(
    serve: Callable[[_Host], _Served], tmp_path: Path
) -> None:
    """The Models tab needs a bar it can move, in bytes, under a model task."""
    served = serve(_Host(body=BODY))
    dest = tmp_path / "RealESRGAN_x4plus.pth"
    events: list[PipelineEvent] = []

    download_file(
        served.url,
        dest,
        expected_size=BODY_SIZE,
        emit=events.append,
        chunk_size=CHUNK,
        timeout=TIMEOUT,
    )

    progress = [event for event in events if event.kind == "device_progress"]
    assert progress
    assert progress[0].processed <= BODY_SIZE
    assert progress[-1].processed == BODY_SIZE
    assert progress[-1].total == BODY_SIZE
    assert progress[-1].fps > 0
    assert {event.task_id for event in progress} == {"model:RealESRGAN_x4plus"}
    stages = [event for event in events if event.kind == "stage"]
    assert [event.stage for event in stages] == ["download"]
    assert "RealESRGAN_x4plus.pth" in stages[0].message
    assert f"{BODY_SIZE:,}" in stages[0].message


def test_a_recoverable_failure_is_retried_a_bounded_number_of_times(
    serve: Callable[[_Host], _Served], tmp_path: Path
) -> None:
    """A server that is down fails the download instead of hanging on it."""
    served = serve(_Host(body=BODY, always_status=503))
    dest = tmp_path / "model.pth"

    with pytest.raises(DownloadError) as failure:
        download_file(served.url, dest, retries=1, timeout=TIMEOUT)

    assert len(served.host.seen) == 2
    assert "2 attempts" in str(failure.value)
    assert not dest.exists()


def test_a_missing_model_is_not_retried(
    serve: Callable[[_Host], _Served], tmp_path: Path
) -> None:
    """A 404 will still be a 404 in half a second, so it is asked once."""
    served = serve(_Host(body=BODY, always_status=404))
    dest = tmp_path / "model.pth"

    with pytest.raises(DownloadError) as failure:
        download_file(served.url, dest, retries=3, timeout=TIMEOUT)

    assert len(served.host.seen) == 1
    assert "404" in str(failure.value)
