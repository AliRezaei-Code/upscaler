"""A backend that upscales by nearest neighbour, for pipeline tests.

It lives in an importable module rather than inside a test file because
`core.worker.ChunkArgs` carries the backend *class* across a `spawn` boundary,
where a class is pickled by reference: a class defined inside a test function
cannot be found in the child process at all.

Two environment variables are test seams, read at construction:

* `UPSCALER_FAKE_BACKEND_LOG` — a directory where each worker appends a line
  naming its pid, the device index and the device's backend. In-process
  assertions cannot see a child's memory, so the record has to outlive it.
* `UPSCALER_FAKE_BACKEND_DELAY` — seconds to sleep per frame, so a stop test can
  interrupt a job at a known point instead of racing it.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from core.backends.base import Backend
from core.config import Device

#: What this backend pretends every model is.
SCALE = 2


def _record(device: Device) -> None:
    log_dir = os.environ.get("UPSCALER_FAKE_BACKEND_LOG")
    if not log_dir:
        return
    path = Path(log_dir) / f"device-{device.index}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"pid={os.getpid()} device={device.index} backend={device.backend}\n"
        )


class FakeBackend(Backend):
    """Nearest-neighbour upscaling by a fixed factor.

    Exact and fast, which is what the resume and completeness tests need: the
    output is a known function of the input, so a test can assert on pixels
    rather than on a tolerance.
    """

    name = "fake"

    def __init__(
        self, *, export_size: tuple[int, int] | None = None, tile_size: int = 0
    ) -> None:
        super().__init__()
        self.export_size = export_size
        self._configured_tile_size = tile_size
        self._delay = float(os.environ.get("UPSCALER_FAKE_BACKEND_DELAY", "0"))
        self.device: Device | None = None
        self.precision = "fp32"

    def load(self, model_path: Path, precision: str, device: Device) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        self.device = device
        self.scale = SCALE
        self.precision = precision
        _record(device)

    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        if self._delay:
            time.sleep(self._delay)
        return np.repeat(np.repeat(image_bgr, self.scale, axis=0), self.scale, axis=1)

    def close(self) -> None:
        self.device = None
