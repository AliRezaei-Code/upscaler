"""The backend contract and the tiling rule shared by both engines.

`Backend` is deliberately about **images**, not tensors: one worker process
serves both ONNX Runtime and PyTorch, and the tile fallback therefore lives
here once instead of once per engine.

`Backend.probe()` is not part of this ABC. `core.devices` is the single
enumerator; a second implementation would let the device list the UI shows and
the list `select_backend` receives drift apart.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np

from ..config import Device

# A frame larger than this is inferred tile by tile. 4000x4000 is roughly what
# a 16 GB accelerator can hold through a 4x model in float32; past it, the
# activation memory — not the weights — is what runs out.
TILE_THRESHOLD_PIXELS = 4000 * 4000 * 3
DEFAULT_TILE_SIZE = 512
MIN_TILE_SIZE = 64
# Tiles overlap by one eighth of their size on each axis and the overlap is
# feathered. The ported `_tile_process` in the original script had no overlap at
# all, which is why it showed hard seams.
TILE_OVERLAP_DIVISOR = 8


def resolve_tile_size(height: int, width: int, configured: int) -> int:
    """The tile edge to use for a `height` x `width` frame.

    `configured == 0` means "decide from the frame": `min(512, h, w)`.
    Otherwise the request is raised to at least `MIN_TILE_SIZE` and then
    clipped to the frame, so a hand-entered tile size can neither exceed the
    frame nor be degenerate — a 50-pixel-wide frame gets a 50-pixel tile, not a
    64-pixel one that every step would have to clip anyway.
    """
    if configured == 0:
        return max(1, min(DEFAULT_TILE_SIZE, height, width))
    return max(1, min(max(MIN_TILE_SIZE, configured), height, width))


def needs_tiling(height: int, width: int) -> bool:
    """Whether this frame is too large to infer in one piece."""
    return height * width * 3 > TILE_THRESHOLD_PIXELS


def tile_overlap(tile_size: int) -> int:
    """The overlap in pixels between neighbouring tiles."""
    return max(1, tile_size // TILE_OVERLAP_DIVISOR)


def _axis_window(
    length: int, overlap: int, feather_start: bool, feather_end: bool
) -> np.ndarray:
    """Feather weights along one axis of one tile.

    The two tiles that share an overlap must sum to 1 across it, or the seam
    shows. The earlier tile ramps up over the overlap and the later one ramps
    down, both over `length - 1` steps so the two profiles add to exactly 1 at
    every pixel; the outer edges of the frame keep full weight so the border is
    not darkened.
    """
    weights = np.ones(length, dtype=np.float32)
    span = min(overlap, length)
    if feather_end:
        weights[length - span :] = np.linspace(
            0.0, 1.0, span, endpoint=False, dtype=np.float32
        )
    if feather_start:
        weights[:span] = np.linspace(
            1.0, 1.0 / span, span, endpoint=False, dtype=np.float32
        )
    return weights


def _window(
    rows: int,
    cols: int,
    overlap: int,
    feather_top: bool,
    feather_bottom: bool,
    feather_left: bool,
    feather_right: bool,
) -> np.ndarray:
    """A 2-D feather weight for one tile, as the outer product of its axes."""
    vertical = _axis_window(rows, overlap, feather_top, feather_bottom)
    horizontal = _axis_window(cols, overlap, feather_left, feather_right)
    return np.outer(vertical, horizontal).astype(np.float32)


def tiled_infer(
    full_infer: Callable[[np.ndarray], np.ndarray],
    image_bgr: np.ndarray,
    tile_size: int,
    scale: int,
) -> np.ndarray:
    """Infer a large frame tile by tile, feathering the overlaps.

    `full_infer` must be the backend's own single-shot path: this function is
    what `Backend.infer` falls back to, so it cannot call `infer` again.
    """
    height, width = image_bgr.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("cannot tile an empty frame")
    if scale < 1:
        raise ValueError(f"scale must be at least 1, got {scale}")
    overlap = tile_overlap(tile_size)
    step = max(1, tile_size - overlap)
    rows = list(range(0, max(1, height), step))
    cols = list(range(0, max(1, width), step))
    if height > tile_size:
        rows[-1] = height - tile_size
    if width > tile_size:
        cols[-1] = width - tile_size

    output: np.ndarray | None = None
    weights: np.ndarray | None = None
    for row_index, y in enumerate(rows):
        y_end = min(y + tile_size, height)
        for col_index, x in enumerate(cols):
            x_end = min(x + tile_size, width)
            patch = full_infer(image_bgr[y:y_end, x:x_end])
            if output is None:
                channels = patch.shape[2]
                output = np.zeros(
                    (height * scale, width * scale, channels), dtype=np.float32
                )
                weights = np.zeros((height * scale, width * scale), dtype=np.float32)
            window = _window(
                y_end - y,
                x_end - x,
                overlap,
                feather_top=row_index > 0,
                feather_bottom=row_index < len(rows) - 1,
                feather_left=col_index > 0,
                feather_right=col_index < len(cols) - 1,
            )
            # The weight multiplies the *upscaled* patch, so it is expanded by
            # the scale before use.
            window = np.repeat(np.repeat(window, scale, axis=0), scale, axis=1)
            top = y * scale
            left = x * scale
            out_h = (y_end - y) * scale
            out_w = (x_end - x) * scale
            assert output is not None and weights is not None
            output[top : top + out_h, left : left + out_w] += (
                patch.astype(np.float32) * window[:, :, None]
            )
            weights[top : top + out_h, left : left + out_w] += window
    if output is None or weights is None:  # pragma: no cover - guarded above
        raise ValueError("cannot tile an empty frame")
    safe = np.maximum(weights, 1e-3)[:, :, None]
    return cast("np.ndarray", np.clip(output / safe, 0, 255).astype(np.uint8))


class Backend(ABC):
    """One loaded model, ready to turn frames into bigger frames.

    Attributes:
        name: Human-readable backend name, shown in the log.
        scale: The model's output multiplier, learned in `load()`. It is an
            attribute rather than a parameter because `load()` is the only
            place that can know it, and four separate consumers — preflight,
            the tiling threshold, the completeness check and the encode's
            output dimensions — need it afterwards.
    """

    name: str = "backend"
    scale: int = 0

    def __init__(self) -> None:
        # DirectML permits only one concurrent Run per session object, so every
        # call is serialised; on the other backends the lock is uncontended.
        self._infer_lock = threading.Lock()

    @abstractmethod
    def load(self, model_path: Path, precision: str, device: Device) -> None:
        """Load `model_path` onto `device` and set `self.scale`."""

    @abstractmethod
    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        """Upscale one BGR uint8 frame, honouring the tiling rule."""

    def close(self) -> None:  # noqa: B027 - concrete on purpose
        """Release the session, the model and the device memory.

        Concrete and empty: a backend with no native resource to free has
        nothing to do, and the pipeline calls this unconditionally rather than
        asking each backend whether it has a teardown at all.
        """

    def infer_guarded(self, image_bgr: np.ndarray) -> np.ndarray:
        """`infer`, serialised.

        Two threads sharing one DirectML session raise rather than queue, and
        the worker already has a reader and a writer thread of its own.
        """
        with self._infer_lock:
            return self.infer(image_bgr)
