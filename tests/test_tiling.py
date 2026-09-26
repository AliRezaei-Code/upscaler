"""The tiling rule: thresholds, clamping, and seam-free stitching.

The ported `_tile_process` in the original script tiled with **no overlap at
all**, which is why a tiled run showed hard seams. These tests assert the
overlap is real and the stitch is invisible: a constant-valued frame must come
back constant everywhere, which any broken weight normalisation fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.backends.base import (
    DEFAULT_TILE_SIZE,
    MIN_TILE_SIZE,
    TILE_OVERLAP_DIVISOR,
    TILE_THRESHOLD_PIXELS,
    needs_tiling,
    resolve_tile_size,
    tile_overlap,
    tiled_infer,
)


def test_default_tile_size_is_512() -> None:
    assert DEFAULT_TILE_SIZE == 512
    assert MIN_TILE_SIZE == 64
    assert TILE_OVERLAP_DIVISOR == 8
    assert TILE_THRESHOLD_PIXELS == 4000 * 4000 * 3


@pytest.mark.parametrize(
    ("height", "width", "configured", "expected"),
    [
        (1080, 1920, 0, 512),
        (100, 100, 0, 100),
        (8, 8, 0, 8),
        (1, 1, 0, 1),
        (1080, 1920, 256, 256),
        (1080, 1920, 4096, 1080),
        (100, 50, 512, 50),
        (100, 50, 8, 50),
        (2000, 2000, 8, MIN_TILE_SIZE),
    ],
)
def test_resolve_tile_size(
    height: int, width: int, configured: int, expected: int
) -> None:
    assert resolve_tile_size(height, width, configured) == expected


def test_resolve_tile_size_never_exceeds_the_frame() -> None:
    for height, width in ((10, 4000), (4000, 10), (37, 37)):
        assert resolve_tile_size(height, width, 0) <= min(height, width)


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (480, 854, False),
        (2160, 3840, False),  # 4K is 24.9M samples, under the 48M threshold
        (4000, 4000, False),  # exactly at the threshold, so not tiled
        (4001, 4000, True),
        (6000, 8000, True),
    ],
)
def test_needs_tiling(height: int, width: int, expected: bool) -> None:
    assert needs_tiling(height, width) is expected


def test_overlap_is_an_eighth_of_the_tile() -> None:
    assert tile_overlap(512) == 64
    assert tile_overlap(64) == 8
    assert tile_overlap(1) == 1


def _nearest_scale(image: np.ndarray, scale: int) -> np.ndarray:
    """A stand-in engine: nearest-neighbour upscaling by `scale`."""
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def test_tiled_infer_returns_the_full_scaled_frame() -> None:
    frame = np.full((200, 300, 3), 128, dtype=np.uint8)
    out = tiled_infer(
        lambda patch: _nearest_scale(patch, 2), frame, tile_size=64, scale=2
    )
    assert out.shape == (400, 600, 3)


def test_tiled_infer_leaves_no_seam_on_a_constant_frame() -> None:
    frame = np.full((200, 300, 3), 200, dtype=np.uint8)
    out = tiled_infer(
        lambda patch: _nearest_scale(patch, 2), frame, tile_size=64, scale=2
    )
    assert out.shape == (400, 600, 3)
    # Every pixel, including the overlap seams, must be the original value.
    assert out.min() == 200
    assert out.max() == 200


def test_tiled_infer_reproduces_the_whole_frame_exactly_on_a_gradient() -> None:
    gradient = np.tile(
        np.linspace(0, 255, 300, dtype=np.float32)[None, :, None], (200, 1, 3)
    ).astype(np.uint8)
    out = tiled_infer(
        lambda patch: _nearest_scale(patch, 2), gradient, tile_size=64, scale=2
    )
    # The stand-in engine is exact, so a correct stitch must reproduce the
    # untiled result pixel for pixel. A seam, a double-counted overlap or a
    # wrong weight normalisation all break this equality.
    assert np.array_equal(out, _nearest_scale(gradient, 2))


def test_tiled_infer_covers_every_pixel_of_a_frame_that_is_not_a_multiple() -> None:
    frame = np.arange(137 * 251 * 3, dtype=np.uint8).reshape(137, 251, 3)
    out = tiled_infer(
        lambda patch: _nearest_scale(patch, 2), frame, tile_size=64, scale=2
    )
    assert out.shape == (274, 502, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out, _nearest_scale(frame, 2))


def test_tiled_infer_handles_a_frame_smaller_than_one_tile() -> None:
    frame = np.full((10, 10, 3), 77, dtype=np.uint8)
    out = tiled_infer(
        lambda patch: _nearest_scale(patch, 4), frame, tile_size=512, scale=4
    )
    assert out.shape == (40, 40, 3)
    assert out.min() == 77 == out.max()


def test_tiled_infer_actually_tiles_rather_than_repeating_one_call() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    sizes: list[tuple[int, int]] = []

    def recording(patch: np.ndarray) -> np.ndarray:
        sizes.append(patch.shape[:2])
        return _nearest_scale(patch, 2)

    tiled_infer(recording, frame, tile_size=64, scale=2)
    assert len(sizes) > 1
    assert all(height <= 64 and width <= 64 for height, width in sizes)


def test_tiled_infer_clamps_output_to_uint8() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    def overdriving(patch: np.ndarray) -> np.ndarray:
        return np.full(
            (patch.shape[0] * 2, patch.shape[1] * 2, 3), 4000, dtype=np.float32
        )

    out = tiled_infer(overdriving, frame, tile_size=64, scale=2)
    assert out.max() == 255


def test_tiled_infer_refuses_an_empty_frame() -> None:
    with pytest.raises(ValueError, match="empty frame"):
        tiled_infer(lambda patch: patch, np.zeros((0, 0, 3), dtype=np.uint8), 64, 2)
