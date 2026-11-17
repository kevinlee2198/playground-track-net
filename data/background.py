"""A clip's median background frame: the static scene with the moving ball removed."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def background_indices(num_frames: int, max_frames: int = 50) -> list[int]:
    """Indices of the frames a clip's background is computed from.

    All of them when the clip has at most ``max_frames`` frames; otherwise
    ``max_frames`` indices evenly spaced from the first frame to the last,
    increasing and without repeats.
    """
    if num_frames <= max_frames:
        return list(range(num_frames))
    # Spacing above 1 (num_frames > max_frames), so rounding never repeats one.
    return np.linspace(0, num_frames - 1, max_frames).round().astype(int).tolist()


def median_frame(frames: Iterable[np.ndarray]) -> np.ndarray:
    """The per-element median of equally shaped arrays, in their dtype and shape.

    For an integer dtype the median is rounded to the nearest integer (half to
    even) before the cast back.
    """
    stack = np.stack(list(frames))
    median = np.median(stack, axis=0)
    if np.issubdtype(stack.dtype, np.integer):
        median = np.round(median)
    return median.astype(stack.dtype)
