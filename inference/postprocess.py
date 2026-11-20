import math
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from scipy import ndimage

from data.heatmap import model_to_source


def heatmap_to_coordinates(
    heatmap: np.ndarray,
    orig_width: int,
    orig_height: int,
    threshold: float = 0.5,
) -> tuple[float, float, float] | None:
    """Convert a sigmoid heatmap to ball coordinates in original resolution.

    Args:
        heatmap: 2D array shape (H, W) with values in [0, 1].
        orig_width: Original video frame width.
        orig_height: Original video frame height.
        threshold: Detection threshold.

    Returns:
        (x, y, confidence) in original resolution (mapped with
        ``model_to_source``), or None if no ball detected.
    """
    h, w = heatmap.shape
    binary = (heatmap > threshold).astype(np.int32)

    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        return None

    # Find largest connected component
    component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    largest_label = np.argmax(component_sizes) + 1

    # Compute centroid of largest component
    cy, cx = ndimage.center_of_mass(binary, labeled, largest_label)

    # Confidence: mean heatmap value within the component
    component_mask = labeled == largest_label
    confidence = float(np.mean(heatmap[component_mask]))

    x_orig = model_to_source(cx, w, orig_width)
    y_orig = model_to_source(cy, h, orig_height)

    return (x_orig, y_orig, confidence)


def trajectory_rectification(
    detections: Sequence[tuple[float, float] | None],
    max_gap: int = 4,
    max_speed: float | None = None,
) -> list[tuple[float, float] | None]:
    """Fill short gaps between detections by linear interpolation.

    A run of missing frames is filled only when it lies between two detections,
    is at most ``max_gap`` frames long and, if ``max_speed`` is given, the
    distance between the two detections divided by the number of frames from
    one to the other is at most ``max_speed``. Each filled frame lies on the
    straight line between the two detections.
    Nothing is extrapolated: frames before the first detection, after the last,
    or in a gap that fails either limit stay None.

    Args:
        detections: (x, y) in source pixels, or None, for each frame.
        max_gap: The longest run of missing frames that is filled.
        max_speed: The fastest, in pixels per frame, the ball may have to travel
            between the two detections for their gap to be filled; None for no
            limit.

    Returns:
        A new list with the gaps filled. Detections are returned unchanged.
    """
    result = list(detections)
    known = [(i, d) for i, d in enumerate(detections) if d is not None]
    for (before, (x0, y0)), (after, (x1, y1)) in pairwise(known):
        frames = after - before
        if frames - 1 > max_gap:
            continue
        if max_speed is not None and math.hypot(x1 - x0, y1 - y0) > max_speed * frames:
            continue
        for i in range(before + 1, after):
            t = (i - before) / frames
            result[i] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    return result
