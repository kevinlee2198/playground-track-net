import numpy as np
from scipy import ndimage
from scipy.interpolate import UnivariateSpline


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
        (x, y, confidence) in original resolution, or None if no ball detected.
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

    # Scale to original resolution
    x_orig = cx * (orig_width / w)
    y_orig = cy * (orig_height / h)

    return (x_orig, y_orig, confidence)


def trajectory_rectification(
    detections: list[tuple[float, float] | None],
    window: int = 8,
) -> list[tuple[float, float] | None]:
    """Interpolate missing ball positions using spline fitting on surrounding detections.

    Args:
        detections: List of (x, y) tuples or None for each frame.
        window: Number of surrounding frames to consider for fitting.

    Returns:
        New list with gaps filled where possible. Existing detections are preserved.
    """
    n = len(detections)
    result = list(detections)

    for i in range(n):
        if result[i] is not None:
            continue

        # Gather known positions within the window
        half_w = window // 2
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)

        known_indices = []
        known_x = []
        known_y = []
        for j in range(start, end):
            if detections[j] is not None:
                known_indices.append(j)
                known_x.append(detections[j][0])
                known_y.append(detections[j][1])

        # Need at least 3 known positions for spline fitting
        if len(known_indices) < 3:
            continue

        k = min(3, len(known_indices) - 1)  # spline degree
        try:
            spline_x = UnivariateSpline(known_indices, known_x, k=k, s=0)
            spline_y = UnivariateSpline(known_indices, known_y, k=k, s=0)
            interp_x = float(spline_x(i))
            interp_y = float(spline_y(i))
            result[i] = (interp_x, interp_y)
        except Exception:
            # Spline fitting can fail with degenerate inputs -- skip
            continue

    return result
