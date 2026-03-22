import cv2
import numpy as np


def draw_ball_on_frame(
    frame: np.ndarray,
    x: float,
    y: float,
    confidence: float,
    radius: int = 10,
) -> np.ndarray:
    """Draw a circle at the detected ball position, color-coded by confidence.

    Args:
        frame: BGR image (H, W, 3) uint8.
        x: Ball x coordinate in frame resolution.
        y: Ball y coordinate in frame resolution.
        confidence: Detection confidence in [0, 1].
        radius: Circle radius in pixels.

    Returns:
        Copy of frame with circle drawn.
    """
    result = frame.copy()
    center = (int(round(x)), int(round(y)))

    # Color by confidence (BGR format)
    if confidence >= 0.8:
        color = (0, 255, 0)  # Green -- high confidence
    elif confidence >= 0.5:
        color = (0, 255, 255)  # Yellow -- medium confidence
    else:
        color = (0, 0, 255)  # Red -- low confidence

    cv2.circle(result, center, radius, color, thickness=-1)
    return result
