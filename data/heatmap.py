import torch


def source_to_model(coord: float, source_size: int, model_size: int) -> float:
    """Map a source-video pixel coordinate onto a frame resized to ``model_size``.

    Pixel index ``i`` covers ``[i, i + 1)`` with its centre at ``i + 0.5``, which
    is how ``cv2.resize`` maps pixels, so the result lands on the ball as the
    resized frame shows it. Apply per axis: widths for x, heights for y.
    """
    return (coord + 0.5) * (model_size / source_size) - 0.5


def model_to_source(coord: float, model_size: int, source_size: int) -> float:
    """The inverse of ``source_to_model``: a model-resolution coordinate in
    source-video pixels."""
    return (coord + 0.5) * (source_size / model_size) - 0.5


def generate_heatmap(
    x: float,
    y: float,
    visibility: int,
    height: int = 288,
    width: int = 512,
    radius: float = 2.5,
) -> torch.Tensor:
    """Generate a binary circular heatmap for a ball position.

    ``(x, y)`` is in heatmap pixels and may be fractional or lie outside the frame;
    the disc is clipped to the frame. Returns an all-zero heatmap when the ball is
    not visible (visibility == 0).
    """
    if visibility == 0:
        return torch.zeros(height, width, dtype=torch.float32)
    ys = torch.arange(height, dtype=torch.float32).unsqueeze(1)
    xs = torch.arange(width, dtype=torch.float32).unsqueeze(0)
    dist_sq = (xs - x) ** 2 + (ys - y) ** 2
    heatmap = (dist_sq <= radius**2).float()
    return heatmap
