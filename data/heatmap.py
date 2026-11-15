import torch


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
