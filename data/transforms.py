from __future__ import annotations
import torch
import torchvision.transforms.v2 as T


class HorizontalFlip:
    """Randomly flip frames and heatmaps horizontally."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, frames: torch.Tensor, heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < self.p:
            frames = frames.flip(-1)
            heatmaps = heatmaps.flip(-1)
        return frames, heatmaps
