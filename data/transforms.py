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


class FrameColorJitter:
    """Apply torchvision ColorJitter to each frame independently. Heatmaps untouched."""
    def __init__(self, brightness: float = 0.3, contrast: float = 0.3, saturation: float = 0.3):
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)

    def __call__(self, frames: torch.Tensor, heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frame_list = frames.chunk(3, dim=0)
        jittered = []
        for frame in frame_list:
            frame = self.jitter(frame)
            frame = frame.clamp(0.0, 1.0)
            jittered.append(frame)
        frames = torch.cat(jittered, dim=0)
        return frames, heatmaps
