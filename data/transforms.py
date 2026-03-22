from __future__ import annotations

from collections.abc import Callable

import torch
import torchvision.transforms.v2 as T

FrameHeatmapPair = tuple[torch.Tensor, torch.Tensor]


class HorizontalFlip:
    """Randomly flip frames and heatmaps horizontally."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, frames: torch.Tensor, heatmaps: torch.Tensor
    ) -> FrameHeatmapPair:
        if torch.rand(1).item() < self.p:
            frames = frames.flip(-1)
            heatmaps = heatmaps.flip(-1)
        return frames, heatmaps


class FrameColorJitter:
    """Apply torchvision ColorJitter to each frame independently. Heatmaps unchanged."""

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
    ) -> None:
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
        )

    def __call__(
        self, frames: torch.Tensor, heatmaps: torch.Tensor
    ) -> FrameHeatmapPair:
        jittered = [
            self.jitter(frame).clamp(0.0, 1.0) for frame in frames.chunk(3, dim=0)
        ]
        return torch.cat(jittered, dim=0), heatmaps


class Mixup:
    """Mixup augmentation: blend two samples with a random lambda from Beta(alpha, alpha)."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def __call__(
        self,
        frames_a: torch.Tensor,
        heatmaps_a: torch.Tensor,
        frames_b: torch.Tensor,
        heatmaps_b: torch.Tensor,
    ) -> FrameHeatmapPair:
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        frames = lam * frames_a + (1 - lam) * frames_b
        heatmaps = lam * heatmaps_a + (1 - lam) * heatmaps_b
        return frames, heatmaps


class Compose:
    """Compose multiple (frames, heatmaps) transforms sequentially."""

    def __init__(self, transforms: list[Callable[..., FrameHeatmapPair]]) -> None:
        self.transforms = transforms

    def __call__(
        self, frames: torch.Tensor, heatmaps: torch.Tensor
    ) -> FrameHeatmapPair:
        for t in self.transforms:
            frames, heatmaps = t(frames, heatmaps)
        return frames, heatmaps
