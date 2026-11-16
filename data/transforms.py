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
    """Apply one torchvision ColorJitter draw to every RGB image in the window.

    ``frames`` holds any number of RGB images stacked on the channel axis: the
    three frames, and the background when the dataset appends one. Heatmaps
    are unchanged. The parameters are sampled once per window, so the jitter
    adds no frame-to-frame (or frame-to-background) colour change that the
    model could mistake for motion.

    A window is jittered with probability ``p`` and otherwise returned
    unchanged. With ``p`` 1.0 no random number is drawn for that decision.

    Raises:
        ValueError: If ``p`` is outside [0, 1].
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        p: float = 1.0,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"colour-jitter p must be in [0, 1], got {p}")
        self.p = p
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
        )

    def __call__(
        self, frames: torch.Tensor, heatmaps: torch.Tensor
    ) -> FrameHeatmapPair:
        if self.p < 1.0 and torch.rand(1).item() >= self.p:
            return frames, heatmaps
        # An (N, 3, H, W) batch: transforms.v2 samples parameters once per call.
        batch = frames.unflatten(0, (-1, 3))
        jittered = self.jitter(batch).clamp(0.0, 1.0)
        return jittered.flatten(0, 1), heatmaps


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
