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
    unchanged. With ``p`` 1.0 no random number is drawn for that decision;
    otherwise one number is drawn first, and a window it skips draws nothing
    more.

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


def mixup_batch(
    frames: torch.Tensor, heatmaps: torch.Tensor, alpha: float
) -> FrameHeatmapPair:
    """TrackNetV3's sample mixup of a training batch.

    For each of the B samples a weight ``lam`` is drawn from Beta(alpha,
    alpha) and replaced by ``max(lam, 1 - lam)``, so each sample keeps the
    larger share. One random permutation of the batch picks each sample's
    partner, and the result is ``x * lam + x[perm] * (1 - lam)`` for the
    frames and the heatmaps alike, with one ``lam`` per sample over its
    channels, height and width. A sample the permutation maps to itself, as
    in a batch of one, comes back unchanged.

    The random numbers come from torch's global generator, drawn on the CPU,
    so ``torch.manual_seed`` covers them whatever device the batch is on.
    These draws advance that generator, so a run with mixup on does not share
    the later sample order (from the second epoch) of the same-seed run with
    it off. Outputs have the inputs' shape, dtype and device.

    Raises:
        ValueError: If ``alpha`` is not above 0.
    """
    if not alpha > 0:
        raise ValueError(f"mixup alpha must be positive, got {alpha}")
    batch = frames.shape[0]
    lam = torch.distributions.Beta(alpha, alpha).sample((batch,))
    lam = torch.maximum(lam, 1 - lam)
    perm = torch.randperm(batch)

    def mix(x: torch.Tensor) -> torch.Tensor:
        weight = lam.to(x.device, x.dtype).view(-1, *[1] * (x.dim() - 1))
        # lerp(partner, x, lam) = x * lam + partner * (1 - lam), and is
        # exactly x where the partner is x.
        return torch.lerp(x[perm.to(x.device)], x, weight)

    return mix(frames), mix(heatmaps)


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
