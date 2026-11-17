from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.backbone import UNetBackbone
from models.tracknet import TrackNet


def build_model(config: dict | None = None) -> TrackNet:
    """The V2 ``TrackNet`` that ``config`` trains.

    Three RGB frames make 9 input channels; a true ``background_frame`` adds
    the clip's median background as a fourth image, 12 channels. A config
    without the key, or none at all, builds the 9-channel default, initialised
    exactly as ``TrackNet()`` is.
    """
    images = 4 if config and config.get("background_frame", False) else 3
    return TrackNet(UNetBackbone(in_channels=3 * images, num_classes=3))


def load_model(path: str | Path, device: torch.device) -> tuple[TrackNet, dict | None]:
    """The model saved at ``path``, built as it was trained, on ``device``.

    A trainer checkpoint (a dict holding ``"model_state_dict"``) builds the
    model from the config stored beside the weights; a bare state dict builds
    the default. Keys must match exactly.

    Returns:
        The model and the checkpoint's stored config, or None for a bare
        state dict.

    Raises:
        RuntimeError: If the weights do not fit the architecture built.
    """
    state_dict, config = _read_checkpoint(path, device)
    model = build_model(config)
    model.load_state_dict(state_dict)
    return model.to(device), config


def load_model_weights(
    model: nn.Module, path: str | Path, device: torch.device
) -> None:
    """Load weights into ``model`` from a trainer checkpoint or a bare state dict.

    A trainer checkpoint is a dict holding the weights under
    ``"model_state_dict"``; anything else is treated as a state dict. Keys must
    match exactly.

    Raises:
        RuntimeError: If the weights do not fit ``model``'s architecture.
    """
    state_dict, _config = _read_checkpoint(path, device)
    model.load_state_dict(state_dict)


def _read_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[dict, dict | None]:
    """The weights at ``path`` and the config stored beside them, or None for a
    bare state dict."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("config")
    return checkpoint, None
