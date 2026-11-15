from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


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
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
