import torch
import torch.nn as nn


class WBCEFocalLoss(nn.Module):
    """Weighted Binary Cross-Entropy with focal-style dynamic weights.

    L = -1/N * sum[(1-p)^2 * y * log(p) + p^2 * (1-y) * log(1-p)]

    Expects predictions after sigmoid (probabilities in [0, 1]).
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.clamp(self.eps, 1.0 - self.eps)
        pos_weight = (1.0 - p) ** 2
        neg_weight = p**2
        loss = -(
            pos_weight * target * torch.log(p)
            + neg_weight * (1.0 - target) * torch.log(1.0 - p)
        )
        return loss.mean()
