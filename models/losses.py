import torch
import torch.nn as nn
import torch.nn.functional as F


class WBCEFocalLoss(nn.Module):
    """Weighted Binary Cross-Entropy with focal-style dynamic weights.

    L = -1/N * sum[(1-p)^2 * y * log(p) + p^2 * (1-y) * log(1-p)],  p = sigmoid(z)

    Expects pre-sigmoid logits z of any float dtype. The loss is computed in
    float32 via logsigmoid, so confident predictions stay finite even when the
    logits come from a bfloat16 autocast region.
    """

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        z = logits.float()
        y = target.float()
        p = torch.sigmoid(z)
        one_minus_p = torch.sigmoid(-z)
        loss = -(
            one_minus_p**2 * y * F.logsigmoid(z) + p**2 * (1.0 - y) * F.logsigmoid(-z)
        )
        return loss.mean()
