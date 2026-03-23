import torch
import torch.nn as nn


class MotionDirectionDecoupling(nn.Module):
    """Motion Direction Decoupling (MDD) module for TrackNet V5.

    Computes signed frame differences to preserve trajectory direction, then
    applies a learnable adaptive sigmoid attention to highlight motion regions.

    Input:  (B, 9, H, W) -- 3 concatenated RGB frames [I_{t-1}, I_t, I_{t+1}]
    Output: enriched  (B, 13, H, W) -- [I_{t-1}(3), A_prev(2), I_t(3), A_next(2), I_{t+1}(3)]
            attention (B, 4, H, W)  -- [A_prev(2), A_next(2)] for R-STR fusion

    Only 2 learnable scalar parameters: alpha (slope), beta (threshold).

    Note: sigmoid is applied per-RGB-channel first, then averaged to produce
    1 attention channel per polarity (sigmoid-then-mean). This preserves
    per-channel sensitivity before collapsing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(()))
        self.beta = nn.Parameter(torch.zeros(()))

    def _adaptive_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable adaptive sigmoid: sigmoid(k * (|x| - m)).

        Args:
            x: Polarity tensor (B, 3, H, W) for one polarity of one transition.

        Returns:
            Attention map (B, 1, H, W) averaged across RGB channels.
        """
        k = 5.0 / (0.45 * torch.tanh(self.alpha).abs() + 1e-8)
        m = 0.6 * torch.tanh(self.beta)
        att = torch.sigmoid(k * (x.abs() - m))
        return att.mean(dim=1, keepdim=True)

    def _polarity_attention(self, diff: torch.Tensor) -> torch.Tensor:
        """Compute 2-channel attention [plus, minus] from a frame difference.

        Args:
            diff: Signed frame difference (B, 3, H, W).

        Returns:
            Attention (B, 2, H, W) with plus (brightening) and minus (darkening).
        """
        a_plus = self._adaptive_attention(diff.clamp(min=0))
        a_minus = self._adaptive_attention((-diff).clamp(min=0))
        return torch.cat([a_plus, a_minus], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        i_prev, i_curr, i_next = x[:, 0:3], x[:, 3:6], x[:, 6:9]

        a_prev = self._polarity_attention(i_curr - i_prev)  # (B, 2, H, W)
        a_next = self._polarity_attention(i_next - i_curr)  # (B, 2, H, W)

        enriched = torch.cat([i_prev, a_prev, i_curr, a_next, i_next], dim=1)
        attention = torch.cat([a_prev, a_next], dim=1)

        return enriched, attention
