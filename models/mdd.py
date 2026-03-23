import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionDirectionDecoupling(nn.Module):
    """Motion Direction Decoupling (MDD) module for TrackNet V5.

    Sits before the U-Net backbone. Computes signed frame differences to
    preserve trajectory direction, then applies a learnable adaptive sigmoid
    attention to highlight motion regions.

    Input:  (B, 9, H, W) -- 3 concatenated RGB frames [I_{t-1}, I_t, I_{t+1}]
    Output: enriched  (B, 13, H, W) -- [I_{t-1}(3), A_prev(2), I_t(3), A_next(2), I_{t+1}(3)]
            attention (B, 4, H, W)  -- [A_prev(2), A_next(2)] for R-STR fusion

    Only 2 learnable parameters: alpha, beta (scalars, initialized to 0).

    Known limitations:
        - RGB reduction order: sigmoid is applied per-RGB-channel FIRST, then
          the 3 channels are averaged to produce 1 attention channel per polarity.
          The V5 paper is ambiguous on whether mean-then-sigmoid or
          sigmoid-then-mean is intended. We chose sigmoid-then-mean because it
          lets the adaptive threshold operate on individual color channels before
          collapsing, preserving per-channel sensitivity differences.
    """

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(()))
        self.beta = nn.Parameter(torch.zeros(()))

    def _adaptive_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable adaptive sigmoid: sigmoid(k * (|x| - m)).

        Args:
            x: Polarity tensor of shape (B, 3, H, W) -- one polarity from
               one frame transition (e.g., P_plus_prev with 3 RGB channels).

        Returns:
            Attention map of shape (B, 1, H, W) -- averaged across RGB.
        """
        eps = 1e-8
        k = 5.0 / (0.45 * torch.abs(torch.tanh(self.alpha)) + eps)
        m = 0.6 * torch.tanh(self.beta)
        # Apply sigmoid per-channel, then average across RGB (sigmoid-then-mean)
        att = torch.sigmoid(k * (torch.abs(x) - m))
        return att.mean(dim=1, keepdim=True)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Split 9-channel input into 3 RGB frames
        i_prev = x[:, 0:3]   # I_{t-1}: (B, 3, H, W)
        i_curr = x[:, 3:6]   # I_t:     (B, 3, H, W)
        i_next = x[:, 6:9]   # I_{t+1}: (B, 3, H, W)

        # Frame differences (signed)
        d_prev = i_curr - i_prev  # (B, 3, H, W)
        d_next = i_next - i_curr  # (B, 3, H, W)

        # Signed polarity decomposition
        p_plus_prev = F.relu(d_prev)    # brightening (arrival)
        p_minus_prev = F.relu(-d_prev)  # darkening (departure)
        p_plus_next = F.relu(d_next)
        p_minus_next = F.relu(-d_next)

        # Learnable attention: 2 channels per transition (plus + minus)
        a_prev_plus = self._adaptive_attention(p_plus_prev)    # (B, 1, H, W)
        a_prev_minus = self._adaptive_attention(p_minus_prev)  # (B, 1, H, W)
        a_next_plus = self._adaptive_attention(p_plus_next)    # (B, 1, H, W)
        a_next_minus = self._adaptive_attention(p_minus_next)  # (B, 1, H, W)

        a_prev = torch.cat([a_prev_plus, a_prev_minus], dim=1)  # (B, 2, H, W)
        a_next = torch.cat([a_next_plus, a_next_minus], dim=1)  # (B, 2, H, W)

        # Enriched output: interleave frames with attention maps
        enriched = torch.cat(
            [i_prev, a_prev, i_curr, a_next, i_next], dim=1
        )  # (B, 13, H, W)

        # Attention maps for downstream R-STR fusion
        attention = torch.cat([a_prev, a_next], dim=1)  # (B, 4, H, W)

        return enriched, attention
