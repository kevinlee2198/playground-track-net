import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d(3x3, pad=1) -> GroupNorm(num_groups=8) -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity="relu")
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))
