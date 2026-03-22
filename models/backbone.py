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


class DownBlock(nn.Module):
    """2x ConvBlock + MaxPool2x2. Returns (pooled, skip)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        pooled = self.pool(x)
        return pooled, skip


class Bottleneck(nn.Module):
    """3x ConvBlock at the U-Net bottom."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.conv3 = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
