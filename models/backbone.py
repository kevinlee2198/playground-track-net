import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d(3x3, pad=1) -> GroupNorm(num_groups=8) -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity="relu")

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
        return self.pool(x), x


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
        return self.conv3(x)


class UpBlock(nn.Module):
    """Upsample2x + skip concat + 2x ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against spatial mismatch from odd input dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = x[:, :, : skip.shape[2], : skip.shape[3]]
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class UNetBackbone(nn.Module):
    """V2 U-Net encoder-decoder with skip connections and sigmoid output head."""

    def __init__(self, in_channels: int = 9, num_classes: int = 3) -> None:
        super().__init__()
        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.bottleneck = Bottleneck(256, 512)
        self.up1 = UpBlock(512 + 256, 256)
        self.up2 = UpBlock(256 + 128, 128)
        self.up3 = UpBlock(128 + 64, 64)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self._init_head()

    def _init_head(self) -> None:
        nn.init.kaiming_uniform_(self.head.weight, nonlinearity="relu")
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1, skip1 = self.down1(x)
        d2, skip2 = self.down2(d1)
        d3, skip3 = self.down3(d2)
        b = self.bottleneck(d3)
        u1 = self.up1(b, skip3)
        u2 = self.up2(u1, skip2)
        u3 = self.up3(u2, skip1)
        return self.sigmoid(self.head(u3))
