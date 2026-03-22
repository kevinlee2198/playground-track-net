import torch
import torch.nn as nn

from models.backbone import UNetBackbone


class TrackNet(nn.Module):
    """TrackNet model wrapper.

    V2: backbone only (mdd=None, rstr=None).
    V5: backbone + MDD preprocessing + R-STR refinement head.
    """

    def __init__(
        self,
        backbone: UNetBackbone | None = None,
        mdd: nn.Module | None = None,
        rstr: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone if backbone is not None else UNetBackbone(in_channels=9, num_classes=3)
        self.mdd = mdd
        self.rstr = rstr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mdd is not None:
            x = self.mdd(x)
        out = self.backbone(x)
        if self.rstr is not None:
            out = self.rstr(out)
        return out
