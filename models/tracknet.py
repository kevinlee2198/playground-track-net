import torch
import torch.nn as nn

from models.backbone import UNetBackbone
from models.mdd import MotionDirectionDecoupling
from models.rstr import RSTRHead


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
        self.backbone = (
            backbone
            if backbone is not None
            else UNetBackbone(in_channels=9, num_classes=3)
        )
        self.mdd = mdd
        self.rstr = rstr

        if self.rstr is not None and getattr(self.backbone, "apply_sigmoid", True):
            raise ValueError(
                "R-STR requires raw logits from the backbone, but "
                "backbone.apply_sigmoid is True. Construct the backbone "
                "with apply_sigmoid=False when using R-STR."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = None
        if self.mdd is not None:
            mdd_out = self.mdd(x)
            if isinstance(mdd_out, tuple):
                x, attention = mdd_out
            else:
                x = mdd_out
        out = self.backbone(x)
        if self.rstr is not None:
            out = self.rstr(out, attention)
        return out


def tracknet_v5() -> TrackNet:
    """Create a complete TrackNet V5 model.

    V5 = MDD preprocessing + UNetBackbone(13ch, no sigmoid) + RSTRHead.

    Returns:
        TrackNet instance configured for V5 operation.
    """
    mdd = MotionDirectionDecoupling()
    backbone = UNetBackbone(in_channels=13, num_classes=3, apply_sigmoid=False)
    rstr = RSTRHead()
    return TrackNet(backbone=backbone, mdd=mdd, rstr=rstr)
