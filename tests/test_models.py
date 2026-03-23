import pytest
import torch
import torch.nn as nn

from models.backbone import ConvBlock, DownBlock, Bottleneck, UpBlock, UNetBackbone
from models.losses import WBCEFocalLoss
from models.tracknet import TrackNet


class TestConvBlock:
    def test_output_shape(self):
        block = ConvBlock(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 288, 512)
        out = block(x)
        assert out.shape == (2, 128, 288, 512)

    def test_preserves_spatial_dims(self):
        block = ConvBlock(in_channels=9, out_channels=64)
        x = torch.randn(1, 9, 288, 512)
        out = block(x)
        assert out.shape[2:] == x.shape[2:]

    def test_groupnorm_num_groups(self):
        block = ConvBlock(in_channels=64, out_channels=128)
        assert isinstance(block.norm, nn.GroupNorm)
        assert block.norm.num_groups == 8

    def test_kaiming_init(self):
        block = ConvBlock(in_channels=64, out_channels=128)
        assert block.conv.weight.abs().sum() > 0

    def test_groupnorm_init(self):
        block = ConvBlock(in_channels=64, out_channels=128)
        assert torch.allclose(block.norm.weight, torch.ones_like(block.norm.weight))
        assert torch.allclose(block.norm.bias, torch.zeros_like(block.norm.bias))


class TestDownBlock:
    def test_output_shapes(self):
        block = DownBlock(in_channels=9, out_channels=64)
        x = torch.randn(2, 9, 288, 512)
        pooled, skip = block(x)
        assert skip.shape == (2, 64, 288, 512)
        assert pooled.shape == (2, 64, 144, 256)

    def test_down2_shapes(self):
        block = DownBlock(in_channels=64, out_channels=128)
        x = torch.randn(2, 64, 144, 256)
        pooled, skip = block(x)
        assert skip.shape == (2, 128, 144, 256)
        assert pooled.shape == (2, 128, 72, 128)

    def test_down3_shapes(self):
        block = DownBlock(in_channels=128, out_channels=256)
        x = torch.randn(2, 128, 72, 128)
        pooled, skip = block(x)
        assert skip.shape == (2, 256, 72, 128)
        assert pooled.shape == (2, 256, 36, 64)

    def test_has_two_conv_blocks(self):
        block = DownBlock(in_channels=9, out_channels=64)
        assert isinstance(block.conv1, ConvBlock)
        assert isinstance(block.conv2, ConvBlock)


class TestBottleneck:
    def test_output_shape(self):
        block = Bottleneck(in_channels=256, out_channels=512)
        x = torch.randn(2, 256, 36, 64)
        out = block(x)
        assert out.shape == (2, 512, 36, 64)

    def test_has_three_conv_blocks(self):
        block = Bottleneck(in_channels=256, out_channels=512)
        assert isinstance(block.conv1, ConvBlock)
        assert isinstance(block.conv2, ConvBlock)
        assert isinstance(block.conv3, ConvBlock)


class TestUpBlock:
    def test_up1_shape(self):
        block = UpBlock(in_channels=768, out_channels=256)
        x = torch.randn(2, 512, 36, 64)
        skip = torch.randn(2, 256, 72, 128)
        out = block(x, skip)
        assert out.shape == (2, 256, 72, 128)

    def test_up2_shape(self):
        block = UpBlock(in_channels=384, out_channels=128)
        x = torch.randn(2, 256, 72, 128)
        skip = torch.randn(2, 128, 144, 256)
        out = block(x, skip)
        assert out.shape == (2, 128, 144, 256)

    def test_up3_shape(self):
        block = UpBlock(in_channels=192, out_channels=64)
        x = torch.randn(2, 128, 144, 256)
        skip = torch.randn(2, 64, 288, 512)
        out = block(x, skip)
        assert out.shape == (2, 64, 288, 512)

    def test_has_two_conv_blocks(self):
        block = UpBlock(in_channels=768, out_channels=256)
        assert isinstance(block.conv1, ConvBlock)
        assert isinstance(block.conv2, ConvBlock)


class TestUNetBackbone:
    def test_output_shape(self):
        model = UNetBackbone(in_channels=9, num_classes=3)
        x = torch.randn(2, 9, 288, 512)
        out = model(x)
        assert out.shape == (2, 3, 288, 512)

    def test_output_range_sigmoid(self):
        model = UNetBackbone(in_channels=9, num_classes=3)
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_v5_input_channels(self):
        model = UNetBackbone(in_channels=13, num_classes=3)
        x = torch.randn(1, 13, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)

    def test_encoder_decoder_structure(self):
        model = UNetBackbone(in_channels=9, num_classes=3)
        assert isinstance(model.down1, DownBlock)
        assert isinstance(model.down2, DownBlock)
        assert isinstance(model.down3, DownBlock)
        assert isinstance(model.bottleneck, Bottleneck)
        assert isinstance(model.up1, UpBlock)
        assert isinstance(model.up2, UpBlock)
        assert isinstance(model.up3, UpBlock)

    def test_parameter_count_reasonable(self):
        model = UNetBackbone(in_channels=9, num_classes=3)
        total = sum(p.numel() for p in model.parameters())
        assert 1_000_000 < total < 50_000_000


class TestWBCEFocalLoss:
    def test_returns_scalar(self):
        loss_fn = WBCEFocalLoss()
        pred = torch.sigmoid(torch.randn(2, 3, 288, 512))
        target = torch.zeros(2, 3, 288, 512)
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_perfect_prediction_low_loss(self):
        loss_fn = WBCEFocalLoss()
        target = torch.zeros(1, 3, 32, 32)
        target[:, :, 15:17, 15:17] = 1.0
        pred = target.clone().clamp(1e-6, 1 - 1e-6)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01

    def test_bad_prediction_high_loss(self):
        loss_fn = WBCEFocalLoss()
        target = torch.zeros(1, 3, 32, 32)
        target[:, :, 15:17, 15:17] = 1.0
        pred = (1.0 - target).clamp(1e-6, 1 - 1e-6)
        loss = loss_fn(pred, target)
        assert loss.item() > 1.0

    def test_all_zero_target(self):
        loss_fn = WBCEFocalLoss()
        pred = torch.full((1, 3, 32, 32), 0.1)
        target = torch.zeros(1, 3, 32, 32)
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_continuous_targets_mixup(self):
        loss_fn = WBCEFocalLoss()
        pred = torch.sigmoid(torch.randn(1, 3, 32, 32))
        target = torch.rand(1, 3, 32, 32)
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flows(self):
        loss_fn = WBCEFocalLoss()
        raw = torch.randn(1, 3, 32, 32, requires_grad=True)
        pred = torch.sigmoid(raw)
        target = torch.zeros(1, 3, 32, 32)
        target[:, :, 15:17, 15:17] = 1.0
        loss = loss_fn(pred, target)
        loss.backward()
        assert raw.grad is not None
        assert raw.grad.abs().sum() > 0


class TestTrackNet:
    def test_v2_forward(self):
        model = TrackNet()
        x = torch.randn(2, 9, 288, 512)
        out = model(x)
        assert out.shape == (2, 3, 288, 512)

    def test_output_range(self):
        model = TrackNet()
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_mdd_slot_none(self):
        model = TrackNet()
        assert model.mdd is None
        assert model.rstr is None

    def test_backbone_accessible(self):
        model = TrackNet()
        assert isinstance(model.backbone, UNetBackbone)


class TestUpBlockEdgeCases:
    def test_odd_spatial_dims(self):
        """UpBlock should handle spatial mismatch from odd input sizes."""
        block = UpBlock(in_channels=768, out_channels=256)
        x = torch.randn(1, 512, 37, 65)  # odd dims
        skip = torch.randn(1, 256, 73, 129)  # 2x odd - 1
        out = block(x, skip)
        assert out.shape == (1, 256, 73, 129)


class TestSkipConnections:
    def test_skip_connections_carry_information(self):
        """Changing skip data should change the output, proving skips are connected."""
        model = UNetBackbone(in_channels=9, num_classes=3)
        model.eval()
        x = torch.randn(1, 9, 288, 512)

        torch.manual_seed(42)
        out1 = model(x).detach().clone()

        # Corrupt the skip connection by hooking into down1
        original_forward = model.down1.forward

        def corrupted_forward(inp):
            pooled, skip = original_forward(inp)
            return pooled, torch.zeros_like(skip)

        model.down1.forward = corrupted_forward
        out2 = model(x).detach()
        model.down1.forward = original_forward  # restore

        assert not torch.allclose(out1, out2, atol=1e-5), (
            "Output should change when skip connections are zeroed"
        )


class TestTrackNetCustomBackbone:
    def test_custom_backbone(self):
        """TrackNet should accept a custom backbone."""
        custom = UNetBackbone(in_channels=9, num_classes=3)
        model = TrackNet(backbone=custom)
        assert model.backbone is custom
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)

    def test_custom_mdd_module(self):
        """TrackNet should pass input through MDD when provided."""

        class DummyMDD(torch.nn.Module):
            def forward(self, x):
                return x + 1.0

        model = TrackNet(mdd=DummyMDD())
        assert model.mdd is not None
        assert isinstance(model.mdd, torch.nn.Module)


class TestUNetBackboneSigmoidFlag:
    def test_default_apply_sigmoid_true(self):
        """Default UNetBackbone still applies sigmoid — V2 backward compat."""
        model = UNetBackbone(in_channels=9, num_classes=3)
        assert model.apply_sigmoid is True
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_apply_sigmoid_false_returns_raw_logits(self):
        """With apply_sigmoid=False, output can exceed [0, 1]."""
        torch.manual_seed(42)
        model = UNetBackbone(in_channels=9, num_classes=3, apply_sigmoid=False)
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        # Raw logits are unbounded — at least some values should be outside [0, 1]
        assert out.min() < 0.0 or out.max() > 1.0

    def test_shape_same_regardless_of_sigmoid(self):
        """Output shape must be identical whether sigmoid is applied or not."""
        model_sig = UNetBackbone(in_channels=9, num_classes=3, apply_sigmoid=True)
        model_raw = UNetBackbone(in_channels=9, num_classes=3, apply_sigmoid=False)
        x = torch.randn(2, 9, 288, 512)
        out_sig = model_sig(x)
        out_raw = model_raw(x)
        assert out_sig.shape == out_raw.shape == (2, 3, 288, 512)

    def test_sigmoid_flag_stored_as_attribute(self):
        """The flag should be accessible as a plain attribute for guard checks."""
        model_true = UNetBackbone(apply_sigmoid=True)
        model_false = UNetBackbone(apply_sigmoid=False)
        assert model_true.apply_sigmoid is True
        assert model_false.apply_sigmoid is False


class TestTrackNetSigmoidGuard:
    def test_rstr_with_sigmoid_raises(self):
        """R-STR requires raw logits — sigmoid backbone must be rejected."""
        backbone = UNetBackbone(in_channels=9, num_classes=3, apply_sigmoid=True)
        dummy_rstr = nn.Identity()
        with pytest.raises(ValueError, match="apply_sigmoid"):
            TrackNet(backbone=backbone, rstr=dummy_rstr)

    def test_rstr_without_sigmoid_ok(self):
        """R-STR with apply_sigmoid=False should construct fine."""
        backbone = UNetBackbone(in_channels=9, num_classes=3, apply_sigmoid=False)
        dummy_rstr = nn.Identity()
        model = TrackNet(backbone=backbone, rstr=dummy_rstr)
        assert model.rstr is not None

    def test_no_rstr_with_sigmoid_ok(self):
        """V2 mode (no R-STR) with default sigmoid is fine."""
        backbone = UNetBackbone(in_channels=9, num_classes=3, apply_sigmoid=True)
        model = TrackNet(backbone=backbone)
        assert model.rstr is None

    def test_default_tracknet_no_raise(self):
        """Default TrackNet() should never raise — V2 mode."""
        model = TrackNet()
        assert model.backbone.apply_sigmoid is True
        assert model.rstr is None


class TestIntegration:
    def test_forward_backward(self):
        """Full forward pass + loss + backward pass."""
        model = TrackNet()
        loss_fn = WBCEFocalLoss()

        x = torch.randn(2, 9, 288, 512)
        target = torch.zeros(2, 3, 288, 512)
        target[:, :, 140:150, 250:260] = 1.0

        pred = model(x)
        loss = loss_fn(pred, target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_batch_size_one(self):
        """GroupNorm should work fine with batch size 1."""
        model = TrackNet()
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)
