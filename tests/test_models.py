import pytest
import torch
import torch.nn as nn

from models.backbone import ConvBlock, DownBlock, Bottleneck, UpBlock, UNetBackbone
from models.losses import WBCEFocalLoss
from models.rstr import FactorizedAttentionLayer, RSTRHead, TSATTHead
from models.tracknet import TrackNet, tracknet_v5


def _make_dummy_mdd() -> torch.nn.Module:
    """Create a dummy MDD module that returns (enriched, attention) tuple."""

    class DummyMDD(torch.nn.Module):
        def forward(self, x):
            enriched = x
            attention = torch.ones(x.shape[0], 2, x.shape[2], x.shape[3])
            return enriched, attention

    return DummyMDD()


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
        """TrackNet should pass input through MDD when provided.
        MDD forward returns (enriched, attention) tuple — V5 interface.
        """
        model = TrackNet(mdd=_make_dummy_mdd())
        assert model.mdd is not None
        assert isinstance(model.mdd, torch.nn.Module)

    def test_mdd_tuple_unpacked_in_forward(self):
        """When MDD returns (enriched, attention), forward unpacks correctly."""
        model = TrackNet(mdd=_make_dummy_mdd())
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)

    def test_mdd_plain_tensor_backward_compat(self):
        """MDD returning a plain tensor (not tuple) still works."""

        class SimpleMDD(torch.nn.Module):
            def forward(self, x):
                return x  # no attention, plain tensor

        model = TrackNet(mdd=SimpleMDD())
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)

    def test_mdd_attention_passed_to_rstr(self):
        """When MDD returns (enriched, attention), attention is forwarded to R-STR."""
        captured = {}

        class CapturingRSTR(torch.nn.Module):
            def forward(self, logits, attention):
                captured["attention"] = attention
                return torch.sigmoid(logits)

        backbone = UNetBackbone(in_channels=9, num_classes=3, apply_sigmoid=False)
        model = TrackNet(backbone=backbone, mdd=_make_dummy_mdd(), rstr=CapturingRSTR())
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)
        assert "attention" in captured
        assert captured["attention"].shape == (1, 2, 288, 512)


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


class TestFactorizedAttentionLayer:
    def test_output_shape(self):
        """Output shape matches input: (B, T*S, D)."""
        layer = FactorizedAttentionLayer(
            embed_dim=128,
            num_heads=4,
            ff_dim=256,
            num_frames=3,
            num_patches=576,
        )
        x = torch.randn(2, 1728, 128)  # B=2, T*S=3*576=1728, D=128
        out = layer(x)
        assert out.shape == (2, 1728, 128)

    def test_gradient_flows(self):
        """Gradients should flow through both temporal and spatial attention."""
        layer = FactorizedAttentionLayer(
            embed_dim=128,
            num_heads=4,
            ff_dim=256,
            num_frames=3,
            num_patches=576,
        )
        x = torch.randn(2, 1728, 128, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_small_input(self):
        """Verify with minimal dimensions for fast testing."""
        layer = FactorizedAttentionLayer(
            embed_dim=32,
            num_heads=4,
            ff_dim=64,
            num_frames=3,
            num_patches=4,
        )
        x = torch.randn(1, 12, 32)  # T*S=3*4=12
        out = layer(x)
        assert out.shape == (1, 12, 32)

    def test_residual_connection(self):
        """Output should not be identical to input but should be correlated
        (residual connections keep them close at init)."""
        layer = FactorizedAttentionLayer(
            embed_dim=32,
            num_heads=4,
            ff_dim=64,
            num_frames=3,
            num_patches=4,
        )
        x = torch.randn(1, 12, 32)
        out = layer(x)
        # They should differ (attention modifies the signal)
        assert not torch.allclose(x, out, atol=1e-6)


class TestTSATTHead:
    def test_output_shape(self):
        """TSATTHead: (B, 3, 288, 512) -> (B, 3, 288, 512)."""
        head = TSATTHead()
        x = torch.randn(2, 3, 288, 512)
        out = head(x)
        assert out.shape == (2, 3, 288, 512)

    def test_small_residuals_at_init(self):
        """Zero-init output projection means initial output should be near-zero."""
        head = TSATTHead()
        x = torch.randn(1, 3, 288, 512)
        out = head(x)
        assert out.abs().max() < 0.1, (
            f"Initial residuals should be near-zero, got max={out.abs().max():.4f}"
        )

    def test_gradient_flow(self):
        """Gradients should flow from output back to input."""
        head = TSATTHead()
        # Perturb zero-init output projection so gradients are non-zero
        with torch.no_grad():
            head.output_proj.weight.add_(0.01)
        x = torch.randn(1, 3, 288, 512, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_parameter_count(self):
        """TSATTHead should be lightweight: expect ~200K-800K params.
        Must stay small to satisfy V5 paper's 3.7% FLOP increase constraint."""
        head = TSATTHead()
        total = sum(p.numel() for p in head.parameters())
        assert 100_000 < total < 2_000_000, f"Unexpected param count: {total:,}"

    def test_batch_size_one(self):
        """Should work with batch size 1."""
        head = TSATTHead()
        x = torch.randn(1, 3, 288, 512)
        out = head(x)
        assert out.shape == (1, 3, 288, 512)

    def test_deterministic_in_eval(self):
        """In eval mode, same input should produce same output."""
        head = TSATTHead()
        head.eval()
        x = torch.randn(1, 3, 288, 512)
        out1 = head(x)
        out2 = head(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_patchify_unpatchify_roundtrip(self):
        """Patchify followed by unpatchify should recover the original frame."""
        head = TSATTHead()
        frame = torch.randn(1, 1, 288, 512)
        patches = head._patchify(frame)
        recovered = head._unpatchify(patches)
        assert torch.allclose(frame, recovered, atol=1e-6), (
            "Patchify/unpatchify should be exact inverses"
        )


class TestRSTRHead:
    def test_output_shape(self):
        """RSTRHead: logits (B, 3, 288, 512) + attention (B, 4, 288, 512)
        -> refined heatmaps (B, 3, 288, 512)."""
        head = RSTRHead()
        logits = torch.randn(2, 3, 288, 512)
        attention = torch.randn(2, 4, 288, 512)
        out = head(logits, attention)
        assert out.shape == (2, 3, 288, 512)

    def test_output_in_sigmoid_range(self):
        """Output must be in [0, 1] (sigmoid applied)."""
        head = RSTRHead()
        head.eval()
        logits = torch.randn(1, 3, 288, 512)
        attention = torch.randn(1, 4, 288, 512)
        out = head(logits, attention)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_dropout_active_in_train(self):
        """In training mode, stochastic masking should produce different
        outputs across calls (with high probability)."""
        head = RSTRHead()
        # Perturb zero-init output projection so dropout effect is visible
        with torch.no_grad():
            head.tsatt.output_proj.weight.add_(0.01)
        head.train()
        logits = torch.randn(1, 3, 288, 512)
        attention = torch.randn(1, 4, 288, 512)
        out1 = head(logits, attention)
        out2 = head(logits, attention)
        # Dropout should cause different outputs in training
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_deterministic_in_inference(self):
        """In inference mode, dropout is off, so same input -> same output."""
        head = RSTRHead()
        head.eval()
        logits = torch.randn(1, 3, 288, 512)
        attention = torch.randn(1, 4, 288, 512)
        out1 = head(logits, attention)
        out2 = head(logits, attention)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_gradient_flow(self):
        """Gradients should flow through both logits and attention inputs."""
        head = RSTRHead()
        logits = torch.randn(1, 3, 288, 512, requires_grad=True)
        attention = torch.randn(1, 4, 288, 512, requires_grad=True)
        out = head(logits, attention)
        out.sum().backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0
        assert attention.grad is not None
        assert attention.grad.abs().sum() > 0

    def test_none_attention_raises(self):
        """RSTRHead requires MDD attention; None should raise ValueError."""
        head = RSTRHead()
        logits = torch.randn(1, 3, 288, 512)
        with pytest.raises(ValueError, match="attention"):
            head(logits, None)

    def test_residual_uses_pre_dropout_draft(self):
        """The residual should use pre-dropout draft_mdd, not the masked
        version. With zero delta (at init), output should equal
        sigmoid(draft_mdd) regardless of dropout state."""
        head = RSTRHead()

        logits = torch.randn(1, 3, 288, 512)
        attention = torch.randn(1, 4, 288, 512)

        # Get draft_mdd by running just the fusion conv
        with torch.no_grad():
            fused = torch.cat([logits, attention], dim=1)
            draft_mdd = head.fusion_conv(fused)
            expected = torch.sigmoid(draft_mdd)

        # At init, TSATTHead output projection is zero-initialized,
        # so delta should be near-zero. In inference mode (no dropout),
        # output should be close to sigmoid(draft_mdd).
        head.eval()
        out = head(logits, attention)
        assert torch.allclose(out, expected, atol=0.05), (
            "At init, RSTRHead output should approximate sigmoid(draft_mdd)"
        )


class TestTrackNetV5Flow:
    def test_v5_forward_with_mock_mdd_and_rstr(self):
        """V5 flow: MDD returns (input, attention), backbone returns logits,
        RSTR receives (logits, attention)."""

        class MockMDD(nn.Module):
            def forward(self, x):
                B = x.shape[0]
                enriched = torch.randn(B, 13, 288, 512)
                attention = torch.randn(B, 4, 288, 512)
                return enriched, attention

        class MockRSTR(nn.Module):
            def forward(self, logits, attention):
                assert logits.shape == (logits.shape[0], 3, 288, 512)
                assert attention.shape == (logits.shape[0], 4, 288, 512)
                return torch.sigmoid(logits)

        backbone = UNetBackbone(in_channels=13, num_classes=3, apply_sigmoid=False)
        model = TrackNet(backbone=backbone, mdd=MockMDD(), rstr=MockRSTR())
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)

    def test_v2_backward_compatible(self):
        """V2 mode should still work unchanged after TrackNet update."""
        model = TrackNet()
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestTrackNetV5Factory:
    def test_factory_creates_all_components(self):
        """tracknet_v5() should create MDD, backbone, and RSTR."""
        model = tracknet_v5()
        assert model.mdd is not None
        assert model.backbone is not None
        assert model.rstr is not None

    def test_backbone_input_channels(self):
        """V5 backbone should accept 13 channels."""
        model = tracknet_v5()
        # Inspect first conv layer input channels
        first_conv = model.backbone.down1.conv1.conv
        assert first_conv.in_channels == 13

    def test_backbone_no_sigmoid(self):
        """V5 backbone should return raw logits (apply_sigmoid=False)."""
        model = tracknet_v5()
        assert not model.backbone.apply_sigmoid

    def test_rstr_is_rstr_head(self):
        """RSTR component should be an RSTRHead instance."""
        model = tracknet_v5()
        assert isinstance(model.rstr, RSTRHead)

    def test_forward_pass(self):
        """Full V5 forward pass: (B, 9, 288, 512) -> (B, 3, 288, 512)."""
        model = tracknet_v5()
        model.eval()
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_param_count_increase(self):
        """V5 should have more params than V2 but not excessively more."""
        v2 = TrackNet()
        v5 = tracknet_v5()
        v2_params = sum(p.numel() for p in v2.parameters())
        v5_params = sum(p.numel() for p in v5.parameters())
        # V5 adds MDD (~2 params) + backbone channel increase + RSTR (~200K-800K)
        assert v5_params > v2_params
        # Should not more than double the V2 param count
        assert v5_params < v2_params * 2, (
            f"V5 params ({v5_params:,}) should not be >2x V2 ({v2_params:,})"
        )


class TestV5Integration:
    def test_end_to_end_forward_backward(self):
        """Full V5: forward pass + WBCE loss + backward pass."""
        model = tracknet_v5()
        loss_fn = WBCEFocalLoss()

        x = torch.randn(2, 9, 288, 512)
        target = torch.zeros(2, 3, 288, 512)
        target[:, :, 140:150, 250:260] = 1.0

        pred = model(x)
        loss = loss_fn(pred, target)
        loss.backward()

        # Gradients should flow through entire V5 model
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_v2_still_works(self):
        """V2 model must still work after V5 additions."""
        model = TrackNet()
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_v5_output_range(self):
        """V5 output should be in [0, 1] (sigmoid at end of R-STR)."""
        model = tracknet_v5()
        model.eval()
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_weight_transfer_shape_compatibility(self):
        """V2 backbone weights (except first conv) should be shape-compatible
        with V5 backbone for weight transfer."""
        v2_backbone = UNetBackbone(in_channels=9, num_classes=3)
        v5_backbone = UNetBackbone(in_channels=13, num_classes=3, apply_sigmoid=False)

        v2_state = v2_backbone.state_dict()
        v5_state = v5_backbone.state_dict()

        mismatches = []
        for key in v2_state:
            if key in v5_state:
                if v2_state[key].shape != v5_state[key].shape:
                    mismatches.append(key)

        # Only the first conv layer should differ (9ch -> 13ch)
        assert len(mismatches) == 1, f"Expected 1 mismatch, got: {mismatches}"
        assert "down1.conv1.conv.weight" in mismatches[0]
