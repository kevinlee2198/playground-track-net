import math

import pytest
import torch

from models.mdd import MotionDirectionDecoupling


@pytest.fixture()
def mdd():
    """Fresh MDD module at default initialization (alpha=0, beta=0)."""
    return MotionDirectionDecoupling()


def _make_input(
    frames: tuple[float, float, float] = (0.5, 0.5, 0.5),
    batch: int = 1,
    h: int = 32,
    w: int = 32,
) -> torch.Tensor:
    """Build a (B, 9, H, W) input from three constant-valued RGB frames."""
    return torch.cat([torch.full((batch, 3, h, w), v) for v in frames], dim=1)


# -- Output shapes -----------------------------------------------------------


class TestMDDOutputShape:
    @pytest.mark.parametrize(
        "batch, h, w",
        [(2, 288, 512), (1, 288, 512), (1, 64, 128)],
        ids=["standard", "batch1", "arbitrary-spatial"],
    )
    def test_enriched_and_attention_shapes(self, mdd, batch, h, w):
        enriched, attention = mdd(torch.randn(batch, 9, h, w))
        assert enriched.shape == (batch, 13, h, w)
        assert attention.shape == (batch, 4, h, w)


# -- Attention value range ----------------------------------------------------


class TestMDDAttentionRange:
    def test_attention_in_zero_one(self, mdd):
        """All attention values must be in [0, 1] (sigmoid output)."""
        _, attention = mdd(torch.randn(2, 9, 64, 64))
        assert attention.min() >= 0.0
        assert attention.max() <= 1.0

    def test_enriched_attention_channels_in_zero_one(self, mdd):
        """Attention channels within enriched output are in [0, 1]."""
        enriched, _ = mdd(torch.randn(2, 9, 64, 64))
        # A_prev is channels 3-4, A_next is channels 8-9
        for slc in (slice(3, 5), slice(8, 10)):
            assert enriched[:, slc].min() >= 0.0
            assert enriched[:, slc].max() <= 1.0

    def test_attention_with_extreme_inputs(self, mdd):
        """Attention remains valid with large pixel differences."""
        _, attention = mdd(torch.randn(1, 9, 32, 32) * 10.0)
        assert not torch.isnan(attention).any()
        assert not torch.isinf(attention).any()
        assert attention.min() >= 0.0
        assert attention.max() <= 1.0


# -- Learnable parameters ----------------------------------------------------


class TestMDDParameters:
    def test_exactly_two_learnable_params(self, mdd):
        assert len(list(mdd.parameters())) == 2

    def test_alpha_beta_init(self, mdd):
        """Alpha and beta are zero-initialized scalars."""
        for p in (mdd.alpha, mdd.beta):
            assert p.shape == ()
            assert p.item() == 0.0

    def test_param_names(self, mdd):
        names = {n for n, _ in mdd.named_parameters()}
        assert names == {"alpha", "beta"}

    def test_total_parameter_count(self, mdd):
        """Only 2 scalar floats -- no hidden conv layers."""
        assert sum(p.numel() for p in mdd.parameters()) == 2


# -- Gradient flow ------------------------------------------------------------


class TestMDDGradientFlow:
    def test_gradient_flows_through_alpha(self):
        """Nudge alpha away from zero (where |tanh(0)| has zero gradient)."""
        mdd = MotionDirectionDecoupling()
        with torch.no_grad():
            mdd.alpha.fill_(0.5)
        enriched, _ = mdd(torch.randn(1, 9, 32, 32))
        enriched.sum().backward()
        assert mdd.alpha.grad is not None
        assert mdd.alpha.grad.abs() > 0

    def test_gradient_flows_through_beta(self, mdd):
        enriched, _ = mdd(torch.randn(1, 9, 32, 32))
        enriched.sum().backward()
        assert mdd.beta.grad is not None
        assert mdd.beta.grad.abs() > 0

    def test_gradient_from_attention_output(self, mdd):
        _, attention = mdd(torch.randn(1, 9, 32, 32))
        attention.sum().backward()
        assert mdd.alpha.grad is not None
        assert mdd.beta.grad is not None

    def test_no_gradient_on_input(self, mdd):
        x = torch.randn(1, 9, 32, 32)
        mdd(x)
        assert not x.requires_grad


# -- Static / motion contrast ------------------------------------------------


class TestMDDStaticInput:
    def test_identical_frames_produce_half_attention(self, mdd):
        """Identical frames -> zero difference -> sigmoid(0) = 0.5 at init."""
        frame = torch.rand(1, 3, 64, 64)
        x = frame.repeat(1, 3, 1, 1)  # (1, 9, 64, 64)
        _, attention = mdd(x)
        assert torch.allclose(attention, torch.full_like(attention, 0.5), atol=1e-4)

    def test_static_vs_moving_attention_contrast(self, mdd):
        """Moving input should produce higher max attention than static input."""
        _, static_att = mdd(_make_input((0.5, 0.5, 0.5), h=64, w=64))
        _, moving_att = mdd(_make_input((0.0, 1.0, 0.0), h=64, w=64))
        assert moving_att.max() > static_att.max()


# -- Numerical verification --------------------------------------------------


class TestMDDNumerical:
    def test_known_alpha_beta_output(self):
        """Manually verify output for alpha=0.5, beta=0.3 with constant frames.

        I_{t-1}=0.2, I_t=0.7, I_{t+1}=0.4  (1x1 spatial for simplicity).
        """
        mdd = MotionDirectionDecoupling()
        with torch.no_grad():
            mdd.alpha.fill_(0.5)
            mdd.beta.fill_(0.3)

        x = _make_input((0.2, 0.7, 0.4), h=1, w=1)
        enriched, attention = mdd(x)

        # Expected adaptive sigmoid parameters
        k = 5.0 / (0.45 * abs(math.tanh(0.5)) + 1e-8)
        m = 0.6 * math.tanh(0.3)

        def _sigmoid(v: float) -> float:
            return 1.0 / (1.0 + math.exp(-v))

        # D_prev=0.5, D_next=-0.3
        # P+prev=0.5, P-prev=0.0, P+next=0.0, P-next=0.3
        expected = torch.tensor(
            [
                [
                    [[_sigmoid(k * (0.5 - m))]],  # a_prev_plus
                    [[_sigmoid(k * (0.0 - m))]],  # a_prev_minus
                    [[_sigmoid(k * (0.0 - m))]],  # a_next_plus
                    [[_sigmoid(k * (0.3 - m))]],  # a_next_minus
                ]
            ]
        )
        assert torch.allclose(attention, expected, atol=1e-5), (
            f"Expected {expected.squeeze()}, got {attention.squeeze()}"
        )

        # Enriched channel layout: [I_{t-1}(3), A_prev(2), I_t(3), A_next(2), I_{t+1}(3)]
        assert torch.allclose(enriched[:, 0:3], x[:, 0:3], atol=1e-7)
        assert torch.allclose(enriched[:, 5:8], x[:, 3:6], atol=1e-7)
        assert torch.allclose(enriched[:, 10:13], x[:, 6:9], atol=1e-7)
        assert torch.allclose(enriched[:, 3:5], attention[:, 0:2], atol=1e-7)
        assert torch.allclose(enriched[:, 8:10], attention[:, 2:4], atol=1e-7)

    def test_symmetry_of_polarity(self, mdd):
        """Symmetric motion (D_prev = -D_next) swaps plus/minus channels."""
        x = _make_input((0.0, 0.5, 0.0), h=8, w=8)
        _, attention = mdd(x)

        assert torch.allclose(attention[:, 0:1], attention[:, 3:4], atol=1e-6)
        assert torch.allclose(attention[:, 1:2], attention[:, 2:3], atol=1e-6)


# -- Integration with TrackNet -----------------------------------------------


class TestMDDIntegration:
    def test_mdd_with_tracknet_v5(self):
        """Full V5 forward pass: MDD -> backbone (13ch input)."""
        from models.backbone import UNetBackbone
        from models.tracknet import TrackNet

        mdd = MotionDirectionDecoupling()
        backbone = UNetBackbone(in_channels=13, num_classes=3)
        model = TrackNet(backbone=backbone, mdd=mdd)
        out = model(torch.randn(1, 9, 288, 512))
        assert out.shape == (1, 3, 288, 512)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_v2_backward_compatible(self):
        """V2 mode (no MDD) still works."""
        from models.tracknet import TrackNet

        model = TrackNet()
        out = model(torch.randn(1, 9, 288, 512))
        assert out.shape == (1, 3, 288, 512)

    def test_v5_full_backward_pass(self):
        """Gradients flow through MDD -> backbone end-to-end."""
        from models.backbone import UNetBackbone
        from models.tracknet import TrackNet

        mdd = MotionDirectionDecoupling()
        backbone = UNetBackbone(in_channels=13, num_classes=3)
        model = TrackNet(backbone=backbone, mdd=mdd)
        out = model(torch.randn(1, 9, 64, 64))
        out.sum().backward()
        assert mdd.alpha.grad is not None
        assert mdd.beta.grad is not None
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for backbone.{name}"
