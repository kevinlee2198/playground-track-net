# V5 Motion Direction Decoupling (MDD) Module

**Feature Name:** Motion Direction Decoupling preprocessing module for TrackNet V5
**Goal:** Implement the MDD module that sits before the U-Net backbone, computing signed frame differences with learnable adaptive attention to produce a 13-channel enriched input. This replaces the raw 9-channel concatenation used in V2 and preserves trajectory direction information that absolute differences (V4) discard.
**Architecture:** Pure tensor math + 2 learnable scalar parameters (alpha, beta). No conv layers. Input: (B, 9, H, W) three concatenated RGB frames. Output: (B, 13, H, W) enriched frames interleaved with attention maps + (B, 4, H, W) attention maps for downstream R-STR fusion.
**Tech Stack:** Python 3.12+, PyTorch 2.10+, pytest

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement each task below via TDD -- write the failing test first, implement code to pass, then verify.

---

## File Map

| File | Contents |
|------|----------|
| `models/mdd.py` | `MotionDirectionDecoupling` module -- signed polarity decomposition + learnable adaptive sigmoid attention |
| `models/__init__.py` | Add `MotionDirectionDecoupling` to public API re-exports |
| `models/tracknet.py` | Update `TrackNet.forward` to unpack MDD tuple output `(enriched, attention)` and store attention for R-STR |
| `tests/test_mdd.py` | All MDD unit tests (shapes, ranges, parameters, gradients, numerical correctness) |

---

## Task 1 -- Scaffold and output shape

**Files:** `models/mdd.py`, `tests/test_mdd.py`

**Steps:**

- [ ] Create `tests/test_mdd.py` with shape tests
- [ ] Create `models/mdd.py` with `MotionDirectionDecoupling` class skeleton
- [ ] Run tests, verify they pass

### Test code (`tests/test_mdd.py`)

```python
import torch
import pytest
from models.mdd import MotionDirectionDecoupling


class TestMDDOutputShape:
    def test_enriched_shape(self):
        """MDD enriched output: (B, 13, H, W) from (B, 9, H, W)."""
        mdd = MotionDirectionDecoupling()
        x = torch.randn(2, 9, 288, 512)
        enriched, attention = mdd(x)
        assert enriched.shape == (2, 13, 288, 512)

    def test_attention_shape(self):
        """MDD attention output: (B, 4, H, W) -- 2 channels per transition."""
        mdd = MotionDirectionDecoupling()
        x = torch.randn(2, 9, 288, 512)
        enriched, attention = mdd(x)
        assert attention.shape == (2, 4, 288, 512)

    def test_batch_size_one(self):
        """Works with batch size 1."""
        mdd = MotionDirectionDecoupling()
        x = torch.randn(1, 9, 288, 512)
        enriched, attention = mdd(x)
        assert enriched.shape == (1, 13, 288, 512)
        assert attention.shape == (1, 4, 288, 512)

    def test_arbitrary_spatial_dims(self):
        """Works with non-standard spatial dimensions."""
        mdd = MotionDirectionDecoupling()
        x = torch.randn(1, 9, 64, 128)
        enriched, attention = mdd(x)
        assert enriched.shape == (1, 13, 64, 128)
        assert attention.shape == (1, 4, 64, 128)
```

### Implementation code (`models/mdd.py`)

```python
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
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_mdd.py::TestMDDOutputShape -v
```

---

## Task 2 -- Attention value range

**Files:** `tests/test_mdd.py`

**Steps:**

- [ ] Write attention range tests
- [ ] Run tests, verify they pass (no implementation changes needed -- sigmoid guarantees [0, 1])

### Test code (append to `tests/test_mdd.py`)

```python
class TestMDDAttentionRange:
    def test_attention_in_zero_one(self):
        """All attention values must be in [0, 1] (sigmoid output)."""
        mdd = MotionDirectionDecoupling()
        x = torch.randn(2, 9, 64, 64)
        _, attention = mdd(x)
        assert attention.min() >= 0.0
        assert attention.max() <= 1.0

    def test_enriched_attention_channels_in_zero_one(self):
        """Attention channels within enriched output are in [0, 1]."""
        mdd = MotionDirectionDecoupling()
        x = torch.randn(2, 9, 64, 64)
        enriched, _ = mdd(x)
        # A_prev is channels 3-4, A_next is channels 8-9
        a_prev_enriched = enriched[:, 3:5]
        a_next_enriched = enriched[:, 8:10]
        assert a_prev_enriched.min() >= 0.0
        assert a_prev_enriched.max() <= 1.0
        assert a_next_enriched.min() >= 0.0
        assert a_next_enriched.max() <= 1.0

    def test_attention_with_extreme_inputs(self):
        """Attention remains valid with large pixel differences."""
        mdd = MotionDirectionDecoupling()
        x = torch.randn(1, 9, 32, 32) * 10.0
        _, attention = mdd(x)
        assert not torch.isnan(attention).any()
        assert not torch.isinf(attention).any()
        assert attention.min() >= 0.0
        assert attention.max() <= 1.0
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_mdd.py::TestMDDAttentionRange -v
```

---

## Task 3 -- Learnable parameters (alpha and beta)

**Files:** `tests/test_mdd.py`

**Steps:**

- [ ] Write parameter constraint tests
- [ ] Run tests, verify they pass

### Test code (append to `tests/test_mdd.py`)

```python
class TestMDDParameters:
    def test_exactly_two_learnable_params(self):
        """MDD has exactly 2 learnable parameters."""
        mdd = MotionDirectionDecoupling()
        params = list(mdd.parameters())
        assert len(params) == 2

    def test_alpha_is_scalar_init_zero(self):
        mdd = MotionDirectionDecoupling()
        assert mdd.alpha.shape == ()
        assert mdd.alpha.item() == 0.0

    def test_beta_is_scalar_init_zero(self):
        mdd = MotionDirectionDecoupling()
        assert mdd.beta.shape == ()
        assert mdd.beta.item() == 0.0

    def test_params_are_named_correctly(self):
        mdd = MotionDirectionDecoupling()
        names = {name for name, _ in mdd.named_parameters()}
        assert names == {"alpha", "beta"}

    def test_total_parameter_count(self):
        """Only 2 scalar floats -- no hidden conv layers."""
        mdd = MotionDirectionDecoupling()
        total = sum(p.numel() for p in mdd.parameters())
        assert total == 2
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_mdd.py::TestMDDParameters -v
```

---

## Task 4 -- Gradient flow through alpha and beta

**Files:** `tests/test_mdd.py`

**Steps:**

- [ ] Write gradient flow tests
- [ ] Run tests, verify they pass

### Test code (append to `tests/test_mdd.py`)

```python
class TestMDDGradientFlow:
    def test_gradient_flows_through_alpha(self):
        """Backprop through enriched output produces gradient on alpha."""
        mdd = MotionDirectionDecoupling()
        x = torch.randn(1, 9, 32, 32)
        enriched, _ = mdd(x)
        loss = enriched.sum()
        loss.backward()
        assert mdd.alpha.grad is not None
        assert mdd.alpha.grad.abs() > 0

    def test_gradient_flows_through_beta(self):
        """Backprop through enriched output produces gradient on beta."""
        mdd = MotionDirectionDecoupling()
        x = torch.randn(1, 9, 32, 32)
        enriched, _ = mdd(x)
        loss = enriched.sum()
        loss.backward()
        assert mdd.beta.grad is not None
        assert mdd.beta.grad.abs() > 0

    def test_gradient_from_attention_output(self):
        """Backprop through the attention output also reaches alpha and beta."""
        mdd = MotionDirectionDecoupling()
        x = torch.randn(1, 9, 32, 32)
        _, attention = mdd(x)
        loss = attention.sum()
        loss.backward()
        assert mdd.alpha.grad is not None
        assert mdd.beta.grad is not None

    def test_no_gradient_on_input(self):
        """Input frames are data, not parameters -- no grad required."""
        mdd = MotionDirectionDecoupling()
        x = torch.randn(1, 9, 32, 32)
        enriched, attention = mdd(x)
        # x.requires_grad is False by default, so this is a sanity check
        assert not x.requires_grad
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_mdd.py::TestMDDGradientFlow -v
```

---

## Task 5 -- Static input produces low attention

**Files:** `tests/test_mdd.py`

**Steps:**

- [ ] Write static-input test
- [ ] Run tests, verify they pass

### Test code (append to `tests/test_mdd.py`)

```python
class TestMDDStaticInput:
    def test_identical_frames_low_attention(self):
        """When all 3 frames are identical, frame differences are zero,
        polarity is zero, and attention should be low.

        At init (alpha=0, beta=0): k = 5.0/(0.45*|tanh(0)|+eps) ~ very large,
        m = 0.6*tanh(0) = 0. So sigmoid(k*(0-0)) = sigmoid(0) = 0.5.
        After RGB mean: still 0.5. This is the "resting" attention for zero motion.

        With trained beta > 0, m becomes positive, pushing sigmoid(k*(0 - m))
        toward 0. But at init, 0.5 is expected.
        """
        mdd = MotionDirectionDecoupling()
        # Create 3 identical frames
        frame = torch.rand(1, 3, 64, 64)
        x = torch.cat([frame, frame, frame], dim=1)  # (1, 9, 64, 64)
        _, attention = mdd(x)
        # At initialization, identical frames produce sigmoid(0) = 0.5
        assert torch.allclose(attention, torch.full_like(attention, 0.5), atol=1e-4)

    def test_static_vs_moving_attention_contrast(self):
        """Moving input should produce higher max attention than static input."""
        mdd = MotionDirectionDecoupling()

        # Static: identical frames
        frame = torch.full((1, 3, 64, 64), 0.5)
        static_input = torch.cat([frame, frame, frame], dim=1)
        _, static_att = mdd(static_input)

        # Moving: large difference between frames
        f1 = torch.zeros(1, 3, 64, 64)
        f2 = torch.ones(1, 3, 64, 64)
        f3 = torch.zeros(1, 3, 64, 64)
        moving_input = torch.cat([f1, f2, f3], dim=1)
        _, moving_att = mdd(moving_input)

        # Moving attention max should exceed static attention max
        assert moving_att.max() > static_att.max()
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_mdd.py::TestMDDStaticInput -v
```

---

## Task 6 -- Numerical correctness test

**Files:** `tests/test_mdd.py`

**Steps:**

- [ ] Write numerical test with known alpha, beta, and input
- [ ] Run tests, verify they pass

### Test code (append to `tests/test_mdd.py`)

```python
import math


class TestMDDNumerical:
    def test_known_alpha_beta_output(self):
        """Manually compute expected output for known alpha, beta, and input.

        Setup:
            alpha = 0.5, beta = 0.3
            I_{t-1} = 0.2 (constant), I_t = 0.7 (constant), I_{t+1} = 0.4 (constant)
            All spatial positions identical (1x1 image for simplicity).

        D_prev = I_t - I_{t-1} = 0.5 (all channels)
        D_next = I_{t+1} - I_t = -0.3 (all channels)

        P_plus_prev = ReLU(0.5) = 0.5,  P_minus_prev = ReLU(-0.5) = 0.0
        P_plus_next = ReLU(-0.3) = 0.0, P_minus_next = ReLU(0.3) = 0.3

        k = 5.0 / (0.45 * |tanh(0.5)| + eps) = 5.0 / (0.45 * 0.46212... + eps)
        m = 0.6 * tanh(0.3) = 0.6 * 0.29131...

        For P_plus_prev (val=0.5): sigmoid(k * (0.5 - m))  -> per-channel, then mean over RGB
        For P_minus_prev (val=0.0): sigmoid(k * (0.0 - m)) -> per-channel, then mean over RGB
        etc.
        """
        mdd = MotionDirectionDecoupling()
        # Override alpha and beta
        with torch.no_grad():
            mdd.alpha.fill_(0.5)
            mdd.beta.fill_(0.3)

        # Constant frames: I_{t-1}=0.2, I_t=0.7, I_{t+1}=0.4
        i_prev = torch.full((1, 3, 1, 1), 0.2)
        i_curr = torch.full((1, 3, 1, 1), 0.7)
        i_next = torch.full((1, 3, 1, 1), 0.4)
        x = torch.cat([i_prev, i_curr, i_next], dim=1)  # (1, 9, 1, 1)

        enriched, attention = mdd(x)

        # Compute expected values manually
        alpha_val = 0.5
        beta_val = 0.3
        tanh_alpha = math.tanh(alpha_val)  # 0.46211715...
        tanh_beta = math.tanh(beta_val)    # 0.29131261...

        k = 5.0 / (0.45 * abs(tanh_alpha) + 1e-8)
        m = 0.6 * tanh_beta

        # D_prev = 0.5, D_next = -0.3
        # P_plus_prev=0.5, P_minus_prev=0.0, P_plus_next=0.0, P_minus_next=0.3

        def sigmoid(val):
            return 1.0 / (1.0 + math.exp(-val))

        # Attention values (sigmoid per-channel then mean across 3 identical RGB channels = same value)
        a_prev_plus = sigmoid(k * (abs(0.5) - m))     # from P_plus_prev
        a_prev_minus = sigmoid(k * (abs(0.0) - m))    # from P_minus_prev
        a_next_plus = sigmoid(k * (abs(0.0) - m))     # from P_plus_next
        a_next_minus = sigmoid(k * (abs(0.3) - m))    # from P_minus_next

        # Check attention output: [a_prev_plus, a_prev_minus, a_next_plus, a_next_minus]
        expected_att = torch.tensor([
            [[[a_prev_plus]], [[a_prev_minus]], [[a_next_plus]], [[a_next_minus]]]
        ])
        assert torch.allclose(attention, expected_att, atol=1e-5), (
            f"Expected {expected_att.squeeze()}, got {attention.squeeze()}"
        )

        # Check enriched channel layout: [I_{t-1}(3), A_prev(2), I_t(3), A_next(2), I_{t+1}(3)]
        assert torch.allclose(enriched[:, 0:3], i_prev, atol=1e-7)   # I_{t-1}
        assert torch.allclose(enriched[:, 5:8], i_curr, atol=1e-7)   # I_t
        assert torch.allclose(enriched[:, 10:13], i_next, atol=1e-7)  # I_{t+1}

        # Attention channels within enriched match attention output
        assert torch.allclose(enriched[:, 3:5], attention[:, 0:2], atol=1e-7)  # A_prev
        assert torch.allclose(enriched[:, 8:10], attention[:, 2:4], atol=1e-7)  # A_next

    def test_symmetry_of_polarity(self):
        """If D_prev = -D_next (symmetric motion), the attention maps should
        have swapped plus/minus channels but same magnitudes."""
        mdd = MotionDirectionDecoupling()
        # I_{t-1}=0.0, I_t=0.5, I_{t+1}=0.0
        # D_prev = 0.5 - 0.0 = 0.5,  D_next = 0.0 - 0.5 = -0.5
        # P_plus_prev = 0.5, P_minus_prev = 0.0
        # P_plus_next = 0.0, P_minus_next = 0.5
        # So a_prev_plus == a_next_minus and a_prev_minus == a_next_plus
        f1 = torch.zeros(1, 3, 8, 8)
        f2 = torch.full((1, 3, 8, 8), 0.5)
        f3 = torch.zeros(1, 3, 8, 8)
        x = torch.cat([f1, f2, f3], dim=1)
        _, attention = mdd(x)

        a_prev_plus = attention[:, 0:1]
        a_prev_minus = attention[:, 1:2]
        a_next_plus = attention[:, 2:3]
        a_next_minus = attention[:, 3:4]

        assert torch.allclose(a_prev_plus, a_next_minus, atol=1e-6)
        assert torch.allclose(a_prev_minus, a_next_plus, atol=1e-6)
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_mdd.py::TestMDDNumerical -v
```

---

## Task 7 -- Update TrackNet wrapper and `__init__.py`

**Files:** `models/tracknet.py`, `models/__init__.py`, `tests/test_mdd.py`

**Steps:**

- [ ] Write integration test for MDD + TrackNet
- [ ] Update `TrackNet.forward` to unpack MDD tuple `(enriched, attention)`, passing `enriched` to backbone and storing `attention` on `self` for downstream R-STR access
- [ ] Add `MotionDirectionDecoupling` to `models/__init__.py`
- [ ] Run tests, verify they pass

### Test code (append to `tests/test_mdd.py`)

```python
from models.backbone import UNetBackbone
from models.tracknet import TrackNet


class TestMDDIntegration:
    def test_mdd_with_tracknet_v5(self):
        """Full V5 forward pass: MDD -> backbone (13ch input)."""
        mdd = MotionDirectionDecoupling()
        backbone = UNetBackbone(in_channels=13, num_classes=3)
        model = TrackNet(backbone=backbone, mdd=mdd)
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_v2_backward_compatible(self):
        """V2 mode (no MDD) still works after code change."""
        model = TrackNet()
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)
        assert not hasattr(model, '_mdd_attention') or model._mdd_attention is None

    def test_v5_full_backward_pass(self):
        """Gradients flow through MDD -> backbone end-to-end."""
        mdd = MotionDirectionDecoupling()
        backbone = UNetBackbone(in_channels=13, num_classes=3)
        model = TrackNet(backbone=backbone, mdd=mdd)
        x = torch.randn(1, 9, 64, 64)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # MDD alpha and beta should have gradients
        assert mdd.alpha.grad is not None
        assert mdd.beta.grad is not None
        # Backbone params should also have gradients
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for backbone.{name}"
```

### Implementation changes

**`models/tracknet.py`** -- DO NOT rewrite `forward()`. Plan 1 (sigmoid refactor) already wrote the correct version with tuple unpacking and attention passthrough. Only update `__init__.py` exports here.

**`models/__init__.py`** -- add MDD to exports:

```python
from models.backbone import UNetBackbone
from models.losses import WBCEFocalLoss
from models.mdd import MotionDirectionDecoupling
from models.tracknet import TrackNet

__all__ = ["MotionDirectionDecoupling", "TrackNet", "UNetBackbone", "WBCEFocalLoss"]
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_mdd.py -v
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py -v
```

---

## Interface Summary

**What this module provides to others:**

| Consumer | Interface | Description |
|----------|-----------|-------------|
| TrackNet wrapper | `enriched, attention = mdd(x)` | `(B, 9, H, W)` -> `(B, 13, H, W)` enriched + `(B, 4, H, W)` attention |
| R-STR head (future) | `model._mdd_attention` | `(B, 4, H, W)` attention maps stored after forward pass, for fusion with draft logits |
| Training loop | `mdd.alpha`, `mdd.beta` | 2 learnable scalars, trained end-to-end via standard optimizer |

**What this module expects from others:**

| Provider | Interface | Description |
|----------|-----------|-------------|
| Data pipeline | Input tensor `(B, 9, H, W)` | 3 RGB frames concatenated channel-wise, normalized to [0, 1] |
| UNetBackbone | `in_channels=13` | Backbone must accept 13 input channels when MDD is active |

**Enriched channel layout:**

| Channel indices | Content | Source |
|-----------------|---------|--------|
| 0-2 | I_{t-1} (RGB) | Passthrough from input |
| 3-4 | A_prev (plus, minus) | Learnable attention on D_prev polarity |
| 5-7 | I_t (RGB) | Passthrough from input |
| 8-9 | A_next (plus, minus) | Learnable attention on D_next polarity |
| 10-12 | I_{t+1} (RGB) | Passthrough from input |
