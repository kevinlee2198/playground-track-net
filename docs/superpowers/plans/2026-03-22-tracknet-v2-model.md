# TrackNet V2 Model Subsystem

**Feature Name:** TrackNet V2 U-Net backbone, model wrapper, and WBCE loss
**Goal:** Implement the complete model subsystem — backbone encoder-decoder with skip connections, TrackNet wrapper with pluggable architecture, and focal-variant WBCE loss — ready for the training and data subsystems to consume.
**Architecture:** U-Net encoder (3 down blocks) + bottleneck + decoder (3 up blocks with skip connections) + Conv1x1 sigmoid output head. GroupNorm (num_groups=8) instead of BatchNorm. Kaiming uniform weight init.
**Tech Stack:** Python 3.12+, PyTorch 2.10+, pytest

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement each task below via TDD — write the failing test first, implement code to pass, then verify.

---

## File Map

| File | Contents |
|------|----------|
| `models/__init__.py` | Public API re-exports: `TrackNet`, `UNetBackbone`, `WBCEFocalLoss` |
| `models/backbone.py` | `ConvBlock`, `DownBlock`, `UpBlock`, `Bottleneck`, `UNetBackbone` |
| `models/tracknet.py` | `TrackNet` model wrapper (assembles backbone + output head, optional MDD/R-STR slots) |
| `models/losses.py` | `WBCEFocalLoss` — weighted binary cross-entropy with focal-style dynamic weights |
| `tests/test_models.py` | All model unit tests (shapes, skip connections, loss values, weight init, forward pass) |

---

## Task 1 — Scaffold files and ConvBlock

**Files:** `models/__init__.py`, `models/backbone.py`, `tests/test_models.py`

**Steps:**

- [ ] Create `models/__init__.py` with empty placeholder
- [ ] Create `models/backbone.py` with `ConvBlock` class
- [ ] Create `tests/test_models.py` with `ConvBlock` tests
- [ ] Run tests, verify they pass

### Test code (`tests/test_models.py`)

```python
import torch
import pytest
from models.backbone import ConvBlock


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
        assert isinstance(block.norm, torch.nn.GroupNorm)
        assert block.norm.num_groups == 8

    def test_kaiming_init(self):
        """Conv weights should not be all zeros (Kaiming init)."""
        block = ConvBlock(in_channels=64, out_channels=128)
        assert block.conv.weight.abs().sum() > 0

    def test_groupnorm_init(self):
        """GroupNorm weight=1, bias=0."""
        block = ConvBlock(in_channels=64, out_channels=128)
        assert torch.allclose(block.norm.weight, torch.ones_like(block.norm.weight))
        assert torch.allclose(block.norm.bias, torch.zeros_like(block.norm.bias))
```

### Implementation code (`models/backbone.py`)

```python
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
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestConvBlock -v
```

---

## Task 2 — DownBlock

**Files:** `models/backbone.py`, `tests/test_models.py`

**Steps:**

- [ ] Write `TestDownBlock` tests
- [ ] Implement `DownBlock`
- [ ] Run tests, verify they pass

### Test code (append to `tests/test_models.py`)

```python
from models.backbone import DownBlock


class TestDownBlock:
    def test_output_shapes(self):
        """DownBlock returns (pooled, skip) where skip is pre-pool."""
        block = DownBlock(in_channels=9, out_channels=64)
        x = torch.randn(2, 9, 288, 512)
        pooled, skip = block(x)
        # skip is pre-pool (same spatial as input)
        assert skip.shape == (2, 64, 288, 512)
        # pooled is after MaxPool2x2 (half spatial)
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
```

### Implementation code (add to `models/backbone.py`)

```python
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
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestDownBlock -v
```

---

## Task 3 — Bottleneck

**Files:** `models/backbone.py`, `tests/test_models.py`

**Steps:**

- [ ] Write `TestBottleneck` tests
- [ ] Implement `Bottleneck`
- [ ] Run tests, verify they pass

### Test code (append to `tests/test_models.py`)

```python
from models.backbone import Bottleneck


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
```

### Implementation code (add to `models/backbone.py`)

```python
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
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestBottleneck -v
```

---

## Task 4 — UpBlock

**Files:** `models/backbone.py`, `tests/test_models.py`

**Steps:**

- [ ] Write `TestUpBlock` tests
- [ ] Implement `UpBlock`
- [ ] Run tests, verify they pass

### Test code (append to `tests/test_models.py`)

```python
from models.backbone import UpBlock


class TestUpBlock:
    def test_up1_shape(self):
        """Up1: in=512+256=768, out=256, spatial 64x36 -> 128x72."""
        block = UpBlock(in_channels=768, out_channels=256)
        x = torch.randn(2, 512, 36, 64)  # bottleneck output
        skip = torch.randn(2, 256, 72, 128)  # skip from Down3
        out = block(x, skip)
        assert out.shape == (2, 256, 72, 128)

    def test_up2_shape(self):
        """Up2: in=256+128=384, out=128, spatial 128x72 -> 256x144."""
        block = UpBlock(in_channels=384, out_channels=128)
        x = torch.randn(2, 256, 72, 128)
        skip = torch.randn(2, 128, 144, 256)
        out = block(x, skip)
        assert out.shape == (2, 128, 144, 256)

    def test_up3_shape(self):
        """Up3: in=128+64=192, out=64, spatial 256x144 -> 512x288."""
        block = UpBlock(in_channels=192, out_channels=64)
        x = torch.randn(2, 128, 144, 256)
        skip = torch.randn(2, 64, 288, 512)
        out = block(x, skip)
        assert out.shape == (2, 64, 288, 512)

    def test_has_two_conv_blocks(self):
        block = UpBlock(in_channels=768, out_channels=256)
        assert isinstance(block.conv1, ConvBlock)
        assert isinstance(block.conv2, ConvBlock)
```

### Implementation code (add to `models/backbone.py`)

```python
class UpBlock(nn.Module):
    """Upsample2x + skip concat + 2x ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestUpBlock -v
```

---

## Task 5 — UNetBackbone

**Files:** `models/backbone.py`, `tests/test_models.py`

**Steps:**

- [ ] Write `TestUNetBackbone` tests
- [ ] Implement `UNetBackbone`
- [ ] Run tests, verify they pass

### Test code (append to `tests/test_models.py`)

```python
from models.backbone import UNetBackbone


class TestUNetBackbone:
    def test_output_shape(self):
        """Full backbone: (batch, 9, 288, 512) -> (batch, 3, 288, 512)."""
        model = UNetBackbone(in_channels=9, num_classes=3)
        x = torch.randn(2, 9, 288, 512)
        out = model(x)
        assert out.shape == (2, 3, 288, 512)

    def test_output_range_sigmoid(self):
        """Output should be in [0, 1] after sigmoid."""
        model = UNetBackbone(in_channels=9, num_classes=3)
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_v5_input_channels(self):
        """V5 mode: 13 input channels."""
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
        """Sanity check: V2 backbone should be ~7-12M params."""
        model = UNetBackbone(in_channels=9, num_classes=3)
        total = sum(p.numel() for p in model.parameters())
        assert 1_000_000 < total < 50_000_000
```

### Implementation code (add to `models/backbone.py`)

```python
class UNetBackbone(nn.Module):
    """V2 U-Net encoder-decoder with skip connections and sigmoid output head."""

    def __init__(self, in_channels: int = 9, num_classes: int = 3) -> None:
        super().__init__()
        # Encoder
        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)

        # Bottleneck
        self.bottleneck = Bottleneck(256, 512)

        # Decoder
        self.up1 = UpBlock(512 + 256, 256)  # 768 -> 256
        self.up2 = UpBlock(256 + 128, 128)  # 384 -> 128
        self.up3 = UpBlock(128 + 64, 64)    # 192 -> 64

        # Output head
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self._init_head()

    def _init_head(self) -> None:
        nn.init.kaiming_uniform_(self.head.weight, nonlinearity="relu")
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        d1, skip1 = self.down1(x)
        d2, skip2 = self.down2(d1)
        d3, skip3 = self.down3(d2)

        # Bottleneck
        b = self.bottleneck(d3)

        # Decoder with skip connections
        u1 = self.up1(b, skip3)
        u2 = self.up2(u1, skip2)
        u3 = self.up3(u2, skip1)

        # Output
        return self.sigmoid(self.head(u3))
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestUNetBackbone -v
```

---

## Task 6 — WBCE Focal Loss

**Files:** `models/losses.py`, `tests/test_models.py`

**Steps:**

- [ ] Write `TestWBCEFocalLoss` tests
- [ ] Implement `WBCEFocalLoss` in `models/losses.py`
- [ ] Run tests, verify they pass

### Test code (append to `tests/test_models.py`)

```python
from models.losses import WBCEFocalLoss


class TestWBCEFocalLoss:
    def test_returns_scalar(self):
        loss_fn = WBCEFocalLoss()
        pred = torch.sigmoid(torch.randn(2, 3, 288, 512))
        target = torch.zeros(2, 3, 288, 512)
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_perfect_prediction_low_loss(self):
        """Perfect prediction should give very low loss."""
        loss_fn = WBCEFocalLoss()
        target = torch.zeros(1, 3, 32, 32)
        target[:, :, 15:17, 15:17] = 1.0
        pred = target.clone().clamp(1e-6, 1 - 1e-6)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01

    def test_bad_prediction_high_loss(self):
        """Inverted prediction should give high loss."""
        loss_fn = WBCEFocalLoss()
        target = torch.zeros(1, 3, 32, 32)
        target[:, :, 15:17, 15:17] = 1.0
        pred = (1.0 - target).clamp(1e-6, 1 - 1e-6)
        loss = loss_fn(pred, target)
        assert loss.item() > 1.0

    def test_all_zero_target(self):
        """All-zero target (no ball) should still produce valid loss."""
        loss_fn = WBCEFocalLoss()
        pred = torch.full((1, 3, 32, 32), 0.1)
        target = torch.zeros(1, 3, 32, 32)
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_continuous_targets_mixup(self):
        """Continuous targets from mixup should work."""
        loss_fn = WBCEFocalLoss()
        pred = torch.sigmoid(torch.randn(1, 3, 32, 32))
        target = torch.rand(1, 3, 32, 32)  # continuous [0,1]
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flows(self):
        """Loss should produce gradients for backprop."""
        loss_fn = WBCEFocalLoss()
        pred = torch.sigmoid(torch.randn(1, 3, 32, 32, requires_grad=True))
        target = torch.zeros(1, 3, 32, 32)
        target[:, :, 15:17, 15:17] = 1.0
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0
```

### Implementation code (`models/losses.py`)

```python
import torch
import torch.nn as nn


class WBCEFocalLoss(nn.Module):
    """Weighted Binary Cross-Entropy with focal-style dynamic weights.

    L = -1/N * sum[(1-p)^2 * y * log(p) + p^2 * (1-y) * log(1-p)]

    Expects predictions after sigmoid (probabilities in [0, 1]).
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.clamp(self.eps, 1.0 - self.eps)
        pos_weight = (1.0 - p) ** 2
        neg_weight = p ** 2
        loss = -(
            pos_weight * target * torch.log(p)
            + neg_weight * (1.0 - target) * torch.log(1.0 - p)
        )
        return loss.mean()
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestWBCEFocalLoss -v
```

---

## Task 7 — TrackNet Wrapper

**Files:** `models/tracknet.py`, `tests/test_models.py`

**Steps:**

- [ ] Write `TestTrackNet` tests
- [ ] Implement `TrackNet` in `models/tracknet.py`
- [ ] Run tests, verify they pass

### Test code (append to `tests/test_models.py`)

```python
from models.tracknet import TrackNet


class TestTrackNet:
    def test_v2_forward(self):
        """V2 mode: no MDD, no R-STR."""
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
        """V2 mode: mdd and rstr are None."""
        model = TrackNet()
        assert model.mdd is None
        assert model.rstr is None

    def test_backbone_accessible(self):
        model = TrackNet()
        assert isinstance(model.backbone, UNetBackbone)
```

### Implementation code (`models/tracknet.py`)

```python
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
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestTrackNet -v
```

---

## Task 8 — Public API and `__init__.py`

**Files:** `models/__init__.py`

**Steps:**

- [ ] Populate `models/__init__.py` with re-exports
- [ ] Run full test suite
- [ ] Commit

### Implementation code (`models/__init__.py`)

```python
from models.backbone import UNetBackbone
from models.losses import WBCEFocalLoss
from models.tracknet import TrackNet

__all__ = ["TrackNet", "UNetBackbone", "WBCEFocalLoss"]
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py -v
```

---

## Task 9 — Integration test: full forward + backward pass

**Files:** `tests/test_models.py`

**Steps:**

- [ ] Add integration test
- [ ] Run it, verify it passes
- [ ] Commit all model subsystem work

### Test code (append to `tests/test_models.py`)

```python
class TestIntegration:
    def test_forward_backward(self):
        """Full forward pass + loss + backward pass."""
        model = TrackNet()
        loss_fn = WBCEFocalLoss()

        x = torch.randn(2, 9, 288, 512)
        target = torch.zeros(2, 3, 288, 512)
        target[:, :, 140:150, 250:260] = 1.0  # ball region

        pred = model(x)
        loss = loss_fn(pred, target)
        loss.backward()

        # Gradients should flow through entire model
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_batch_size_one(self):
        """GroupNorm should work fine with batch size 1."""
        model = TrackNet()
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py -v
```

---

## Interface Summary

**What this subsystem provides to others:**

| Consumer | Interface | Description |
|----------|-----------|-------------|
| Training subsystem | `model = TrackNet()` | Instantiate V2 model |
| Training subsystem | `heatmaps = model(input_tensor)` | Forward pass: `(B, 9, 288, 512)` -> `(B, 3, 288, 512)` |
| Training subsystem | `loss = WBCEFocalLoss()(pred, target)` | Scalar loss for backprop |
| Inference subsystem | `heatmaps = model(input_tensor)` | Same forward pass |
| V5 (future) | `TrackNet(backbone, mdd=mdd_module, rstr=rstr_head)` | Plug in MDD + R-STR |

**What this subsystem expects from others:**

| Provider | Interface | Description |
|----------|-----------|-------------|
| Data subsystem | Input tensor `(B, 9, 288, 512)` | 3 RGB frames concatenated, normalized to [0, 1] |
| Data subsystem | Target heatmaps `(B, 3, 288, 512)` | Binary (or continuous with mixup) ground truth |
