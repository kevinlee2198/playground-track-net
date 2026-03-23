# V5 Sigmoid Refactor — Conditional Sigmoid in UNetBackbone

**Feature Name:** Make UNetBackbone sigmoid application configurable for R-STR raw logit passthrough
**Goal:** Add an `apply_sigmoid` flag to `UNetBackbone` so V5's R-STR head receives raw logits instead of double-sigmoid-compressed probabilities. V2 default behavior is fully preserved. A guard in `TrackNet.__init__` prevents accidental double-sigmoid when R-STR is attached.
**Architecture:** Single boolean flag on `UNetBackbone.__init__`, conditional in `forward()`, validation in `TrackNet.__init__`. Update `DummyMDD` test to anticipate V5 forward pass where MDD returns `(enriched, attention)` tuple.
**Tech Stack:** Python 3.12+, PyTorch 2.10+, pytest

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement each task below via TDD — write the failing test first, implement code to pass, then verify.

---

## File Map

| File | Contents |
|------|----------|
| `models/backbone.py` | Add `apply_sigmoid: bool = True` param; conditional sigmoid in `forward()` |
| `models/tracknet.py` | Add double-sigmoid `ValueError` guard; handle MDD `(enriched, attention)` tuple; pass attention to R-STR |
| `tests/test_models.py` | New: `TestUNetBackboneSigmoidFlag`, `TestTrackNetSigmoidGuard`; Updated: `TestTrackNetCustomBackbone.test_custom_mdd_module` |

---

## Task 1 — Tests for `apply_sigmoid` flag on UNetBackbone

**Files:** `tests/test_models.py`

**Steps:**

- [ ] Add `TestUNetBackboneSigmoidFlag` test class with four tests
- [ ] Run tests, verify they FAIL (flag not yet implemented)

### Test code (append to `tests/test_models.py`)

```python
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
```

### Verify (expect FAIL)

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestUNetBackboneSigmoidFlag -v
```

---

## Task 2 — Implement `apply_sigmoid` flag on UNetBackbone

**Files:** `models/backbone.py`

**Steps:**

- [ ] Add `apply_sigmoid: bool = True` parameter to `UNetBackbone.__init__`
- [ ] Store it as `self.apply_sigmoid = apply_sigmoid`
- [ ] Change `forward()` return to conditionally apply sigmoid
- [ ] Run Task 1 tests, verify they PASS
- [ ] Run ALL existing tests, verify no regressions

### Implementation changes (`models/backbone.py`)

In `UNetBackbone.__init__`, change the signature and store the flag:

```python
class UNetBackbone(nn.Module):
    """V2 U-Net encoder-decoder with skip connections and sigmoid output head."""

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 3,
        apply_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
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
```

In `UNetBackbone.forward`, change the return statement:

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1, skip1 = self.down1(x)
        d2, skip2 = self.down2(d1)
        d3, skip3 = self.down3(d2)
        b = self.bottleneck(d3)
        u1 = self.up1(b, skip3)
        u2 = self.up2(u1, skip2)
        u3 = self.up3(u2, skip1)
        logits = self.head(u3)
        return self.sigmoid(logits) if self.apply_sigmoid else logits
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestUNetBackboneSigmoidFlag -v
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py -v
```

---

## Task 3 — Tests for double-sigmoid guard in TrackNet

**Files:** `tests/test_models.py`

**Steps:**

- [ ] Add `TestTrackNetSigmoidGuard` test class
- [ ] Run tests, verify they FAIL (guard not yet implemented)

### Test code (append to `tests/test_models.py`)

```python
import pytest


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
```

### Verify (expect FAIL)

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestTrackNetSigmoidGuard -v
```

---

## Task 4 — Implement double-sigmoid guard and MDD tuple handling in TrackNet

**Files:** `models/tracknet.py`

**Steps:**

- [ ] Add `ValueError` guard in `TrackNet.__init__` for rstr + sigmoid combination
- [ ] Update `TrackNet.forward()` to handle MDD returning `(enriched, attention)` tuple
- [ ] Store attention from MDD and pass it to R-STR alongside backbone output
- [ ] Run Task 3 tests, verify they PASS
- [ ] Run ALL existing tests, verify no regressions

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
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestTrackNetSigmoidGuard -v
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py -v
```

---

## Task 5 — Update DummyMDD test to return `(enriched, attention)` tuple

**Files:** `tests/test_models.py`

**Steps:**

- [ ] Update `TestTrackNetCustomBackbone.test_custom_mdd_module` to use a DummyMDD that returns `(enriched, attention)` tuple matching the real MDD interface
- [ ] Add a forward-pass test proving the tuple is unpacked correctly and attention reaches R-STR
- [ ] Run tests, verify they PASS

### Test code (replace `test_custom_mdd_module` in `TestTrackNetCustomBackbone`)

```python
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

        class DummyMDD(torch.nn.Module):
            def forward(self, x):
                # MDD returns (enriched_input, attention_maps)
                enriched = x + 1.0
                attention = torch.ones(x.shape[0], 2, x.shape[2], x.shape[3])
                return enriched, attention

        model = TrackNet(mdd=DummyMDD())
        assert model.mdd is not None
        assert isinstance(model.mdd, torch.nn.Module)

    def test_mdd_tuple_unpacked_in_forward(self):
        """When MDD returns (enriched, attention), forward unpacks correctly."""

        class DummyMDD(torch.nn.Module):
            def forward(self, x):
                enriched = x
                attention = torch.ones(x.shape[0], 2, x.shape[2], x.shape[3])
                return enriched, attention

        # V2 backbone (sigmoid on) with MDD but no R-STR — should work fine
        model = TrackNet(mdd=DummyMDD())
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)

    def test_mdd_attention_passed_to_rstr(self):
        """When MDD returns (enriched, attention), attention is forwarded to R-STR."""
        captured = {}

        class DummyMDD(torch.nn.Module):
            def forward(self, x):
                enriched = x
                attention = torch.ones(x.shape[0], 2, x.shape[2], x.shape[3])
                return enriched, attention

        class DummyRSTR(torch.nn.Module):
            def forward(self, logits, attention):
                captured["attention"] = attention
                # Just apply sigmoid for test simplicity
                return torch.sigmoid(logits)

        backbone = UNetBackbone(in_channels=9, num_classes=3, apply_sigmoid=False)
        model = TrackNet(backbone=backbone, mdd=DummyMDD(), rstr=DummyRSTR())
        x = torch.randn(1, 9, 288, 512)
        out = model(x)
        assert out.shape == (1, 3, 288, 512)
        assert "attention" in captured
        assert captured["attention"].shape == (1, 2, 288, 512)
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestTrackNetCustomBackbone -v
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py -v
```

---

## Task 6 — Full regression pass

**Files:** None (verification only)

**Steps:**

- [ ] Run the complete test suite across all test files
- [ ] Verify zero failures, zero errors
- [ ] Commit with message: `feat(backbone): add apply_sigmoid flag for V5 R-STR raw logit passthrough`

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/ -v
```

---

## Interface Summary

**What this change provides:**

| Consumer | Interface | Description |
|----------|-----------|-------------|
| V5 R-STR head | `UNetBackbone(apply_sigmoid=False)` | Raw logits output for residual refinement |
| V5 TrackNet | `TrackNet(backbone, mdd, rstr)` | Guard prevents double-sigmoid; MDD attention forwarded to R-STR |
| V2 (unchanged) | `UNetBackbone()` / `TrackNet()` | Default `apply_sigmoid=True` preserves all V2 behavior |

**What this change does NOT touch:**

| Unchanged | Reason |
|-----------|--------|
| `WBCEFocalLoss` | Loss still expects sigmoid probabilities — V2 feeds sigmoid output, V5 feeds R-STR sigmoid output |
| Encoder/decoder blocks | `ConvBlock`, `DownBlock`, `UpBlock`, `Bottleneck` are unmodified |
| Inference pipeline | Inference consumes `TrackNet.forward()` output — no change needed |
| Training loop | Training calls `model(x)` — no change needed |

**Design rationale (from spec Section 5):**

R-STR predicts a residual correction `delta` on draft logits, then applies sigmoid once: `H_final = sigmoid(logits + delta)`. If the backbone already applies sigmoid, the R-STR would receive values in [0, 1] and the final `sigmoid(sigmoid(logits) + delta)` would double-compress the output, destroying gradient signal and capping effective range. The `apply_sigmoid=False` flag keeps logits in unbounded space through the R-STR pipeline.
