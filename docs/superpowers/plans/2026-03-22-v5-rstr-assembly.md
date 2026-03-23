# V5 R-STR Head and Assembly

**Feature Name:** TSATTHead (factorized Transformer), RSTRHead (residual refinement wrapper), and `tracknet_v5()` factory
**Goal:** Implement the R-STR refinement head that sits after the U-Net decoder, predicting a residual correction on draft heatmap logits using factorized spatio-temporal attention, and wire it into a complete V5 model assembly.
**Architecture:** RSTRHead fuses raw logits with MDD attention via Conv1x1, applies stochastic masking during training, feeds draft heatmaps through TSATTHead to predict a residual delta, and returns `sigmoid(draft + delta)`. TSATTHead is a factorized Transformer (TimeSformer-style) operating on 16x16 non-overlapping patches with 2 layers of temporal-then-spatial self-attention.
**Tech Stack:** Python 3.12+, PyTorch 2.10+, pytest
**Prerequisites:** Sigmoid refactor (`apply_sigmoid=False` on UNetBackbone) and MDD module must be merged before this plan begins. The backbone must be able to return raw logits (no sigmoid) so R-STR receives unbounded values.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement each task below via TDD -- write the failing test first, implement code to pass, then verify.

---

## Known Limitation: Patch Size and Output Decoding

The V5 paper describes PixelShuffle(4) for output decoding, which implies 4x4 patches producing a 72x128 spatial grid (72*128 = 9,216 patches per frame, 27,648 tokens total). We use 16x16 patches with direct unpatchification instead because:

- **4x4 patches produce 27K tokens** -- infeasible for full self-attention on 8GB VRAM (quadratic memory)
- **Paper likely uses windowed attention** (e.g., Swin-style), but implementation details are unavailable without source code
- **Direct unpatchify is standard** (MAE, ViT) and produces correct output dimensions (576 patches -> 18x32 grid -> reshape to 288x512)
- If windowed attention details become available, the patch size can be reduced and PixelShuffle added without changing the RSTRHead interface

This deviation is documented in code comments on the TSATTHead class.

---

## File Map

| File | Contents |
|------|----------|
| `models/rstr.py` | `TSATTHead`, `FactorizedAttentionLayer`, `RSTRHead` |
| `models/tracknet.py` | `tracknet_v5()` factory function (added to existing file) |
| `models/__init__.py` | Updated exports: add `TSATTHead`, `RSTRHead`, `tracknet_v5` |
| `tests/test_models.py` | All R-STR and V5 assembly unit tests (appended to existing file) |

---

## Task 1 -- FactorizedAttentionLayer

**Files:** `models/rstr.py`, `tests/test_models.py`

**Steps:**

- [ ] Create `models/rstr.py` with `FactorizedAttentionLayer` class
- [ ] Write `TestFactorizedAttentionLayer` tests in `tests/test_models.py`
- [ ] Run tests, verify they pass

### Architecture

A single layer of TimeSformer-style factorized attention:
1. **Temporal self-attention:** Reshape tokens to `(B*S, T, D)`, apply `nn.MultiheadAttention`, reshape back. LayerNorm pre-norm.
2. **Spatial self-attention:** Reshape tokens to `(B*T, S, D)`, apply `nn.MultiheadAttention`, reshape back. LayerNorm pre-norm.
3. **Feed-forward:** LayerNorm -> Linear(D, ff_dim) -> GELU -> Linear(ff_dim, D). Residual connection around each sub-block.

Where `T=3` (temporal frames), `S=576` (spatial patches), `D=128` (embed_dim).

### Test code (append to `tests/test_models.py`)

```python
from models.rstr import FactorizedAttentionLayer


class TestFactorizedAttentionLayer:
    def test_output_shape(self):
        """Output shape matches input: (B, T*S, D)."""
        layer = FactorizedAttentionLayer(
            embed_dim=128, num_heads=4, ff_dim=256,
            num_frames=3, num_patches=576,
        )
        x = torch.randn(2, 1728, 128)  # B=2, T*S=3*576=1728, D=128
        out = layer(x)
        assert out.shape == (2, 1728, 128)

    def test_gradient_flows(self):
        """Gradients should flow through both temporal and spatial attention."""
        layer = FactorizedAttentionLayer(
            embed_dim=128, num_heads=4, ff_dim=256,
            num_frames=3, num_patches=576,
        )
        x = torch.randn(2, 1728, 128, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_small_input(self):
        """Verify with minimal dimensions for fast testing."""
        layer = FactorizedAttentionLayer(
            embed_dim=32, num_heads=4, ff_dim=64,
            num_frames=3, num_patches=4,
        )
        x = torch.randn(1, 12, 32)  # T*S=3*4=12
        out = layer(x)
        assert out.shape == (1, 12, 32)

    def test_residual_connection(self):
        """Output should not be identical to input but should be correlated
        (residual connections keep them close at init)."""
        layer = FactorizedAttentionLayer(
            embed_dim=32, num_heads=4, ff_dim=64,
            num_frames=3, num_patches=4,
        )
        x = torch.randn(1, 12, 32)
        out = layer(x)
        # They should differ (attention modifies the signal)
        assert not torch.allclose(x, out, atol=1e-6)
```

### Implementation code (`models/rstr.py`)

```python
import torch
import torch.nn as nn


class FactorizedAttentionLayer(nn.Module):
    """Single layer of TimeSformer-style factorized attention.

    Applies temporal self-attention across frames first, then spatial
    self-attention within each frame. Each sub-block uses pre-norm
    (LayerNorm before attention) and residual connections.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward hidden dimension.
        num_frames: Number of temporal frames (T).
        num_patches: Number of spatial patches per frame (S).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_frames: int,
        num_patches: int,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.num_patches = num_patches

        # Temporal attention
        self.norm_t = nn.LayerNorm(embed_dim)
        self.attn_t = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )

        # Spatial attention
        self.norm_s = nn.LayerNorm(embed_dim)
        self.attn_s = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )

        # Feed-forward
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape  # N = T * S
        T, S = self.num_frames, self.num_patches

        # --- Temporal self-attention: group by spatial position ---
        # Reshape: (B, T, S, D) -> (B*S, T, D)
        xt = x.view(B, T, S, D).permute(0, 2, 1, 3).reshape(B * S, T, D)
        xt_norm = self.norm_t(xt)
        xt_attn, _ = self.attn_t(xt_norm, xt_norm, xt_norm)
        xt = xt + xt_attn  # residual
        # Reshape back: (B*S, T, D) -> (B, T*S, D)
        x = xt.reshape(B, S, T, D).permute(0, 2, 1, 3).reshape(B, N, D)

        # --- Spatial self-attention: group by frame ---
        # Reshape: (B, T, S, D) -> (B*T, S, D)
        xs = x.view(B, T, S, D).reshape(B * T, S, D)
        xs_norm = self.norm_s(xs)
        xs_attn, _ = self.attn_s(xs_norm, xs_norm, xs_norm)
        xs = xs + xs_attn  # residual
        # Reshape back: (B*T, S, D) -> (B, T*S, D)
        x = xs.reshape(B, T, S, D).reshape(B, N, D)

        # --- Feed-forward with residual ---
        x = x + self.ff(self.norm_ff(x))

        return x
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestFactorizedAttentionLayer -v
```

---

## Task 2 -- TSATTHead

**Files:** `models/rstr.py`, `tests/test_models.py`

**Steps:**

- [ ] Write `TestTSATTHead` tests
- [ ] Implement `TSATTHead` in `models/rstr.py`
- [ ] Run tests, verify they pass

### Architecture

1. **Input:** `(B, 3, 288, 512)` draft heatmap logits (3 channels = 3 frames)
2. **Split:** Separate into 3 single-channel frames, each `(B, 1, 288, 512)`
3. **Patchify:** 16x16 non-overlapping patches -> `(B, 576, 256)` per frame (576 = 18*32 spatial patches, 256 = 16*16 patch_dim)
4. **Linear projection:** `(B, 576, 256)` -> `(B, 576, 128)` per frame
5. **Stack frames + add positional encodings:**
   - Stack 3 frames: `(B, 1728, 128)` where 1728 = 3 * 576
   - Add learnable spatial positional encoding: `(1, 576, 128)` broadcast across frames
   - Add learnable temporal positional encoding: `(1, 3, 128)` broadcast across patches
6. **Transformer:** 2 layers of `FactorizedAttentionLayer`
7. **Output projection:** Linear `(128 -> 256)` per token, reshape to `(B, 3, 18, 32, 16, 16)`, unpatchify to `(B, 3, 288, 512)`
8. **Zero-init:** Output projection weights and bias initialized to zero so initial residuals are near-zero

### Test code (append to `tests/test_models.py`)

```python
from models.rstr import TSATTHead


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
        assert 100_000 < total < 2_000_000, (
            f"Unexpected param count: {total:,}"
        )

    def test_batch_size_one(self):
        """Should work with batch size 1."""
        head = TSATTHead()
        x = torch.randn(1, 3, 288, 512)
        out = head(x)
        assert out.shape == (1, 3, 288, 512)

    def test_deterministic_in_mode(self):
        """In inference mode, same input should produce same output."""
        head = TSATTHead()
        head.train(False)
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
```

### Implementation code (add to `models/rstr.py`)

```python
class TSATTHead(nn.Module):
    """Factorized spatio-temporal attention Transformer for residual refinement.

    Takes draft heatmap logits (B, 3, H, W), patchifies into 16x16 tokens,
    applies factorized temporal-then-spatial self-attention, and produces
    a residual correction delta of the same shape.

    KNOWN LIMITATION: The V5 paper describes PixelShuffle(4) for output
    decoding, implying 4x4 patches (72x128 grid = 27K tokens per frame).
    We use 16x16 patches with direct unpatchification instead because:
    - 4x4 patches produce 27K tokens -- infeasible for full self-attention
      on 8GB VRAM (quadratic memory scaling)
    - Paper likely uses windowed attention (details unavailable)
    - Direct unpatchify is standard (MAE, ViT) and produces correct dims

    Args:
        patch_size: Side length of non-overlapping patches.
        embed_dim: Transformer token embedding dimension.
        num_heads: Number of attention heads per layer.
        num_layers: Number of FactorizedAttentionLayer layers.
        ff_dim: Feed-forward hidden dimension.
        num_frames: Number of input frames (channels).
        img_h: Input image height (must be divisible by patch_size).
        img_w: Input image width (must be divisible by patch_size).
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        num_frames: int = 3,
        img_h: int = 288,
        img_w: int = 512,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.img_h = img_h
        self.img_w = img_w
        self.grid_h = img_h // patch_size  # 18
        self.grid_w = img_w // patch_size  # 32
        self.num_patches = self.grid_h * self.grid_w  # 576
        self.patch_dim = patch_size * patch_size  # 256

        # Patch embedding: flatten patch pixels -> project to embed_dim
        self.patch_proj = nn.Linear(self.patch_dim, embed_dim)

        # Factorized positional encodings
        self.pos_spatial = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
        self.pos_temporal = nn.Parameter(
            torch.randn(1, num_frames, embed_dim) * 0.02
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            FactorizedAttentionLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_frames=num_frames,
                num_patches=self.num_patches,
            )
            for _ in range(num_layers)
        ])

        # Output projection: embed_dim -> patch_dim (for unpatchify)
        self.output_proj = nn.Linear(embed_dim, self.patch_dim)
        self._zero_init_output()

    def _zero_init_output(self) -> None:
        """Zero-init output projection so initial residuals are near-zero."""
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _patchify(self, frame: torch.Tensor) -> torch.Tensor:
        """Convert a single-channel frame to patch tokens.

        Args:
            frame: (B, 1, H, W)

        Returns:
            patches: (B, num_patches, patch_dim)
        """
        B = frame.shape[0]
        p = self.patch_size
        # (B, 1, grid_h, p, grid_w, p) -> (B, grid_h, grid_w, p*p)
        patches = frame.reshape(B, 1, self.grid_h, p, self.grid_w, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(
            B, self.num_patches, self.patch_dim
        )
        return patches

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert patch tokens back to spatial frame.

        Args:
            tokens: (B, num_patches, patch_dim)

        Returns:
            frame: (B, 1, H, W)
        """
        B = tokens.shape[0]
        p = self.patch_size
        # (B, num_patches, patch_dim) -> (B, grid_h, grid_w, p, p)
        tokens = tokens.reshape(B, self.grid_h, self.grid_w, p, p)
        # -> (B, grid_h, p, grid_w, p) -> (B, 1, H, W)
        frame = tokens.permute(0, 1, 3, 2, 4).reshape(
            B, 1, self.img_h, self.img_w
        )
        return frame

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict residual delta from draft heatmap logits.

        Args:
            x: (B, 3, H, W) draft heatmap logits.

        Returns:
            delta: (B, 3, H, W) residual correction.
        """
        B = x.shape[0]
        T = self.num_frames

        # 1. Split channels into individual frames and patchify each
        frames = x.unsqueeze(2)  # (B, 3, 1, H, W)
        patches_list = []
        for t in range(T):
            frame_t = frames[:, t]  # (B, 1, H, W)
            patches_list.append(self._patchify(frame_t))  # (B, S, patch_dim)

        # 2. Project to embedding dim
        # Stack: (B, T, S, patch_dim)
        all_patches = torch.stack(patches_list, dim=1)
        tokens = self.patch_proj(all_patches)  # (B, T, S, D)

        # 3. Add factorized positional encodings
        # Spatial: broadcast across T -> (1, 1, S, D)
        tokens = tokens + self.pos_spatial.unsqueeze(1)
        # Temporal: broadcast across S -> (1, T, 1, D)
        tokens = tokens + self.pos_temporal.unsqueeze(2)

        # 4. Flatten to sequence: (B, T*S, D)
        tokens = tokens.reshape(B, T * self.num_patches, -1)

        # 5. Transformer layers
        for layer in self.layers:
            tokens = layer(tokens)

        # 6. Output projection and unpatchify
        tokens = self.output_proj(tokens)  # (B, T*S, patch_dim)
        tokens = tokens.reshape(B, T, self.num_patches, self.patch_dim)

        # Unpatchify each frame
        delta_frames = []
        for t in range(T):
            frame_tokens = tokens[:, t]  # (B, S, patch_dim)
            delta_frames.append(self._unpatchify(frame_tokens))  # (B, 1, H, W)

        delta = torch.cat(delta_frames, dim=1)  # (B, 3, H, W)
        return delta
```

**Implementation note:** The `_patchify` and `_unpatchify` methods must be exact inverses. The round-trip test in `TestTSATTHead.test_patchify_unpatchify_roundtrip` verifies this. The key permutation for unpatchify is `(B, grid_h, grid_w, p, p) -> permute(0, 1, 3, 2, 4) -> (B, grid_h*p, grid_w*p)` which interleaves the patch rows and grid rows correctly.

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestTSATTHead -v
```

---

## Task 3 -- RSTRHead

**Files:** `models/rstr.py`, `tests/test_models.py`

**Steps:**

- [ ] Write `TestRSTRHead` tests
- [ ] Implement `RSTRHead` in `models/rstr.py`
- [ ] Run tests, verify they pass

### Architecture

```
Input: logits (B, 3, H, W)  +  mdd_attention (B, 4, H, W)
  1. Concat -> (B, 7, H, W) -> Conv1x1 -> (B, 3, H, W) = draft_mdd
  2. draft_masked = Dropout(draft_mdd, p=0.1)          [training only]
  3. delta = TSATTHead(draft_masked)                     (B, 3, H, W)
  4. return sigmoid(draft_mdd + delta)
```

**Critical detail:** The residual addition uses `draft_mdd` (pre-dropout), NOT `draft_masked`. Dropout is applied only to the TSATTHead input to prevent the Transformer from relying too heavily on the draft. The final sigmoid is the only sigmoid in the entire pipeline.

**Validation:** RSTRHead requires `attention is not None`. If called with `attention=None`, it must raise `ValueError` because MDD attention maps are architecturally required for the fusion step.

### Test code (append to `tests/test_models.py`)

```python
from models.rstr import RSTRHead


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
        head.train(False)
        logits = torch.randn(1, 3, 288, 512)
        attention = torch.randn(1, 4, 288, 512)
        out = head(logits, attention)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_dropout_active_in_train(self):
        """In training mode, stochastic masking should produce different
        outputs across calls (with high probability)."""
        head = RSTRHead()
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
        head.train(False)
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
        head.train(False)
        out = head(logits, attention)
        assert torch.allclose(out, expected, atol=0.05), (
            "At init, RSTRHead output should approximate sigmoid(draft_mdd)"
        )
```

### Implementation code (add to `models/rstr.py`)

```python
class RSTRHead(nn.Module):
    """Residual-Driven Spatio-Temporal Refinement head.

    Fuses raw logits with MDD attention maps, applies stochastic masking
    during training, predicts a residual delta via TSATTHead, and returns
    sigmoid(draft + delta).

    Args:
        logit_channels: Number of draft logit channels (3 for 3 frames).
        attention_channels: Number of MDD attention channels (4: 2 prev + 2 next).
        dropout_p: Dropout probability for stochastic masking (training only).
        **tsatt_kwargs: Forwarded to TSATTHead constructor.
    """

    def __init__(
        self,
        logit_channels: int = 3,
        attention_channels: int = 4,
        dropout_p: float = 0.1,
        **tsatt_kwargs,
    ) -> None:
        super().__init__()
        # Fusion: concat logits + attention -> 3-channel draft_mdd
        self.fusion_conv = nn.Conv2d(
            logit_channels + attention_channels,
            logit_channels,
            kernel_size=1,
            bias=True,
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.tsatt = TSATTHead(num_frames=logit_channels, **tsatt_kwargs)
        self.sigmoid = nn.Sigmoid()

        self._init_fusion()

    def _init_fusion(self) -> None:
        nn.init.kaiming_uniform_(self.fusion_conv.weight, nonlinearity="relu")
        if self.fusion_conv.bias is not None:
            nn.init.zeros_(self.fusion_conv.bias)

    def forward(
        self,
        logits: torch.Tensor,
        attention: torch.Tensor | None,
    ) -> torch.Tensor:
        """Refine draft logits using MDD attention and Transformer residual.

        Args:
            logits: (B, 3, H, W) raw heatmap logits from backbone (pre-sigmoid).
            attention: (B, 4, H, W) MDD attention maps. Must not be None.

        Returns:
            heatmaps: (B, 3, H, W) refined heatmaps in [0, 1].

        Raises:
            ValueError: If attention is None.
        """
        if attention is None:
            raise ValueError(
                "RSTRHead requires MDD attention maps but received "
                "attention=None. The R-STR pipeline depends on MDD "
                "attention for logit fusion."
            )

        # 1. Fuse logits + attention -> draft_mdd
        fused = torch.cat([logits, attention], dim=1)  # (B, 7, H, W)
        draft_mdd = self.fusion_conv(fused)  # (B, 3, H, W)

        # 2. Stochastic masking (training only)
        draft_masked = self.dropout(draft_mdd)  # (B, 3, H, W)

        # 3. Predict residual delta
        delta = self.tsatt(draft_masked)  # (B, 3, H, W)

        # 4. Residual addition uses pre-dropout draft_mdd
        return self.sigmoid(draft_mdd + delta)
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestRSTRHead -v
```

---

## Task 4 -- Backbone sigmoid refactor prerequisite check

**Files:** `models/backbone.py`, `models/tracknet.py`

**Steps:**

- [ ] Verify `UNetBackbone` has `apply_sigmoid` parameter (merged prerequisite)
- [ ] If not present, this task is BLOCKED -- stop and report

This task is a gate check, not an implementation task. The sigmoid refactor is a prerequisite that should already be merged. It adds an `apply_sigmoid: bool = True` parameter to `UNetBackbone.__init__()` so that the `forward()` method can return raw logits when `apply_sigmoid=False`.

**Expected state after sigmoid refactor:**

```python
class UNetBackbone(nn.Module):
    def __init__(self, in_channels=9, num_classes=3, apply_sigmoid=True):
        ...
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x):
        ...
        logits = self.head(u3)
        return self.sigmoid(logits) if self.apply_sigmoid else logits
```

The V5 assembly requires `apply_sigmoid=False` because R-STR operates on raw logits and applies sigmoid once at the very end.

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run python -c "from models.backbone import UNetBackbone; m = UNetBackbone(apply_sigmoid=False); print('OK: apply_sigmoid refactor is present')"
```

If this fails with `TypeError`, the sigmoid refactor has not been merged and Tasks 5-7 are blocked.

---

## Task 5 -- TrackNet wrapper update for V5 data flow

**Files:** `models/tracknet.py`, `tests/test_models.py`

**Steps:**

- [ ] Write tests for V5-style forward pass with MDD attention passthrough
- [ ] Update `TrackNet.forward()` to pass MDD attention to R-STR head
- [ ] Run tests, verify both V2 and V5 paths work

### Design

The V5 forward pass requires MDD attention maps to be passed through to RSTRHead. The MDD module produces both the 13-channel input and the 4-channel attention maps. TrackNet needs to handle this.

**Approach:** The MDD module returns a tuple `(enriched_input, attention_maps)`. TrackNet unpacks this when MDD is present, passes `enriched_input` to the backbone, and passes both `logits` and `attention_maps` to R-STR.

### Test code (append to `tests/test_models.py`)

```python
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
```

### Implementation changes (`models/tracknet.py`)

**DO NOT rewrite `forward()`.** Plan 1 (sigmoid refactor) already wrote the correct version with MDD tuple unpacking and attention passthrough to R-STR. Verify it works by running the tests above — they should pass against the existing `forward()` without any code changes here.

If the tests fail, the sigmoid refactor branch was not merged correctly. Fix the prerequisite, don't rewrite `forward()`.

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestTrackNetV5Flow tests/test_models.py::TestTrackNet -v
```

---

## Task 6 -- `tracknet_v5()` factory function

**Files:** `models/tracknet.py`, `tests/test_models.py`

**Steps:**

- [ ] Write `TestTrackNetV5Factory` tests
- [ ] Implement `tracknet_v5()` factory in `models/tracknet.py`
- [ ] Run tests, verify they pass

### Design

The factory creates a complete V5 model with MDD + UNetBackbone(in_channels=13, apply_sigmoid=False) + RSTRHead. It mirrors the V2 default constructor but wires all three components together.

### Test code (append to `tests/test_models.py`)

```python
from models.tracknet import tracknet_v5


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
        model.train(False)
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
```

### Implementation code (add to `models/tracknet.py`)

```python
from models.rstr import RSTRHead

# Import MDD (merged prerequisite from v5-mdd-module branch)
from models.mdd import MotionDirectionDecoupling


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
```

### Verify

```bash
cd /home/kevinlee/workspace/playground/playground-track-net && uv run pytest tests/test_models.py::TestTrackNetV5Factory -v
```

---

## Task 7 -- Public API exports and integration tests

**Files:** `models/__init__.py`, `tests/test_models.py`

**Steps:**

- [ ] Update `models/__init__.py` with new exports
- [ ] Write V5 integration tests (end-to-end forward+backward, V2 regression)
- [ ] Run full test suite, verify everything passes

### Updated exports (`models/__init__.py`)

```python
from models.backbone import UNetBackbone
from models.losses import WBCEFocalLoss
from models.mdd import MotionDirectionDecoupling
from models.rstr import FactorizedAttentionLayer, RSTRHead, TSATTHead
from models.tracknet import TrackNet, tracknet_v5

__all__ = [
    "FactorizedAttentionLayer",
    "MotionDirectionDecoupling",
    "RSTRHead",
    "TSATTHead",
    "TrackNet",
    "UNetBackbone",
    "WBCEFocalLoss",
    "tracknet_v5",
]
```

### Test code (append to `tests/test_models.py`)

```python
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
        model.train(False)
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
| Training subsystem | `model = tracknet_v5()` | Instantiate complete V5 model |
| Training subsystem | `heatmaps = model(input_tensor)` | Forward pass: `(B, 9, 288, 512)` -> `(B, 3, 288, 512)` |
| Training subsystem | `loss = WBCEFocalLoss()(pred, target)` | Scalar loss for backprop (unchanged from V2) |
| Inference subsystem | `heatmaps = model(input_tensor)` | Same forward pass interface as V2 |
| Weight transfer | `v5_backbone.load_state_dict(v2_state, strict=False)` | All layers transfer except first conv |

**What this subsystem expects from others (prerequisites):**

| Provider | Interface | Description |
|----------|-----------|-------------|
| Sigmoid refactor | `UNetBackbone(apply_sigmoid=False)` | Backbone must support returning raw logits |
| MDD module | `MDD()` returning `(enriched_input, attention_maps)` | MDD module must be merged and return a 2-tuple |
| MDD module | `enriched_input: (B, 13, H, W)` | 13-channel enriched input for backbone |
| MDD module | `attention_maps: (B, 4, H, W)` | 4-channel attention maps for R-STR fusion |
| Data subsystem | Input tensor `(B, 9, 288, 512)` | 3 RGB frames concatenated, normalized to [0, 1] |
| Data subsystem | Target heatmaps `(B, 3, 288, 512)` | Binary (or continuous with mixup) ground truth |

**Dependency order:** Sigmoid refactor -> MDD module -> this plan (R-STR + V5 assembly)
