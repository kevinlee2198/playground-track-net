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
