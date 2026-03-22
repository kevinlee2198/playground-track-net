# playground-track-net

PyTorch implementation of TrackNet for real-time ball detection and tracking in racquet sports (tennis, badminton, pickleball).

## Overview

TrackNet is a heatmap-based deep learning network that detects small, fast-moving balls in sports video by leveraging temporal context from consecutive frames. This implementation follows the V2 architecture (U-Net encoder-decoder) with a path to V5 enhancements (Motion Direction Decoupling + Transformer refinement).

**Why TrackNet over YOLO/DETR?** Single-frame detectors struggle with sub-10px, motion-blurred balls. TrackNet's multi-frame input achieves 97.5% tracking accuracy vs 53.5% for YOLOv7 on shuttlecock tracking.

## Architecture

```
3 RGB frames (512x288) -> concat (9ch) -> U-Net Backbone -> 3 heatmaps -> post-processing -> ball (x,y)
```

- **Encoder:** 3 downsampling blocks (64 -> 128 -> 256 channels) + bottleneck (512)
- **Decoder:** 3 upsampling blocks with skip connections (256 -> 128 -> 64)
- **Output:** 3 sigmoid heatmaps (one per input frame)
- **Normalization:** GroupNorm (not BatchNorm -- stable at batch size 2)
- **Loss:** Weighted BCE with focal-style dynamic weights

## Project Structure

```
playground-track-net/
  models/
    backbone.py       # U-Net encoder-decoder (ConvBlock, DownBlock, UpBlock, Bottleneck, UNetBackbone)
    tracknet.py       # TrackNet wrapper with pluggable MDD/R-STR slots for V5
    losses.py         # WBCEFocalLoss
  data/               # Dataset loading, heatmap generation, augmentations (planned)
  training/           # Training loop, evaluation metrics (planned)
  inference/          # Post-processing, trajectory rectification, Kalman tracker (planned)
  configs/            # YAML training configs (planned)
  tests/
    test_models.py    # 32 tests covering all model components
  docs/
    superpowers/
      specs/          # Design specification
      plans/          # Implementation plans (model, data, training, inference)
```

## Implementation Status

| Subsystem | Status | Tests |
|-----------|--------|-------|
| Model (backbone, loss, wrapper) | Done | 32 passing |
| Data pipeline (dataset, heatmaps, augmentations) | Planned | - |
| Training (trainer, evaluation, configs) | Planned | - |
| Inference (post-processing, tracking, CLI) | Planned | - |

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Use the model
uv run python -c "
from models import TrackNet
import torch

model = TrackNet()
frames = torch.randn(1, 9, 288, 512)  # 3 RGB frames concatenated
heatmaps = model(frames)               # (1, 3, 288, 512) sigmoid heatmaps
print(f'Output shape: {heatmaps.shape}')
"
```

## Dependencies

**Runtime:** torch, torchvision, numpy, opencv-python, scipy, pandas, pyyaml, tensorboard, tqdm

**Dev:** pytest, pytest-cov, pytest-xdist, ruff, ty

## Papers

- [TrackNet V1 (2019)](https://arxiv.org/abs/1907.03698) -- Original heatmap-based ball tracking
- [TrackNet V5 (2025)](https://arxiv.org/abs/2512.02789) -- Motion Direction Decoupling + Transformer refinement

## Roadmap

1. **Phase 1 (current):** V2 baseline -- model, data, training, inference pipelines
2. **Phase 2:** V5 enhancements -- MDD module + R-STR Transformer head
3. **Phase 3:** Multi-sport -- fine-tuned weights for tennis, badminton, pickleball
4. **Future:** Court detection, shot classification, line calling, automated scoring
