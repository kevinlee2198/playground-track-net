# playground-track-net

PyTorch implementation of TrackNet for real-time ball detection and tracking in racquet sports (tennis, badminton, pickleball).

## Overview

TrackNet is a heatmap-based deep learning network that detects small, fast-moving balls in sports video by leveraging temporal context from consecutive frames. This repo implements the V2 architecture (U-Net encoder-decoder) end-to-end and adds the V5 enhancements (Motion Direction Decoupling + factorized spatio-temporal Transformer head) as opt-in modules behind a single `TrackNet(backbone, mdd, rstr)` wrapper.

**Why TrackNet over YOLO/DETR?** Single-frame detectors struggle with sub-10px, motion-blurred balls. TrackNet's multi-frame input achieves 97.5% tracking accuracy vs 53.5% for YOLOv7 on shuttlecock tracking.

## Architecture

### V2 (baseline)

```
3 RGB frames (512x288) -> concat (9ch) -> U-Net Backbone -> 3 sigmoid heatmaps -> post-processing -> ball (x,y)
```

- **Encoder:** 3 downsampling blocks (64 -> 128 -> 256 channels) + bottleneck (512)
- **Decoder:** 3 upsampling blocks with skip connections (256 -> 128 -> 64)
- **Output:** 3 sigmoid heatmaps (one per input frame)
- **Normalization:** GroupNorm (not BatchNorm — stable at batch size 2)
- **Loss:** Weighted BCE with focal-style dynamic weights

### V5 (enhanced)

```
3 RGB frames -> MDD -> 13ch enriched + 4ch attention
                |                          |
                v                          v
            UNetBackbone (logits) -----> R-STR head -> 3 sigmoid heatmaps
```

- **MDD:** Motion Direction Decoupling. Computes signed inter-frame differences and applies a learnable adaptive sigmoid (only 2 trainable scalars: `alpha`, `beta`) to surface ball-motion regions. Outputs a 13-channel enriched stack and a 4-channel attention map.
- **Backbone:** Same U-Net, reconfigured to take 13 input channels and emit raw logits (`apply_sigmoid=False`).
- **R-STR head:** TimeSformer-style factorized spatio-temporal attention that refines logits using the MDD attention as a fusion signal, then applies sigmoid.
- **Factory:** [tracknet_v5()](models/tracknet.py#L52) returns a fully wired V5 model.

## Project Structure

```
playground-track-net/
  models/
    backbone.py        # U-Net encoder-decoder (ConvBlock, DownBlock, UpBlock, Bottleneck, UNetBackbone)
    losses.py          # WBCEFocalLoss
    mdd.py             # MotionDirectionDecoupling (V5)
    rstr.py            # FactorizedAttentionLayer, TSATTHead, RSTRHead (V5)
    tracknet.py        # TrackNet wrapper + tracknet_v5() factory
  data/
    dataset.py         # TrackNetDataset (sliding-window CSV loader)
    heatmap.py         # Gaussian heatmap generation
    transforms.py      # HorizontalFlip, FrameColorJitter, Mixup, Compose
  training/
    trainer.py         # Trainer (AMP bf16, torch.compile, TensorBoard, checkpointing)
    evaluate.py        # Detection metrics, evaluate_epoch
  inference/
    video_preprocess.py # Frame extraction, sliding-window batching
    postprocess.py      # Heatmap -> coordinates, trajectory rectification
    tracker.py          # KalmanBallTracker (custom NumPy Kalman, no filterpy)
  utils/
    visualization.py    # Ball overlay drawing for annotated video output
  configs/
    default.yaml        # Training config (optimizer, schedule, AMP, paths)
  tests/
    conftest.py         # Shared synthetic-data fixtures
    test_models.py      # Backbone, loss, V2 wrapper, V5 wiring (77 tests)
    test_mdd.py         # MDD module unit tests (21 tests)
    test_data.py        # Dataset, heatmap, transforms (27 tests)
    test_training.py    # Trainer loop, metrics, checkpointing (15 tests)
    test_inference.py   # Postprocess, tracker, video preprocess (26 tests)
  docs/
    data-sources.md     # Tennis/badminton/pickleball dataset links
    superpowers/specs/  # Design specification
    superpowers/plans/  # Implementation plans (V2 model/data/training/inference, V5 sigmoid/MDD/R-STR)
  main.py              # CLI: train | evaluate | infer
```

## Implementation Status

| Subsystem | Status | Tests |
|-----------|--------|-------|
| Model V2 (UNet backbone, WBCE loss, wrapper) | Done | 77 in `test_models.py` |
| Model V5 (MDD module) | Done | 21 in `test_mdd.py` |
| Model V5 (R-STR head + `tracknet_v5()` factory) | Done | covered in `test_models.py` |
| Data pipeline (dataset, heatmaps, augmentations) | Done | 27 in `test_data.py` |
| Training (trainer, AMP, evaluation, checkpoints) | Done | 15 in `test_training.py` |
| Inference (postprocess, Kalman tracker, video I/O) | Done | 26 in `test_inference.py` |
| CLI (`train` / `evaluate` / `infer` subcommands) | `infer` wired; `train`/`evaluate` stubbed | — |
| Real-data training runs + published weights | Pending | — |

**Total: 166 tests passing (~50s, CPU-only, fully synthetic).**

## Quick Start

```bash
# Install dependencies
uv sync

# Run the full test suite
uv run pytest tests/ -v

# Format and lint
uv run ruff format .
uv run ruff check --fix .

# Type check
uv run ty
```

### Use the model from Python

```python
import torch
from models import TrackNet, tracknet_v5

# V2: backbone only
v2 = TrackNet()
frames = torch.randn(1, 9, 288, 512)        # 3 RGB frames concatenated
heatmaps = v2(frames)                        # (1, 3, 288, 512) sigmoid heatmaps

# V5: MDD + UNet + R-STR
v5 = tracknet_v5()
heatmaps = v5(frames)                        # (1, 3, 288, 512) sigmoid heatmaps
```

### Run inference on a video

```bash
uv run python main.py infer \
  --video path/to/match.mp4 \
  --model path/to/weights.pt \
  --output predictions.csv \
  --output-video annotated.mp4 \
  --threshold 0.5
```

Outputs a CSV with `Frame, X, Y, Confidence, Visibility` columns and (optionally) an annotated MP4 with ball overlays drawn by [utils/visualization.py](utils/visualization.py).

## Dependencies

**Runtime:** torch, torchvision, numpy, opencv-python, scipy, pandas, pyyaml, tensorboard, tqdm

**Dev:** pytest, pytest-cov, pytest-xdist, ruff, ty

No `filterpy` — Kalman filter is a small custom NumPy implementation in [inference/tracker.py](inference/tracker.py) since filterpy has been unmaintained since 2018.

## Datasets

See [docs/data-sources.md](docs/data-sources.md) for download links and formats:

- **Tennis:** TrackNet dataset (~20K frames, Kaggle)
- **Badminton:** CoachAI Shuttlecock dataset (78K frames, SharePoint)
- **Pickleball:** AndrewDettor/TrackNet-Pickleball (~12K frames, low variety)
- **Multi-sport:** RacketVision (435K frames, AAAI 2026 — pending release)

## Papers

- [TrackNet V1 (2019)](https://arxiv.org/abs/1907.03698) — Original heatmap-based ball tracking
- [TrackNet V5 (2025)](https://arxiv.org/abs/2512.02789) — Motion Direction Decoupling + Transformer refinement

## Roadmap

1. **Phase 1 — V2 baseline:** model, data, training, inference pipelines. **Done.**
2. **Phase 2 — V5 architecture:** MDD module + R-STR head wired through `tracknet_v5()`. **Done** (model components); training runs on real data still pending.
3. **Phase 3 — Real-data training:** sport-specific weights for tennis, badminton, pickleball; published checkpoints; benchmarked detection/tracking metrics.
4. **Future:** Court detection, shot classification, line calling, automated scoring.
