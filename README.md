# playground-track-net

PyTorch implementation of TrackNet for real-time ball detection and tracking in racquet sports (tennis, badminton, pickleball).

## Overview

TrackNet is a heatmap-based deep learning network that detects small, fast-moving balls in sports video by leveraging temporal context from consecutive frames. This repo implements the V2 architecture (U-Net encoder-decoder) end-to-end and adds the V5 enhancements (Motion Direction Decoupling + factorized spatio-temporal Transformer head) as opt-in modules behind a single `TrackNet(backbone, mdd, rstr)` wrapper.

**Why TrackNet over YOLO/DETR?** Single-frame detectors struggle with sub-10px, motion-blurred balls. TrackNet's multi-frame input achieves 97.5% tracking accuracy vs 53.5% for YOLOv7 on shuttlecock tracking.

**Status (2026-11-15):** all components are implemented and unit-tested on synthetic data, but nothing has been trained on real data yet, and the code has known defects that block a real run. See [Known Issues](#known-issues) and [docs/2026-11-15-pre-training-review.md](docs/2026-11-15-pre-training-review.md).

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

V5 was implemented from the arXiv paper before any code was public. It differs from the authors' later SDK in several places, and the paper's results have not been independently reproduced, so V5 training is deferred. See the [pre-training review](docs/2026-11-15-pre-training-review.md) §3.2.

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
    heatmap.py         # Binary-disc heatmap generation
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
    data-sources.md     # Dataset links, licences, verification log
    2026-11-15-pre-training-review.md        # Data, literature and code-readiness review
    2026-04-14-architecture-review-synthesis.md  # Full-product (iOS app) architecture review
    superpowers/specs/  # Design specifications (ball tracker, on-device iOS slice)
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
| Real-data training runs + published weights | Blocked on [Known Issues](#known-issues) | — |

**Total: 166 tests (~50s, CPU-only, fully synthetic).** "Done" above means implemented and unit-tested. The tests use 512x288 fixtures and stub models, so they do not exercise real dataset layouts, label rescaling, or mixed-precision training.

## Known Issues

Found in the 2026-11-15 review; none are fixed yet. Details and file references are in [docs/2026-11-15-pre-training-review.md](docs/2026-11-15-pre-training-review.md) §4.

**Blocking a real training run:**

- The loss returns NaN under bfloat16 autocast once predictions are confident.
- Labels are not rescaled when frames are resized to 512x288.
- The label loader only parses our own CSV layout, not the layouts the real datasets ship.
- `infer` feeds BGR frames; training uses RGB.
- `infer` cannot load a checkpoint saved by the trainer.
- `train` and `evaluate` CLI commands are placeholders, and there is no train/val/test split logic.
- `torch.compile` (on by default in `configs/default.yaml`) needs a C compiler, which the current dev machine lacks.

**Recipe mismatches against published code:**

- Heatmap radius is 30 px; reference codebases use 2.5 px at 512x288.
- Training windows step by 3; reference codebases step by 1.
- Colour jitter is sampled per frame, not per window. Mixup is implemented but not wired in.
- Evaluation counts errors differently from the papers, so F1 would not be comparable.

**V5 only:** the MDD module does not train from its zero initialisation.

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

This command does not yet work with checkpoints produced by the trainer; see [Known Issues](#known-issues).

## Dependencies

**Runtime:** torch, torchvision, numpy, opencv-python, scipy, pandas, pyyaml, tensorboard, tqdm

**Dev:** pytest, pytest-cov, pytest-xdist, ruff, ty

No `filterpy` — Kalman filter is a small custom NumPy implementation in [inference/tracker.py](inference/tracker.py) since filterpy has been unmaintained since 2018.

## Datasets

See [docs/data-sources.md](docs/data-sources.md) for download links, formats, licences and the verification log (last checked 2026-11-15):

- **Tennis:** TrackNet dataset (~20K frames; NYCU SharePoint or Kaggle)
- **Badminton:** CoachAI shuttlecock dataset (78K frames; NYCU SharePoint, downloadable with curl)
- **Multi-sport:** RacketVision (AAAI 2026; released April 2026 on Hugging Face). 435K frames, of which 64K carry ball labels
- **Table tennis:** BlurBall (~64K frames, fixed cameras)
- **Pickleball:** AndrewDettor/TrackNet-Pickleball (12.8K labelled frames from one match; raw video not published). We need our own recordings

Most of these state no data licence, and some are explicitly non-commercial. Check before using any in a shipped model.

## Papers

- [TrackNet V1 (2019)](https://arxiv.org/abs/1907.03698) — Original heatmap-based ball tracking
- [TrackNet V5 (2025)](https://arxiv.org/abs/2512.02789) — Motion Direction Decoupling + Transformer refinement
- [WASB (BMVC 2023)](https://arxiv.org/abs/2311.05237) — HRNet-based baseline across five sports; on-device candidate
- [BlurBall (CVPRW 2026)](https://arxiv.org/abs/2509.18387) — Blur-aware labels and dataset
- [RacketVision (AAAI 2026)](https://arxiv.org/abs/2511.17045) — Multi-sport benchmark; background-frame ablation

## Roadmap

1. **Phase 1 — V2 components:** model, data, training, inference pipelines. **Implemented**, with defects listed under Known Issues.
2. **Phase 2 — V5 architecture:** MDD module + R-STR head wired through `tracknet_v5()`. **Implemented, training deferred** (see the note under V5 above).
3. **Phase 3 — First real training run (next):** fix the Known Issues, then train V2 on tennis and report metrics comparable with published ones. First experiment after the baseline: a median background frame as extra input.
4. **Phase 4 — Multi-sport and pickleball:** add badminton and multi-sport pre-training; collect and label our own pickleball footage.
5. **Future:** On-device (iOS) deployment, court detection, shot classification, bounce visualisation, automated scoring. See the specs under `docs/`.
