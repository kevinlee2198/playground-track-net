# playground-track-net

PyTorch implementation of TrackNet for real-time ball detection and tracking in racquet sports (tennis, badminton, pickleball).

## Overview

TrackNet is a heatmap-based deep learning network that detects small, fast-moving balls in sports video by leveraging temporal context from consecutive frames. This repo implements the V2 architecture (U-Net encoder-decoder) end-to-end and adds the V5 enhancements (Motion Direction Decoupling + factorized spatio-temporal Transformer head) as opt-in modules behind a single `TrackNet(backbone, mdd, rstr)` wrapper.

**Why TrackNet over YOLO/DETR?** Single-frame detectors struggle with sub-10px, motion-blurred balls. TrackNet's multi-frame input achieves 97.5% tracking accuracy vs 53.5% for YOLOv7 on shuttlecock tracking.

**Status (2026-11-15):** the V2 pipeline runs end to end on the real TrackNet tennis dataset: `check-data`, `train`, `evaluate` and `infer` have all run on it, and one full training epoch of the default recipe trained and validated cleanly. No full training run exists yet, and nothing has been evaluated on the test split. See [Known Issues](#known-issues) and §4.4–§5 of [docs/2026-11-15-pre-training-review.md](docs/2026-11-15-pre-training-review.md) for the measured numbers and open decisions.

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
    losses.py          # WBCEFocalLoss (float32, from logits)
    mdd.py             # MotionDirectionDecoupling (V5)
    rstr.py            # FactorizedAttentionLayer, TSATTHead, RSTRHead (V5)
    tracknet.py        # TrackNet wrapper + tracknet_v5() factory
    checkpoint.py      # load_model_weights (trainer checkpoint or bare state dict)
  data/
    labels.py          # Label loader: our CSV layout and the tennis Label.csv
    sources.py         # Tennis clip discovery, split by game, build_dataset
    dataset.py         # TrackNetDataset (sliding windows over one clip's frames)
    heatmap.py         # Binary-disc heatmaps; source <-> model coordinate mapping
    transforms.py      # HorizontalFlip, FrameColorJitter, Mixup, Compose
  training/
    trainer.py         # Trainer (AMP bf16, optional torch.compile, TensorBoard, resumable checkpoints)
    evaluate.py        # Detection metrics, evaluate_epoch
    run.py             # Builds the splits; train, evaluate and check-data entry points
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
    test_models.py      # Backbone, loss, V2 wrapper, V5 wiring (81 tests)
    test_mdd.py         # MDD module unit tests (21 tests)
    test_data.py        # Labels, clip discovery, dataset, heatmap, transforms (64 tests)
    test_training.py    # Trainer loop, metrics, checkpointing, resume (52 tests)
    test_inference.py   # Postprocess, tracker, video preprocess (28 tests)
    test_pipeline.py    # End to end: train, evaluate, check-data, infer (25 tests)
  docs/
    data-sources.md     # Dataset links, licences, verification log
    2026-11-15-pre-training-review.md        # Data, literature and code-readiness review
    2026-04-14-architecture-review-synthesis.md  # Full-product (iOS app) architecture review
    superpowers/specs/  # Design specifications (ball tracker, on-device iOS slice)
    superpowers/plans/  # Implementation plans (V2 model/data/training/inference, V5 sigmoid/MDD/R-STR; 2026-11-15-pre-training-fixes.md)
  main.py              # CLI: train | evaluate | check-data | infer
```

## Implementation Status

| Subsystem | Status | Tests |
|-----------|--------|-------|
| Model V2 (UNet backbone, WBCE loss, wrapper) | Done | 81 in `test_models.py` |
| Model V5 (MDD module) | Done | 21 in `test_mdd.py` |
| Model V5 (R-STR head + `tracknet_v5()` factory) | Done | covered in `test_models.py` |
| Data pipeline (label loader, tennis splits, dataset, heatmaps, augmentations) | Done; run on the real tennis dataset | 64 in `test_data.py` |
| Training (trainer, AMP, evaluation, resumable checkpoints) | Done; one real epoch run | 48 in `test_training.py` |
| Inference (postprocess, Kalman tracker, video I/O) | Done; run on one real clip | 28 in `test_inference.py` |
| CLI (`train` / `evaluate` / `check-data` / `infer` subcommands) | Done | 24 in `test_pipeline.py` |
| Full training run + published weights | Not started; see [Known Issues](#known-issues) | — |

**Total: 271 tests (65–80 s, fully synthetic).** "Done" above means implemented and tested. The tests use synthetic frames only, including a miniature tennis dataset with 1280x720 frames and the real `Label.csv` header; `test_pipeline.py` trains, evaluates and runs inference with the real TrackNet at 128x72. They pass on CPU; on a machine with CUDA the Trainer picks the GPU, so the Trainer and pipeline tests run there.

## Known Issues

Details, file references and the measured preflight numbers are in [docs/2026-11-15-pre-training-review.md](docs/2026-11-15-pre-training-review.md) §4–§5.

**Not yet shown:** no full 30-epoch run exists, and nothing has been evaluated on the test split. One epoch reached F1 0.740 on the validation split at 4 px, 512x288 (0.554 at source resolution); that is a check that the model learns, not a result.

**Open before the full run:**

- **Where it runs.** 30 epochs would take about 6.1 hours on the RTX 3070 Laptop (8 GB) without `torch.compile`, extrapolated from one measured epoch.
- **Batch size and learning rate.** Batch 2 (the default) peaks at 1,769 MiB and ran at 8.6–8.7 steps per second through the full epoch. Batch 8 ran out of GPU memory as validation started; batch 4 was not tried. Reference codebases use Adam at 1e-3 with batch 8–10; we use AdamW at 1e-4 with weight decay 0.01 and batch 2. `optimizer` (`AdamW` or `Adam`), `learning_rate` and `weight_decay` are set in the config. Under `Adam`, `weight_decay` is an L2 penalty and PyTorch's default is 0, so set it to 0 when reproducing the reference recipe.
- **`torch.compile`** needs a C compiler, which this machine lacks, so `configs/default.yaml` sets `compile_model: false` and the speed-up is unmeasured. With it set to true on a CUDA machine without a compiler, `train` stops at startup and says how to fix it.
- **Mixup** is implemented but not wired in. **Median background frame** input is the planned first experiment after a baseline. **Colour jitter strength** (0.3 brightness, contrast and saturation, always applied) is stronger than WASB's (0.1 at p = 0.5) and untuned.

**Limitations left in place:**

- Up to two frames at the end of each clip are neither trained on with full context nor scored, because windows are not padded (9 frames in val, 35 in test). This matches WASB's evaluation.
- The 4 px comparison uses `<=`; WASB uses `<`. This only matters at a distance of exactly 4.0.
- At the source-resolution tolerance (1.6 px at 512x288), ground-truth quantisation of up to 0.29 px is a meaningful share of the budget. If results sit near the boundary, switch that protocol to label coordinates.
- Mixed resolutions inside one clip are not supported: the source size is read from the first frame. `evaluate` also stops with an error if the evaluated split's clips differ in source size, or if resizing them to `input_size` changes the aspect ratio, because no single source-pixel tolerance would apply.
- `infer` and `evaluate` assume the default V2 architecture. Checkpoints record their config, but neither command reads it yet; they must before the 12-channel background-frame experiment.
- Resume is per epoch: an interrupted epoch restarts from its beginning.
- Training uses game1–6 and validates on game7; WASB trains on game1–7. Test games (game8–10) match WASB's.

**V5 only:** the MDD module does not train from its zero initialisation.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the full test suite
uv run pytest tests/ -q -n 4

# Format and lint
uv run ruff format .
uv run ruff check --fix .

# Type check
uv run ty check
```

### Train and evaluate on the tennis dataset

`configs/default.yaml` expects the TrackNet tennis dataset at `datasets/tennis/<game>/<clip>/` and splits it by game: train game1–6, validate on game7, test on game8–10. The zip's top-level folder is `Dataset/`, so it must be extracted and renamed, not unpacked straight into `datasets/tennis`; the commands are under "Set up for training" in [docs/data-sources.md](docs/data-sources.md).

```bash
# Log clip, frame and visibility counts per split and draw labels on sample frames.
# Open a few overlays: the ball should sit inside the circle in each enlarged crop.
uv run python main.py check-data --config configs/default.yaml --out /tmp/overlays --count 12

# Train. Writes checkpoints/<experiment_name>/latest.pt every epoch, best.pt (below), and TensorBoard logs in runs/<experiment_name>/.
# --max-steps N stops once N optimizer steps have been taken in total, counting steps restored by --resume
# (the epoch it stops in still validates and checkpoints).
uv run python main.py train --config configs/default.yaml

# Continue a run from its last checkpoint. With --max-steps, N must exceed the checkpoint's step.
uv run python main.py train --config configs/default.yaml \
  --resume checkpoints/tennis_v2_baseline/latest.pt

# Evaluate a checkpoint; prints JSON with both protocols. --split defaults to test,
# so pass --split val until the model is final.
uv run python main.py evaluate --config configs/default.yaml \
  --weights checkpoints/tennis_v2_baseline/latest.pt --split val
```

- **Log lines** carry a timestamp; step lines add steps per second and, on a GPU, peak GPU memory; epoch lines add validation F1, precision and recall and separate train and validation seconds.
- **`train` will not overwrite a run.** If `checkpoints/<experiment_name>/latest.pt` exists, a fresh `train` stops before doing anything: pass `--resume` or set a new `experiment_name`.
- **Resume is per epoch.** Training continues from the epoch after the checkpoint's. A checkpoint whose epoch was cut short by `--max-steps` logs a warning and the rest of that epoch is skipped. A config that differs from the one stored in the checkpoint (apart from `epochs` and `max_steps`) logs a warning listing the differences.
- **The config must hold only plain YAML types** (mappings, lists, strings, numbers, booleans, null). Checkpoints store it and are loaded with `weights_only=True`; the Trainer rejects anything else at startup.
- **Validation during training** uses 4 px at 512x288 only. `best.pt` is written only when an epoch's validation `f1` beats the best so far, which starts at 0, so a short run whose `f1` stays 0 has no `best.pt`; use `latest.pt` then. **`evaluate`** reports two protocols, each with both recall conventions:
  - `model_res_4px`: 4 px at 512x288 (TrackNetV3/V4 and RacketVision).
  - `source_res_4px`: 4 px at source resolution (WASB), which is 1.6 px at 512x288 for 1280-wide video.
  - `recall` counts a detection too far from the ball against precision only (TrackNetV3, V4, WASB); `recall_strict` also counts it as a miss (RacketVision).
- **Label coordinates** are mapped to 512x288 through pixel centres, `x_model = (x_src + 0.5) * 512 / width - 0.5` (and the same for y), the way `cv2.resize` maps pixels; inference maps back with the inverse. This is a deliberate departure from the published code (TrackNetV3 scales by plain `x * scale`): with `cv2.resize`, plain scaling leaves the target 0.3 px off the ball at 1280→512, and horizontal flip reverses that offset on half the samples. The pixel-centre mapping avoids both.
- **Tennis visibility 3** rows carry coordinates in the real files but are treated as ball-absent, as in WASB's config.

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

`--model` takes a trainer checkpoint (`latest.pt` or `best.pt`) or a bare state dict for the default V2 `TrackNet()`. Frames are converted to RGB, as in training.

## Dependencies

**Runtime:** torch, torchvision, numpy, opencv-python, scipy, pandas, pyyaml, tensorboard, tqdm

**Dev:** pytest, pytest-cov, pytest-xdist, ruff, ty

No `filterpy` — Kalman filter is a small custom NumPy implementation in [inference/tracker.py](inference/tracker.py) since filterpy has been unmaintained since 2018.

## Datasets

See [docs/data-sources.md](docs/data-sources.md) for download links, formats, licences and the verification log (last checked 2026-11-15):

- **Tennis:** TrackNet dataset (19,835 frames in 95 clips across 10 games, 1280x720; NYCU SharePoint or Kaggle)
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

1. **Phase 1 — V2 components:** model, data, training, inference pipelines and CLI. **Implemented and run end to end on the real tennis dataset** (one epoch).
2. **Phase 2 — V5 architecture:** MDD module + R-STR head wired through `tracknet_v5()`. **Implemented, training deferred** (see the note under V5 above).
3. **Phase 3 — First real training run (next):** settle where it runs and the batch size (see [Known Issues](#known-issues)), train V2 on tennis for 30 epochs, and report metrics under both protocols. First experiment after the baseline: a median background frame as extra input.
4. **Phase 4 — Multi-sport and pickleball:** add badminton and multi-sport pre-training; collect and label our own pickleball footage.
5. **Future:** On-device (iOS) deployment, court detection, shot classification, bounce visualisation, automated scoring. See the specs under `docs/`.
