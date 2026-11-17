# playground-track-net

PyTorch implementation of TrackNet for real-time ball detection and tracking in racquet sports (tennis, badminton, pickleball).

## Overview

TrackNet is a heatmap-based deep learning network that detects small, fast-moving balls in sports video by leveraging temporal context from consecutive frames. This repo implements the V2 architecture (U-Net encoder-decoder) end-to-end and adds the V5 enhancements (Motion Direction Decoupling + factorized spatio-temporal Transformer head) as opt-in modules behind a single `TrackNet(backbone, mdd, rstr)` wrapper.

**Why TrackNet over YOLO/DETR?** Single-frame detectors struggle with sub-10px, motion-blurred balls. TrackNet's multi-frame input achieves 97.5% tracking accuracy vs 53.5% for YOLOv7 on shuttlecock tracking.

**Status (2026-11-16):** the first full run is done. V2 trained for 30 epochs on the TrackNet tennis dataset reaches F1 0.949 on the validation game at 4 px, 512x288, and 0.771 at 4 px of the 1280x720 source. Most of that gap is one constant 1.1 px horizontal offset, caused by the flip augmentation meeting an offset in the dataset's labels: a second run without the flip has no offset and scores 0.903 at source resolution (0.947 at 4 px, 512x288). Nothing has been evaluated on the test split. A median background frame as a fourth input image is implemented (`background_frame: true`); its run, `tennis_v2_bg`, was started on 2026-11-16 and its results are pending. Mixup (`mixup_alpha`) and colour-jitter strength (`color_jitter`) are config switches; their runs, `tennis_v2_mixup` and `tennis_v2_jitter`, are planned and not started. Results and analysis: [docs/2026-11-16-training-results.md](docs/2026-11-16-training-results.md). Open decisions: [Known Issues](#known-issues) and §5 of [docs/2026-11-15-pre-training-review.md](docs/2026-11-15-pre-training-review.md).

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
- **Optional background frame:** with `background_frame: true` in the config, each window also carries its clip's median background (the per-pixel median of up to 50 evenly spaced frames, at model resolution) as a fourth RGB image, so the input has 12 channels. `build_model(config)` builds the matching model.

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
    checkpoint.py      # build_model (from a config), load_model (as a checkpoint was trained), load_model_weights
  data/
    labels.py          # Label loader: our CSV layout and the tennis Label.csv
    sources.py         # Tennis clip discovery, split by game, build_dataset
    dataset.py         # TrackNetDataset (sliding windows over one clip's frames)
    background.py      # A clip's median background frame
    heatmap.py         # Binary-disc heatmaps; source <-> model coordinate mapping
    transforms.py      # HorizontalFlip, FrameColorJitter, mixup_batch (training batches), Compose
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
    default.yaml        # Training config with every key (optimizer, schedule, AMP, paths)
    tennis_v2_noflip.yaml # default.yaml without the horizontal flip
    tennis_v2_bg.yaml   # tennis_v2_noflip.yaml plus the median background frame
    tennis_v2_mixup.yaml  # tennis_v2_noflip.yaml plus mixup (mixup_alpha: 0.5)
    tennis_v2_jitter.yaml # tennis_v2_noflip.yaml with WASB's colour jitter (0.1 at p 0.5)
  tests/
    conftest.py         # Shared synthetic-data fixtures
    test_models.py      # Backbone, loss, V2 wrapper, V5 wiring, model loading (88 tests)
    test_mdd.py         # MDD module unit tests (21 tests)
    test_data.py        # Labels, clip discovery, dataset, background frame, heatmap, transforms (107 tests)
    test_training.py    # Trainer loop, metrics, checkpointing, resume, configs (66 tests)
    test_inference.py   # Postprocess, tracker, video preprocess (30 tests)
    test_pipeline.py    # End to end: train, evaluate, check-data, infer (52 tests)
  docs/
    data-sources.md     # Dataset links, licences, verification log
    2026-11-15-pre-training-review.md        # Data, literature and code-readiness review
    2026-11-16-training-results.md           # Full training runs: results and error analysis
    2026-04-14-architecture-review-synthesis.md  # Full-product (iOS app) architecture review
    superpowers/specs/  # Design specifications (ball tracker, on-device iOS slice)
    superpowers/plans/  # Implementation plans (V2 model/data/training/inference, V5 sigmoid/MDD/R-STR; 2026-11-15-pre-training-fixes.md)
  main.py              # CLI: train | evaluate | check-data | infer
```

## Implementation Status

| Subsystem | Status | Tests |
|-----------|--------|-------|
| Model V2 (UNet backbone, WBCE loss, wrapper, `build_model` / `load_model`) | Done | 88 in `test_models.py` |
| Model V5 (MDD module) | Done | 21 in `test_mdd.py` |
| Model V5 (R-STR head + `tracknet_v5()` factory) | Done | covered in `test_models.py` |
| Data pipeline (label loader, tennis splits, dataset, heatmaps, augmentations) | Done; run on the real tennis dataset | 107 in `test_data.py` |
| Median background frame (`background_frame: true`) | Done; not yet trained on real data | covered in `test_data.py`, `test_models.py`, `test_inference.py`, `test_pipeline.py` |
| Mixup (`mixup_alpha`) and colour-jitter settings (`color_jitter`) | Done; not yet trained on real data | covered in `test_data.py`, `test_training.py`, `test_pipeline.py` |
| Training (trainer, AMP, evaluation, resumable checkpoints, configs with `base:`) | Done; one full 30-epoch run | 66 in `test_training.py` |
| Inference (postprocess, Kalman tracker, video I/O) | Done; run on one real clip | 30 in `test_inference.py` |
| CLI (`train` / `evaluate` / `check-data` / `infer` subcommands) | Done | 52 in `test_pipeline.py` |
| Full training run + published weights | Baseline trained and scored on validation ([results](docs/2026-11-16-training-results.md)); no weights published | — |

**Total: 364 tests (60–80 s, fully synthetic).** "Done" above means implemented and tested. The tests use synthetic frames only, including a miniature tennis dataset with 1280x720 frames and the real `Label.csv` header; `test_pipeline.py` trains, evaluates and runs inference with the real TrackNet at 128x72. They pass on CPU; on a machine with CUDA the Trainer picks the GPU, so the Trainer and pipeline tests run there.

## Known Issues

Details, file references and the measured preflight numbers are in [docs/2026-11-15-pre-training-review.md](docs/2026-11-15-pre-training-review.md) §4–§5. Training results are in [docs/2026-11-16-training-results.md](docs/2026-11-16-training-results.md).

**Shown so far:** one 30-epoch baseline, scored on the validation game only: F1 0.949 at 4 px, 512x288, and 0.771 at source resolution. Nothing has been evaluated on the test split.

**Found by the baseline:** predictions sit a constant 1.1 px left of the labels on the validation game (+0.1 px on the training games). The dataset's labels are about 1.5 source pixels right of the ball in game1–7, and mirroring half the training windows makes the sign of that offset depend on the orientation. Training without the flip (`horizontal_flip: false`, configs/tennis_v2_noflip.yaml) removes the offset: that run's mean horizontal error is −0.01 px, and its validation F1 is 0.903 at 4 px of the source against the baseline's 0.771 (0.947 against 0.949 at 4 px, 512x288).

**Open:**

- **Batch size and learning rate.** Batch 2 (the default) peaks at 1,769 MiB and ran at 8.6–8.7 steps per second through the full epoch. Batch 8 ran out of GPU memory as validation started; batch 4 was not tried. Reference codebases use Adam at 1e-3 with batch 8–10; we use AdamW at 1e-4 with weight decay 0.01 and batch 2. `optimizer` (`AdamW` or `Adam`), `learning_rate` and `weight_decay` are set in the config. Under `Adam`, `weight_decay` is an L2 penalty and PyTorch's default is 0, so set it to 0 when reproducing the reference recipe.
- **`torch.compile`** needs a C compiler, which this machine lacks, so `configs/default.yaml` sets `compile_model: false` and the speed-up is unmeasured. With it set to true on a CUDA machine without a compiler, `train` stops at startup and says how to fix it.
- **Mixup** is wired in as `mixup_alpha` (0, off, by default); its run, `tennis_v2_mixup` (configs/tennis_v2_mixup.yaml, the no-flip recipe plus `mixup_alpha: 0.5`), is planned and not started. **Median background frame** input is implemented (`background_frame: true`); its run, `tennis_v2_bg` (configs/tennis_v2_bg.yaml, the no-flip recipe plus the background frame), was started on 2026-11-16 and has not finished, so its effect is unknown. **Colour jitter strength** (0.3 brightness, contrast and saturation, always applied, by default) is stronger than WASB's (0.1 at p = 0.5) and untuned; it is set by `color_jitter`, and its run, `tennis_v2_jitter` (configs/tennis_v2_jitter.yaml, the no-flip recipe at WASB's strength), is planned and not started.

**Limitations left in place:**

- Up to two frames at the end of each clip are neither trained on with full context nor scored, because windows are not padded (9 frames in val, 35 in test). This matches WASB's evaluation.
- The 4 px comparison uses `<=`; WASB uses `<`. This only matters at a distance of exactly 4.0.
- At the source-resolution tolerance (1.6 px at 512x288), ground-truth quantisation of up to 0.29 px is a meaningful share of the budget. If results sit near the boundary, switch that protocol to label coordinates.
- Mixed resolutions inside one clip are not supported: the source size is read from the first frame. `evaluate` also stops with an error if the evaluated split's clips differ in source size, or if resizing them to `input_size` changes the aspect ratio, because no single source-pixel tolerance would apply.
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

# Continue a run from its last checkpoint, with the config that started it.
# With --max-steps, N must exceed the checkpoint's step.
uv run python main.py train --config configs/default.yaml \
  --resume checkpoints/tennis_v2_baseline/latest.pt

# Evaluate a checkpoint; prints JSON with both protocols. --split defaults to test,
# so pass --split val until the model is final.
uv run python main.py evaluate --config configs/default.yaml \
  --weights checkpoints/tennis_v2_baseline/latest.pt --split val
```

- **Log lines** carry a timestamp; step lines add steps per second and, on a GPU, peak GPU memory; epoch lines add validation F1, precision and recall and separate train and validation seconds. The training loss is printed in scientific notation: on the baseline its epoch mean was 4.96e-5 at epoch 5 and 2.36e-5 at epoch 30, which four decimals showed as 0.0000.
- **`train` will not overwrite a run.** If `checkpoints/<experiment_name>/latest.pt` exists, `train` stops before doing anything unless `--resume` names that same file: a fresh run, a resume from another run's checkpoint under this config, or a resume from this run's own `best.pt` would all replace it. Set a new `experiment_name` to start a new run, including one branched from an existing checkpoint. A `--resume` file that does not exist is reported as such.
- **Resume is per epoch.** Training continues from the epoch after the checkpoint's. A checkpoint whose epoch was cut short by `--max-steps` logs a warning and the rest of that epoch is skipped. A config that differs from the one stored in the checkpoint (apart from `epochs` and `max_steps`) logs a warning listing the differences. A key added to the config after the checkpoint was saved is compared using the value the code used before the key existed, so resuming an old checkpoint does not warn about `horizontal_flip`, `background_frame`, `color_jitter` or `mixup_alpha` when nothing changed.
- **The config must hold only plain YAML types** (mappings, lists, strings, numbers, booleans, null). Checkpoints store it and are loaded with `weights_only=True`; the Trainer rejects anything else at startup.
- **Experiment configs inherit.** A config file must have every key, or name a base config with `base: <path>` (relative to the file that names it) and set only what differs; bases can chain. Nested mappings such as `data:` merge key by key. `configs/tennis_v2_noflip.yaml` is `default.yaml` without the flip. `configs/tennis_v2_bg.yaml`, `configs/tennis_v2_mixup.yaml` and `configs/tennis_v2_jitter.yaml` are `tennis_v2_noflip.yaml` plus one change each: `background_frame: true`, `mixup_alpha: 0.5`, and `color_jitter` at WASB's 0.1 with `p: 0.5`.
- **`background_frame: true`** adds each clip's median background frame as a fourth input image (12 channels). Every clip decodes up to 50 of its frames when a dataset is built, which takes 28 s for the tennis train and val splits (65 clips).
- **`color_jitter`** sets the colour jitter on training windows: `brightness`, `contrast`, `saturation`, and `p`, the share of windows jittered. One draw covers every image in a window, background included. The default, 0.3 with `p: 1.0`, draws exactly what the jitter drew before it was configurable.
- **`mixup_alpha`** above 0 turns on TrackNetV3's sample mixup (they use 0.5): each training batch is blended with a random permutation of itself, frames and heatmaps alike, with one weight `max(lam, 1 - lam)` per sample and `lam` drawn from Beta(alpha, alpha). Validation is never mixed. With batch size 2 the permutation pairs each sample with itself half the time, so about half the batches are not mixed. The default, 0, draws nothing; a run with mixup on does not share the off run's sample order after the first epoch, because mixup draws from torch's global generator, which also seeds each epoch's shuffle. The training loss of a mixup run has a floor above zero from the soft targets, about 1e-5 by calculation, so compare runs on validation F1, not training loss.
- **`evaluate` builds the model the checkpoint was trained as,** from the config stored in it. If the checkpoint's `background_frame` differs from the config passed with `--config`, it logs a warning and uses the checkpoint's. A bare state dict is taken to be the default 9-channel V2 model.
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

```python
import torch
from models import build_model, load_model

# The V2 model a config trains: 12 input channels with background_frame: true
model = build_model({"background_frame": True})

# A checkpoint's model, built as it was trained, and its stored config
# (None for a bare state dict, which loads as the 9-channel default)
model, config = load_model("checkpoints/tennis_v2_baseline/best.pt", torch.device("cpu"))
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

`--model` takes a trainer checkpoint (`latest.pt` or `best.pt`) or a bare state dict for the default V2 `TrackNet()`. A trainer checkpoint is built as it was trained, from the config stored in it; if that config has `background_frame: true`, every window also gets the median of up to 50 evenly spaced frames of the input video as its background, so give such a model one static-camera clip: a video with cuts or camera motion has no meaningful median. `infer` always works at 512x288. Frames are converted to RGB, as in training.

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

1. **Phase 1 — V2 components:** model, data, training, inference pipelines and CLI. **Implemented and run end to end on the real tennis dataset.**
2. **Phase 2 — V5 architecture:** MDD module + R-STR head wired through `tracknet_v5()`. **Implemented, training deferred** (see the note under V5 above).
3. **Phase 3 — Training runs (in progress):** the V2 tennis baseline is trained and analysed ([results](docs/2026-11-16-training-results.md)). The horizontal offset it exposed is removed by training without the flip (run `tennis_v2_noflip`, done). Now: a median background frame as extra input (run `tennis_v2_bg`, started 2026-11-16). Next: mixup and colour-jitter strength (both configurable; runs `tennis_v2_mixup` and `tennis_v2_jitter` planned), each against the no-flip run.
4. **Phase 4 — Multi-sport and pickleball:** add badminton and multi-sport pre-training; collect and label our own pickleball footage.
5. **Future:** On-device (iOS) deployment, court detection, shot classification, bounce visualisation, automated scoring. See the specs under `docs/`.
