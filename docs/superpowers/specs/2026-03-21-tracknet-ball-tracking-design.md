# TrackNet Ball Tracking System — Design Specification

**Date:** 2026-03-21
**Status:** Draft. Revised 2026-11-15 after the [pre-training review](../../2026-11-15-pre-training-review.md); revisions are marked inline as "Revised 2026-11-15".
**Goal:** Production-ready, sport-agnostic ball detection and tracking for tennis, badminton, and pickleball

### Revision summary (2026-11-15)

The pre-training review changed these parts of the design. The "In code" column says which revisions the code implements; the review's §4 gives the commit for each, and its §4.4 the real-data preflight.

| Section | Change | In code |
|---|---|---|
| §1 | TrackNetV5 now has a self-described official SDK (study-only terms, no weights); independent retrains do not reproduce its numbers | — |
| §4, §5 | Our MDD and R-STR differ from the SDK, and MDD does not train from its zero initialisation. V5 is deferred | No (deferred) |
| §6 | Heatmap radius 2.5 px at 512x288, not 30. Loss must be computed in float32 from logits | Yes |
| §7 | First run is V2 on tennis. Training windows slide by 1. Colour jitter is sampled once per window | Yes |
| §9 | Real datasets do not share one CSV layout; labels must be rescaled to 512x288 | Yes, for our layout and the tennis `Label.csv`; rescaling goes through pixel centres. Pickleball not parsed yet |
| §11, §13 | Tolerance is 4 px at 512x288; metric conventions spelled out | Yes. Both tolerances are reported (4 px at 512x288, and 4 px at source resolution); tail frames are not scored; training uses game1–6 |
| §12 | RacketVision released; dataset table and links refreshed | — |

---

## 1. Problem Statement

Build a system that detects and tracks small, fast-moving balls in racquet sport videos from a single camera. The ball is typically <10px in broadcast frames, frequently occluded, and motion-blurred at high speeds (up to 417 km/h in badminton). This is the foundational layer for a SwingVision-like sports analysis platform.

### Why TrackNet

Research confirms that multi-frame heatmap-based detection (TrackNet) fundamentally outperforms single-frame object detectors for this task:

- **TrackNetV3: 97.5% tracking accuracy** vs **YOLOv7: 53.5%** on shuttlecock tracking
- Temporal context from consecutive frames is essential for detecting motion-blurred, sub-10px balls
- YOLO/RT-DETR/DETR are designed for multi-class multi-object detection — overkill and underperforming for single tiny ball tracking
- Foundation models (SAM2, GroundingDINO) are too slow for real-time inference

### Why Not Alternatives

| Approach | Accuracy (ball) | FPS | Temporal? | Verdict |
|----------|----------------|-----|-----------|---------|
| TrackNetV3 | 97.5% | ~30 | Yes (3 frames) | Best for ball |
| TrackNetV5 | 98.6% (paper, own 7:3 split, no test set) | ~38 | Yes (MDD+R-STR) | Unconfirmed. Revised 2026-11-15: study-only SDK, no weights; independent retrains reached F1 .915 and .773 |
| YOLOv11 | ~53-96% (varies) | 290 | No | Too inaccurate for ball |
| YO-CSA-T (YOLO+attention) | 90.4% mAP@0.75 | 130 | Partial | Better, still below TrackNet |
| RT-DETR | 53.1% AP (COCO) | 108 | No | No sports benchmarks |
| RF-DETR | 60.5% AP (COCO) | 100 | No | Promising, not for ball |
| SAM2 | N/A (segmentation) | 13-44 | Yes | Too slow, overkill |

YOLO is complementary (player/court detection), not competing.

---

## 2. Architecture Overview

The system is built in two phases with clean module boundaries. The V2 backbone is shared and unchanged between phases.

### Phase 1: V2 Baseline (working system)

```
3 RGB frames (512x288)
  -> Channel concatenation (9ch)
  -> V2 U-Net Backbone (encoder-decoder with skip connections)
  -> 3 Sigmoid heatmaps (one per frame)
  -> Post-processing (threshold -> centroid)
  -> Trajectory rectification (interpolate gaps)
  -> Kalman smoother (noise reduction)
  -> Ball (x, y) per frame
```

### Phase 2: V5 Enhancement (bolt-on modules)

```
3 RGB frames (512x288)
  -> MDD module (signed polarity + learnable attention -> 13ch)  <- ADD BEFORE
  -> V2 U-Net Backbone (UNCHANGED, only input channels 9->13)
  -> R-STR head (Transformer residual refinement)               <- ADD AFTER
  -> Refined heatmaps
  -> Post-processing -> Trajectory rectification -> Kalman smoother
  -> Ball (x, y) per frame
```

### Future Phases (not in scope)

- Phase 3: Court detection + homography (pixel coords -> real-world positions)
- Phase 4: Shot classification (trajectory patterns -> serve/forehand/backhand/volley)
- Phase 5: Line calling (ball landing + court lines -> in/out)
- Phase 6: Automated scoring & stats dashboard

---

## 3. V2 U-Net Backbone

### Encoder (downsampling path)

| Block | Layers | In Channels | Out Channels | Pre-Pool Size (skip tap) | Post-Pool Size |
|-------|--------|-------------|--------------|--------------------------|----------------|
| Down1 | 2x (Conv3x3 + BN + ReLU) + MaxPool2x2 | 9 (or 13 for V5) | 64 | 512x288 | 256x144 |
| Down2 | 2x (Conv3x3 + BN + ReLU) + MaxPool2x2 | 64 | 128 | 256x144 | 128x72 |
| Down3 | 2x (Conv3x3 + BN + ReLU) + MaxPool2x2 | 128 | 256 | 128x72 | 64x36 |

**Note:** Skip connections are tapped from the pre-pool feature maps (before MaxPool). The decoder concatenates with these higher-resolution skip tensors.

**V2 -> V5 weight transfer:** When switching from V2 (9ch input) to V5 (13ch input), the first conv layer weights cannot be directly loaded. Strategy: load V2 weights for the first 9 input channels, zero-initialize the 4 new MDD channels. Only the first conv layer is affected; all other layers transfer directly.

### Bottleneck

| Block | Layers | In Channels | Out Channels | Output Size |
|-------|--------|-------------|--------------|-------------|
| Bottleneck | 3x (Conv3x3 + BN + ReLU) | 256 | 512 | 64x36 |

### Decoder (upsampling path with skip connections)

| Block | Layers | In Channels | Out Channels | Output Size |
|-------|--------|-------------|--------------|-------------|
| Up1 | Upsample2x + Concat(skip3) + 2x Conv3x3 | 512+256=768 | 256 | 128x72 |
| Up2 | Upsample2x + Concat(skip2) + 2x Conv3x3 | 256+128=384 | 128 | 256x144 |
| Up3 | Upsample2x + Concat(skip1) + 2x Conv3x3 | 128+64=192 | 64 | 512x288 |

### Output Head

| Layer | In | Out | Activation |
|-------|-----|-----|-----------|
| Conv1x1 | 64 | 3 | Sigmoid |

Output shape: `(batch, 3, 288, 512)` — 3 heatmaps, one per input frame.

### Conv Block Pattern

Every convolution block follows: `Conv2d(3x3, padding=1) -> GroupNorm -> ReLU`

**GroupNorm over BatchNorm:** At batch size 2 (constrained by 8GB VRAM), BatchNorm statistics are noisy and unstable. GroupNorm (num_groups=8 or 16) is independent of batch size and performs significantly better at small batch sizes. This is a modernization from the original TrackNet papers which used BatchNorm with larger batches.

Weight initialization: Kaiming uniform for conv layers; GroupNorm weight=1, bias=0.

---

## 4. MDD Module (V5 — Phase 2)

Motion Direction Decoupling sits before the backbone. It computes signed frame differences to preserve trajectory direction (unlike V4's absolute differences which lose polarity).

### Computation

```python
# Raw frame differences
D_prev = I_t - I_{t-1}      # shape: (3, H, W)
D_next = I_{t+1} - I_t      # shape: (3, H, W)

# Signed polarity decomposition (per-channel)
P_plus_prev  = ReLU(D_prev)   # brightening (arrival)
P_minus_prev = ReLU(-D_prev)  # darkening (departure)
P_plus_next  = ReLU(D_next)
P_minus_next = ReLU(-D_next)

# Learnable attention mapping (adaptive sigmoid)
k_alpha = 5.0 / (0.45 * abs(tanh(alpha)) + epsilon)
m_beta  = 0.6 * tanh(beta)
A = sigmoid(k_alpha * (abs(x) - m_beta))

# Attention maps (2 channels each, from positive+negative polarity)
A_prev = f(P_plus_prev, P_minus_prev; alpha, beta)  # (2, H, W)
A_next = f(P_plus_next, P_minus_next; alpha, beta)  # (2, H, W)

# Final concatenation
X_in = concat(I_{t-1}, A_prev, I_t, A_next, I_{t+1})  # (13, H, W)
#              3ch      2ch    3ch   2ch     3ch = 13 channels
```

### Parameters

- `alpha`, `beta`: 2 learnable scalars (initialized to 0), trained end-to-end
- No additional conv layers — this is a lightweight preprocessing module

**Revised 2026-11-15:** zero initialisation does not train. With `alpha = 0` the gradient on `alpha` is exactly zero and the attention collapses to four discrete values, while the gradient on `beta` is enormous (verified locally). The authors' SDK initialises a = 0.2 and b = 0.15, uses epsilon 1e-6, and differences BT.601 luminance, not per-channel RGB. Fix before any V5 training; see the [review](../../2026-11-15-pre-training-review.md) §3.2 and §4.3.

---

## 5. R-STR Head (V5 — Phase 2)

Residual-Driven Spatio-Temporal Refinement sits after the decoder. Instead of reconstructing heatmaps from scratch, it predicts a residual correction on draft heatmaps.

### Pipeline

1. **Draft generation:** Decoder features -> 1x1 conv -> raw logits (3, H, W). **No sigmoid applied here** — logits stay in unbounded space.
2. **MDD fusion:** Concatenate raw draft logits with MDD attention maps -> `Draft_MDD`
3. **Stochastic masking (training only):** `Dropout(Draft_MDD, p=0.1)`
4. **TSATTHead:** Predicts correction tensor `delta` (same shape as draft logits)
5. **Final output:** `H_final = sigmoid(Draft_MDD + delta)` — sigmoid applied **once** at the very end

**Important:** Sigmoid is applied only once, after the residual correction. Draft heatmaps are raw logits throughout the R-STR pipeline to avoid double-sigmoid compression.

### TSATTHead (Transformer)

| Parameter | Value |
|-----------|-------|
| Patch size | 16x16 |
| Sequence length | 32x18 = 576 spatial patches per frame, x3 frames = 1728 tokens |
| Embedding dimension | 128 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Feed-forward hidden dim | 256 |
| PixelShuffle upscale factor | 4 (from patch grid back to full resolution) |

**Architecture:**
1. **Patch embedding:** Non-overlapping 16x16 patches from each frame's draft, flattened and linearly projected to 128-dim
2. **Positional encoding:** Factorized spatio-temporal encodings (separate spatial position + temporal frame index, added to patch embeddings)
3. **Transformer encoder:** 2 layers of factorized attention — each layer applies temporal self-attention across frames first, then spatial self-attention within each frame (TimeSformer-style)
4. **Output:** Linear projection -> reshape to patch grid -> PixelShuffle(4) -> full-resolution residual map `delta`

**Revised 2026-11-15:** the authors' SDK differs from this table. It uses joint attention over all 1,728 tokens (not factorised), 4 layers, width 256, 2 heads, feed-forward 1024, and a parameter-free gating fusion; its head is about 3.4M parameters against our 0.54M. Its upscale is PixelShuffle(16), equal to the patch size, so the "PixelShuffle(4)" entries above are a misreading and our Linear-plus-unpatchify is already equivalent. Full comparison in the [review](../../2026-11-15-pre-training-review.md) §3.2.

**Note:** These hyperparameters are initial estimates based on the V5 paper's constraint of only 3.7% FLOP increase over V4. The patch count (1728 tokens) is feasible for self-attention on 8GB VRAM. Exact values may need tuning during implementation.

### Why residual learning

The draft heatmaps are already 90%+ correct; R-STR only needs to fix the hard cases (occlusion, blur, ambiguity). Learning a correction is significantly more convergent than full reconstruction.

---

## 6. Loss Function

### Weighted Binary Cross-Entropy (WBCE) — Focal Loss Variant

```
L = -1/N * sum[(1-p_i)^2 * y_i * log(p_i) + p_i^2 * (1-y_i) * log(1-p_i)]
```

Where:
- `p_i` = predicted probability per pixel (after sigmoid)
- `y_i` = ground truth label (0 or 1, or continuous in [0,1] when mixup augmentation is used)
- `(1-p_i)^2` = dynamic weight for positive class (ball pixels) — down-weights easy positives
- `p_i^2` = dynamic weight for negative class (background pixels) — down-weights easy negatives

This is a focal-loss-style dynamic weighting (gamma=2) adapted for the TrackNet WBCE formulation from the V5 paper. It addresses the extreme class imbalance: the ball occupies ~0.01% of pixels. The dynamic weights automatically focus training on hard-to-classify pixels near the ball boundary.

**Revised 2026-11-15 — numerical stability:** computing this from post-sigmoid probabilities inside bf16 autocast returns NaN once predictions are confident, because `1 - 1e-7` rounds to 1.0 in bf16 (reproduced on CPU and CUDA). The model must return logits during training and the loss must be computed in float32 via `logsigmoid`.

**Note on mixup compatibility:** When mixup augmentation blends two samples, `y_i` becomes continuous (e.g., 0.7 instead of 1.0). The formula handles this naturally since it is defined for continuous `y_i` in [0,1].

### Ground Truth Generation

- Binary heatmap with a filled circle of radius `r` centered on the annotated ball position
- `r = 30` pixels for standard resolution (512x288 from 1280x720 native)
- `r = 40` pixels for higher resolution sources (512x288 from 1920x1080 native)
- **Revised 2026-11-15: use `r = 2.5` pixels at 512x288.** The values above come from the arXiv TrackNetV5 paper, which has no code or ablation for them. TrackNetV3, V4, WASB and BlurBall all use 2.5 at 512x288; RacketVision uses a Gaussian with sigma 3.5. A 30 px disc is a 60 px-wide blob that must be localised to within 4 px.
- Label coordinates are in the source video's resolution and must be rescaled to 512x288 before the target is drawn. As implemented, the mapping goes through pixel centres, `x_model = (x_src + 0.5) * (512 / source_width) - 0.5`, the way `cv2.resize` maps pixels, and inference applies the inverse (`data/heatmap.py`). The published TrackNet code scales by `x * scale`, which at 1280→512 leaves the target 0.3 px off the ball.
- Visibility classes: 0 = not visible (all-zero heatmap), 1 = visible, 2 = partially occluded (still labeled)

---

## 7. Training Configuration

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| LR schedule | Multi-step decay (gamma=0.1) at epochs 20, 25 |
| Batch size | 2 (constrained by 8GB VRAM on RTX 3070). Revised 2026-11-15: with GroupNorm this only affects gradient noise; TrackNetV3 and WASB use batch 8–10 with Adam at 1e-3, worth trying if VRAM allows |
| Epochs | 30 |
| Input resolution | 512x288 |
| Frames per sample | 3 consecutive |
| Output | 3 heatmaps (MIMO) |

### Data Augmentation

1. **Mixup** (proven in V2/V3): Blend two training samples with random alpha
2. **Horizontal flip** with label coordinate adjustment
3. **Color jitter**: Brightness, contrast, saturation variation. Revised 2026-11-15: sample the jitter parameters once per 3-frame window; independent per-frame jitter corrupts the motion cue between frames
4. **Motion blur simulation**: Directional Gaussian blur on ball patches
5. **Copy-paste augmentation**: Paste ball patches onto different frames (effective for small objects)

### Training Strategy

1. **Train V2 baseline** on TrackNet tennis dataset (~20K frames) + CoachAI badminton dataset (~78K frames)
2. **Fine-tune** per sport with sport-specific datasets
3. **For pickleball**: Transfer from tennis/badminton weights, fine-tune on ~12K available frames
4. **RacketVision (435K frames, 3 sports)**: Revised 2026-11-15: released 2026-04-11. Only 64,042 frames carry ball labels (every 5th frame), so using it for pre-training needs a per-frame loss mask, and commercial rights to its footage are unresolved. See [data-sources.md](../../data-sources.md).

**Revised 2026-11-15 — first run:** V2 on tennis only, after the fixes in the [review](../../2026-11-15-pre-training-review.md) §4. The first experiment after a baseline is a median background frame concatenated to the input (12 channels), the best-supported addition in the recent literature. V5 is deferred.

### PyTorch Training Optimizations

- **Mixed precision:** Use `torch.amp.autocast("cuda", dtype=torch.bfloat16)`. bf16 is preferred on Ampere GPUs (RTX 3070) — wider dynamic range than fp16, no GradScaler needed.
- **torch.compile:** Wrap model with `torch.compile(model)` for 1.5-2x speedup. Works well with U-Net (skip connections are standard tensor ops). Use **static input sizes** (512x288) — avoid `dynamic=True`. Revised 2026-11-15: off by default (`compile_model: false`) because the dev machine has no C compiler; the speed-up is unmeasured there.
- **Gradient checkpointing:** Use `torch.utils.checkpoint.checkpoint(use_reentrant=False)` on encoder blocks to reduce peak VRAM ~60% at ~25% training time cost. Enable for V5's R-STR Transformer on 8GB GPU.
- **DataLoader:** Use `pin_memory=True`, `num_workers=4`, `persistent_workers=True`.
- **Gradient zeroing:** Use `optimizer.zero_grad(set_to_none=True)` for slight memory savings.
- **Note:** `torch.cuda.amp.autocast()` is deprecated since PyTorch 2.4. Use the unified `torch.amp` API.

---

## 8. Post-Processing Pipeline

### Step 1: Heatmap to Ball Position

```python
# Threshold the sigmoid heatmap
binary = (heatmap > 0.5).float()

# Find largest connected component (filters noise)
components = connected_components(binary)
largest = max(components, key=area)

# Centroid of largest component = ball position
cx, cy = centroid(largest)

# Scale back to original resolution
cx_orig = cx * (orig_width / 512)
cy_orig = cy * (orig_height / 288)
```

### Step 2: Trajectory Rectification (from V3)

When the ball is not detected in a frame (heatmap below threshold):
1. Look at detected positions in surrounding frames (window of ~8 frames)
2. Fit a trajectory curve (polynomial or spline) to known positions
3. Interpolate the missing position from the fitted curve
4. Apply only if trajectory consistency score exceeds threshold

### Step 3: Kalman Filter Smoothing

- **State:** `[x, y, vx, vy]` (position + velocity)
- **Measurement:** Detected ball position `(cx, cy)` from Step 1
- **Process noise:** Tuned for ball dynamics (higher than typical — balls change direction rapidly)
- **Purpose:** Smooth noisy detections, not predict missing ones (that is Step 2's job)

---

## 9. Data Pipeline

### Dataset Format

All datasets use a unified CSV format per video:

```csv
Frame,Visibility,X,Y
0,1,423,187
1,1,425,185
2,0,0,0
3,1,430,180
```

- `Frame`: Frame index (0-based)
- `Visibility`: 0=invisible, 1=visible, 2=partially occluded
- `X, Y`: Ball center in pixel coordinates (0,0 if invisible)

This matches the TrackNet convention used by all existing datasets.

**Revised 2026-11-15:** that last sentence is wrong. This is our internal target format; real datasets need conversion. The tennis set's `Label.csv` uses `file name,visibility,x-coordinate,y-coordinate,status`, with empty coordinates when visibility is 0 and a visibility value 3 that we treat as ball-absent (confirmed on the downloaded files 2026-11-15; `data/labels.py` reads it); the pickleball labels use `Frame,Ball,x,y` with normalised coordinates; RacketVision matches our columns but at 1920x1080 and with labels on every 5th frame only. See [data-sources.md](../../data-sources.md).

### PyTorch Dataset Class

```python
class TrackNetDataset(Dataset):
    def __init__(self, video_path, label_path, seq_len=3, transform=None):
        # Index video frames lazily (store frame offsets, decode on demand)
        # Load CSV labels into memory (small)
        # Pre-extract frames to disk as an alternative for faster training

    def __getitem__(self, idx):
        # Decode 3 consecutive frames from video (lazy, on-demand)
        # frames: (seq_len * 3, H, W) — concatenated RGB channels, normalized to [0,1]
        # heatmaps: (seq_len, H, W) — binary target heatmaps
        return frames, heatmaps
```

### Input Normalization

- Divide pixel values by 255 to get [0,1] range
- No ImageNet mean/std normalization (TrackNet convention — the model learns its own normalization via early conv layers)

### Sliding Window Strategy

- **Stride = 3** (MIMO): Process frames [0,1,2], then [3,4,5], etc. Each sample produces 3 heatmaps. No overlap, no duplicate predictions per frame. This matches the original TrackNet MIMO design.
- **Revised 2026-11-15:** stride 3 is for inference only. Training windows slide by 1, as in the TrackNetV3, V4, WASB and RacketVision code: three times as many distinct windows, and each frame is seen in every slot.
- At video boundaries: pad by duplicating the first/last frame (e.g., frame 0 uses [0,0,1], last frame uses [N-2,N-1,N-1])
- **Revised 2026-11-15 (as implemented):** training and evaluation windows are not padded. Windows never run past the end of a clip, so up to two tail frames per clip are neither trained on with full context nor scored, as in WASB's evaluation. Only a clip shorter than three frames is padded, by repeating its last frame.

### Video Loading Strategy

- **Lazy loading:** Frames decoded on-demand via `cv2.VideoCapture.set(frame_index)` + `read()`. Avoids loading entire video into RAM.
- **Pre-extraction (recommended for training):** Extract all frames to disk as individual images (PNG/JPG) before training. Faster random access, enables standard image augmentation pipelines. Store in `data/{sport}/{match_name}/frames/` directory.

### Train/Val/Test Split

- **Split by video clip, never by frame.** Random frame splitting causes data leakage (adjacent frames are near-identical).
- **Convention:** 70% train / 15% val / 15% test, measured by number of video clips
- **Revised 2026-11-15 (as implemented for tennis):** split by game, not by clip count: train game1–6, validate on game7, test on game8–10 (56 / 9 / 30 clips). The test games match WASB's; WASB has no validation split and trains on game1–7
- **Cross-sport:** Each sport has its own split. No mixing of sports within a split.

### Handling "No Ball" Frames

- When `Visibility=0`, ground truth heatmap is all zeros
- Loss is still computed on these frames (the model should learn to predict all-zero heatmaps for invisible ball)
- No special masking — the WBCE dynamic weights naturally handle this (all-zero target means only the negative class term contributes)

### Supported Input Sources

- Video files (MP4, AVI) — decoded frame-by-frame with OpenCV
- Image directories — pre-extracted frames (recommended for training)
- Streaming (future) — real-time camera feed

---

## 10. Project Structure

```
playground-track-net/
  pyproject.toml
  main.py                    # Entry point (train, eval, infer CLI)
  configs/
    default.yaml             # Default training config
    tennis.yaml              # Sport-specific overrides
    badminton.yaml
    pickleball.yaml
  models/
    __init__.py
    backbone.py              # V2 U-Net encoder-decoder
    mdd.py                   # V5 Motion Direction Decoupling module
    rstr.py                  # V5 R-STR head + TSATTHead Transformer
    tracknet.py              # Full model assembly (backbone + optional MDD/R-STR)
    losses.py                # WBCE loss function
  data/
    __init__.py
    dataset.py               # TrackNetDataset class
    transforms.py            # Augmentations (mixup, flip, jitter, etc.)
    heatmap.py               # Ground truth heatmap generation
  inference/
    __init__.py
    video_preprocess.py      # Frame extraction and preprocessing
    postprocess.py           # Heatmap to coordinates + trajectory rectification
    tracker.py               # Kalman filter smoother
  training/
    __init__.py
    trainer.py               # Training loop
    evaluate.py              # Evaluation metrics (precision, recall, F1, accuracy)
  utils/
    __init__.py
    visualization.py         # Draw ball positions on video frames
  tests/
    ...
  docs/
    ...
```

### Key Design Principles

- **Modular composition:** `TrackNet(backbone, mdd=None, rstr=None)` — MDD and R-STR are optional. V2 = backbone only. V5 = backbone + MDD + R-STR.
- **Config-driven:** YAML configs for hyperparameters, architecture choices, sport-specific settings. No hardcoded values.
- **Sport-agnostic architecture, sport-specific weights:** Same model code for all sports, different trained weights loaded per sport.

### Additional Dependencies (to add to pyproject.toml)

```
pyyaml           # Config loading
scipy            # Connected components (ndimage.label), spline interpolation (UnivariateSpline)
pandas           # CSV label loading
tensorboard      # Training metrics visualization
tqdm             # Progress bars
```

### Dependencies to remove from pyproject.toml

```
filterpy         # REMOVE — unmaintained since 2018, incompatible with Python 3.14.
                 # Replaced by ~40-line custom NumPy Kalman filter implementation.
```

### Torchvision API note

Use `torchvision.transforms.v2` (not `torchvision.transforms`). The v1 API still works in 0.25 with no warnings, but v2 is the recommended path and receives all future improvements. Migration is a one-line import change — same API surface.

### Experiment Tracking

- **TensorBoard** for training curves (loss, F1, precision, recall per epoch)
- Checkpoints saved as `.pt` files (state_dict only) at `checkpoints/{sport}/{experiment_name}/`
- Save best model (by val F1) and latest model
- Log hyperparameters, git commit hash, and config YAML with each run
- Use AMP (automatic mixed precision) by default to reduce VRAM usage on 8GB GPUs
- Set random seeds (torch, numpy, python) for reproducibility

---

## 11. Inference Pipeline

```
Video file / Camera feed
  -> Frame extraction (OpenCV, 30fps)
  -> Sliding window of 3 frames
  -> Resize to 512x288, normalize to [0,1]
  -> [MDD preprocessing if V5]
  -> Channel concatenation
  -> Model forward pass (GPU, torch.compiled, bf16 inference)
  -> 3 heatmaps out
  -> Post-processing (threshold -> centroid -> scale to original resolution)
  -> Trajectory rectification (fill gaps using surrounding detections)
  -> Kalman smoothing
  -> Output: per-frame ball (x, y, confidence, visibility)
```

### Performance Targets

| Metric | Target |
|--------|--------|
| FPS (RTX 3070) | >30 (real-time at 30fps input) |
| FPS (RTX 4090) | >100 (3-frame batch processing) |
| Detection threshold | Euclidean distance <= 4px at 512x288 (revised 2026-11-15; was "original resolution", which contradicted the config. 4 px at 512x288 matches TrackNetV3, V4 and RacketVision; WASB and the arXiv TrackNetV5 paper measure at original resolution). As implemented, `evaluate` reports both: `model_res_4px` (4 px at 512x288) and `source_res_4px` (4 px at source resolution, 1.6 px at 512x288 for 1280x720 video). Per-epoch validation and best-checkpoint selection use 4 px at 512x288 |
| Target F1 (tennis) | >0.97 |
| Target F1 (badminton) | >0.97 |
| Target F1 (pickleball) | >0.90 (limited data) |

---

## 12. Training Data Strategy

### Available Datasets

| Sport | Dataset | Frames | Quality | Availability | Format |
|-------|---------|--------|---------|-------------|--------|
Revised 2026-11-15. [data-sources.md](../../data-sources.md) is the maintained list, with links, licences and a verification log.

| Sport | Dataset | Labelled frames | Quality | Availability | Format |
|-------|---------|--------|---------|-------------|--------|
| Tennis | TrackNet tennis | ~20K | High | Available (NYCU SharePoint, Kaggle) | `Label.csv` per clip, needs conversion |
| Badminton | CoachAI / TrackNetV2 | 78,200 | High | Available (NYCU SharePoint) | CSV (center-point) |
| Multi-sport | RacketVision | 64,042 of 435,179 frames | High, sparse | Released 2026-04-11 (Hugging Face); footage rights unresolved | CSV (center-point) at 1080p + racket pose |
| Table tennis | BlurBall | ~64K | High | Repo live; download host unreachable on 2026-11-15 | CSV (center-point) + blur attributes |
| Badminton | ETH one-shot shuttle detection | 20,510 | Medium (no negatives) | Available (Google Drive) | YOLO bbox |
| Pickleball | TrackNet-Pickleball | 12,810 | Low (single match, no raw video) | Labels and preprocessed arrays only | CSV, normalised coordinates |
| Table tennis | OpenTTGames | Event windows only | High | Available (CC BY-NC-SA 4.0, non-commercial) | Center-point + segmentation |
| Tennis | Roboflow (Hard Court) | 9,836 | Medium | Not re-verified | YOLO/COCO bbox |

WASB-SBDT hosts no data of its own; it links to the tennis and badminton sets above.

### Strategy

Revised 2026-11-15.

1. **Phase 1 (immediate):** Train a V2 baseline on TrackNet tennis only. Validate that the pipeline trains and that our metrics are comparable with published ones.
2. **Phase 2 (multi-sport):** Add CoachAI badminton, then joint pre-training with RacketVision and BlurBall once the loss mask and licence questions are settled. RacketVision's paper reports joint multi-sport training beats single-sport.
3. **Phase 3 (pickleball data collection):** Record and label our own pickleball matches using CVAT. Target 50K+ frames. The public set cannot carry this: one match, no raw video.
4. **Phase 4 (pickleball):** Fine-tune from the multi-sport weights. Split by contiguous rallies, never by window, and hold out a second match as the real test set.
5. **Labeling estimate:** With CVAT point interpolation, ~2-3 seconds per frame average = ~40-80 hours for 50K frames. Label the centre of the blur streak, not its leading edge (BlurBall reports +0.4 to +1.5 F1 across models).

### Annotation Pipeline

- **Tool:** CVAT v2.59+ (self-hosted, open source)
  - **Point Track mode** with linear interpolation (NOT SAM2 — SAM2 is a segmentation tool and does not work reliably for <10px ball localization)
  - Mark ball as "Outside" (shortcut: O) when invisible — maps to Visibility=0
  - Keyframe every 5-15 frames depending on trajectory complexity (linear interpolation cannot model parabolic paths, so more keyframes at bounces/direction changes)
  - Export as CVAT for Video XML, convert to CSV with a ~30-line Python script
- **Pre-annotation workflow (recommended):** Run trained TrackNet inference on unlabeled video, import predictions as pre-annotations into CVAT, manually correct errors. Dramatically faster than annotating from scratch.
- **Format:** Frame, Visibility (0/1/2), X (center), Y (center)
- **Quality:** Double-annotate 10% of frames for inter-annotator agreement

### Reference Implementations (for architecture reference only — not directly compatible)

- **TrackNetV3 (official):** https://github.com/qaz812345/TrackNetV3 — Python 3.8 / PyTorch 1.10. NOT compatible with Python 3.12+. Use as architecture reference only.
- **TrackNetV4 (official):** https://github.com/TrackNetV4/TrackNetV4 — TensorFlow, not PyTorch. Use for motion attention map design reference.
- **TrackNetV2 PyTorch port:** https://github.com/ChgygLin/TrackNetV2-pytorch — Community port, includes ncnn C++ inference. Good architecture reference.

---

## 13. Evaluation Metrics

### Detection Metrics (per-frame)

- **TP:** Predicted ball position within 4px Euclidean distance of ground truth
- **FP:** Predicted ball with no corresponding ground truth within 4px
- **FN:** Ground truth ball with no prediction within 4px
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1:** 2 * Precision * Recall / (Precision + Recall)
- **Accuracy:** (TP + TN) / total frames (including correctly predicted "no ball" frames)

**Revised 2026-11-15 — conventions.** Distances are measured at 512x288. The ground-truth position is the centroid of the target heatmap (as in TrackNetV3's evaluation code), and the prediction is the centroid of the largest connected blob. Published codebases disagree on a detection that is too far from a visible ball: TrackNetV3, V4 and WASB count it as a false positive only, while RacketVision also counts it against recall. Report both so our numbers can be compared with either group.

**As implemented (2026-11-15).** `training/evaluate.py` and `main.py evaluate` do the above, with these specifics:

- **Two tolerances, both reported:** `model_res_4px`, 4 px at 512x288 (TrackNetV3, V4, RacketVision), and `source_res_4px`, 4 px at source resolution (WASB), which is 1.6 px at 512x288 for 1280x720 video. A true positive is a distance `<=` the tolerance; WASB uses `<`.
- **Both recall conventions** under each tolerance: `recall` = TP / (TP + FN), with far detections (fp1) counted against precision only, and `recall_strict` = TP / (TP + FN + fp1). `f1` and `f1_strict` follow. Per-epoch validation and `best.pt` selection use `f1` at 4 px, 512x288.
- **Each frame scored once.** Validation and test windows use stride 3 and are not padded, so up to two frames at the end of each clip are not scored (9 frames in tennis val, 35 in test). This matches WASB's evaluation.
- **Ground truth** is the centroid of the 2.5 px target disc, at most 0.29 px from the label. At the 1.6 px tolerance that is a meaningful share of the budget.
- **Tennis split:** train game1–6, validate on game7, test on game8–10. The test games match WASB's; WASB trains on game1–7, so our model sees one game fewer.
- Tennis visibility 3 counts as ball-absent, for the target and the metrics.

### Tracking Metrics (trajectory-level)

- **Tracking accuracy:** Percentage of frames where ball is correctly located
- **Gap length distribution:** Histogram of consecutive missed detections
- **Trajectory smoothness:** Mean acceleration magnitude (lower = smoother)

---

## 14. Hardware Requirements

### Training

- **Minimum:** NVIDIA RTX 3070 (8GB VRAM), batch size 2
- **Recommended:** NVIDIA RTX 4090 (24GB VRAM), batch size 8
- **Fallback:** AWS p3/p4 instances for larger experiments
- **Gradient checkpointing:** Enable for V5 R-STR Transformer on 8GB GPU

### Inference

- **Real-time (30fps):** Any modern NVIDIA GPU (RTX 2060+)
- **Edge deployment (future):** ONNX/TensorRT export for Jetson or mobile
- **CPU inference:** Possible but likely <10fps — not real-time

---

## 15. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Pickleball data insufficient | High | High | Transfer learning + active data collection via CVAT |
| RacketVision dataset unavailable | Resolved | — | Released 2026-04-11. New risks: sparse labels (20% of frames) and unresolved footage rights for commercial use. |
| Dataset licences unclear for commercial use | High | High | Most sets state no data licence. Confirm with authors before any enter a shipped model; see data-sources.md. |
| TrackNet dataset download links broken | Low | Medium | Verified live 2026-11-15: NYCU SharePoint (official) and Kaggle. |
| V5 R-STR too heavy for 8GB GPU | Medium | Low | Gradient checkpointing + bf16. Train on AWS if needed. |
| Cross-camera generalization poor | High | Medium | Train on diverse camera angles; augment aggressively |
| V5 paper details insufficient for implementation | Realised | Medium | Our MDD and R-STR differ from the authors' SDK, and the paper's results are unreproduced. V5 deferred; V2 plus a background frame is the first-run path. |
| BatchNorm instability at batch size 2 | N/A | N/A | Mitigated: using GroupNorm instead |
| Real-time performance on edge devices | Medium | Low | Not in scope for Phase 1; ONNX/TensorRT for future |

---

## 16. Success Criteria

### Phase 1 (V2 Baseline) — Done when:
- [ ] Model trains on tennis dataset and converges
- [ ] F1 > 0.95 on tennis test set
- [ ] Inference runs at >30fps on RTX 3070
- [ ] End-to-end pipeline: video in -> ball coordinates out

### Phase 2 (V5 Enhancement) — Done when:
- [ ] MDD module integrated, training converges with 13ch input
- [ ] R-STR head integrated, measurable F1 improvement over V2
- [ ] F1 > 0.97 on tennis test set

### Phase 3 (Multi-sport) — Done when:
- [ ] Fine-tuned weights for tennis, badminton, pickleball
- [ ] F1 > 0.97 for tennis and badminton
- [ ] F1 > 0.90 for pickleball (with available data)
