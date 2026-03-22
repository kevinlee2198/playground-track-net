# TrackNet Ball Tracking System — Design Specification

**Date:** 2026-03-21
**Status:** Draft
**Goal:** Production-ready, sport-agnostic ball detection and tracking for tennis, badminton, and pickleball

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
| TrackNetV5 | 98.6% (paper) | ~38 | Yes (MDD+R-STR) | SOTA, no public code |
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

**Note on mixup compatibility:** When mixup augmentation blends two samples, `y_i` becomes continuous (e.g., 0.7 instead of 1.0). The formula handles this naturally since it is defined for continuous `y_i` in [0,1].

### Ground Truth Generation

- Binary heatmap with a filled circle of radius `r` centered on the annotated ball position
- `r = 30` pixels for standard resolution (512x288 from 1280x720 native)
- `r = 40` pixels for higher resolution sources (512x288 from 1920x1080 native)
- Visibility classes: 0 = not visible (all-zero heatmap), 1 = visible, 2 = partially occluded (still labeled)

---

## 7. Training Configuration

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| LR schedule | Multi-step decay (gamma=0.1) at epochs 20, 25 |
| Batch size | 2 (constrained by 8GB VRAM on RTX 3070) |
| Epochs | 30 |
| Input resolution | 512x288 |
| Frames per sample | 3 consecutive |
| Output | 3 heatmaps (MIMO) |

### Data Augmentation

1. **Mixup** (proven in V2/V3): Blend two training samples with random alpha
2. **Horizontal flip** with label coordinate adjustment
3. **Color jitter**: Brightness, contrast, saturation variation
4. **Motion blur simulation**: Directional Gaussian blur on ball patches
5. **Copy-paste augmentation**: Paste ball patches onto different frames (effective for small objects)

### Training Strategy

1. **Train V2 baseline** on TrackNet tennis dataset (~20K frames) + CoachAI badminton dataset (~78K frames)
2. **Fine-tune** per sport with sport-specific datasets
3. **For pickleball**: Transfer from tennis/badminton weights, fine-tune on ~12K available frames
4. **RacketVision (435K frames, 3 sports)**: Monitor for dataset release — as of March 2026, the GitHub repo appears to be a placeholder with no confirmed download links. If/when available, use for unified pre-training.

### PyTorch Training Optimizations

- **Mixed precision:** Use `torch.amp.autocast("cuda", dtype=torch.bfloat16)`. bf16 is preferred on Ampere GPUs (RTX 3070) — wider dynamic range than fp16, no GradScaler needed.
- **torch.compile:** Wrap model with `torch.compile(model)` for 1.5-2x speedup. Works well with U-Net (skip connections are standard tensor ops). Use **static input sizes** (512x288) — avoid `dynamic=True`.
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
- At video boundaries: pad by duplicating the first/last frame (e.g., frame 0 uses [0,0,1], last frame uses [N-2,N-1,N-1])

### Video Loading Strategy

- **Lazy loading:** Frames decoded on-demand via `cv2.VideoCapture.set(frame_index)` + `read()`. Avoids loading entire video into RAM.
- **Pre-extraction (recommended for training):** Extract all frames to disk as individual images (PNG/JPG) before training. Faster random access, enables standard image augmentation pipelines. Store in `data/{sport}/{match_name}/frames/` directory.

### Train/Val/Test Split

- **Split by video clip, never by frame.** Random frame splitting causes data leakage (adjacent frames are near-identical).
- **Convention:** 70% train / 15% val / 15% test, measured by number of video clips
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
| Detection threshold | Euclidean distance <= 4px at original resolution |
| Target F1 (tennis) | >0.97 |
| Target F1 (badminton) | >0.97 |
| Target F1 (pickleball) | >0.90 (limited data) |

---

## 12. Training Data Strategy

### Available Datasets

| Sport | Dataset | Frames | Quality | Availability | Format |
|-------|---------|--------|---------|-------------|--------|
| Tennis | TrackNet V1 | 19,835 | High | Confirmed (Google Drive mirror) | CSV (center-point) |
| Tennis | TrackNet V2 | 20,844 | High | Degraded (original NCTU link may be broken; use NYCU GitLab or Google Drive mirror) | CSV (center-point) |
| Tennis | Roboflow (Hard Court) | 9,836 | Medium | Available | YOLO/COCO bbox |
| Badminton | CoachAI / TrackNet | 78,200 | High | Likely available (SharePoint link actively referenced in docs) | CSV (center-point) |
| Multi-sport | RacketVision | 435,179 | High | **NOT CONFIRMED** — GitHub repo likely placeholder, no download links verified | Center-point + racket pose |
| Multi-sport | WASB-SBDT | varies | High | Available (GitHub) | Multiple formats |
| Pickleball | TrackNet-Pickleball | ~12,000 | Low (single match) | Available (GitHub) | CSV (center-point) |
| Table Tennis | OpenTTGames | ~50K+ | High | Available (CC BY-NC-SA 4.0) | Center-point + segmentation |

### Download Links (verified March 2026)

- **TrackNet Tennis (Google Drive mirror):** https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut
- **TrackNet V2 (NYCU GitLab):** https://gitlab.nol.cs.nycu.edu.tw/open-source/TrackNetv2
- **CoachAI Badminton:** https://nycu1-my.sharepoint.com/:u:/g/personal/tik_m365_nycu_edu_tw/EWisYhAiai9Ju7L-tQp0ykEBZJd9VQkKqsFrjcqqYIDP-g
- **CoachAI Project (GitHub):** https://github.com/wywyWang/CoachAI-Projects
- **TrackNet-Pickleball:** https://github.com/AndrewDettor/TrackNet-Pickleball
- **WASB-SBDT:** https://github.com/nttcom/WASB-SBDT
- **RacketVision:** https://github.com/OrcustD/RacketVision (MONITOR — likely not yet released)

### Strategy

1. **Phase 1 (immediate):** Train V2 baseline on TrackNet tennis (~20K frames) + CoachAI badminton (~78K frames). Validate architecture works.
2. **Phase 2 (pickleball):** Transfer learn from tennis/badminton weights onto ~12K pickleball frames. Evaluate performance gap.
3. **Phase 3 (data collection):** Record and label additional pickleball matches using CVAT. Target 50K+ frames.
4. **Phase 4 (if RacketVision releases):** Use 435K-frame multi-sport dataset for unified pre-training. Re-fine-tune per sport.
5. **Labeling estimate:** With CVAT point interpolation, ~2-3 seconds per frame average = ~40-80 hours for 50K frames.

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
| RacketVision dataset unavailable | High | Medium | Not a blocker — train on sport-specific datasets instead. Monitor for release. |
| TrackNet dataset download links broken | Medium | Medium | Multiple mirrors exist (Google Drive, NYCU GitLab). Verify before depending on them. |
| V5 R-STR too heavy for 8GB GPU | Medium | Low | Gradient checkpointing + bf16. Train on AWS if needed. |
| Cross-camera generalization poor | High | Medium | Train on diverse camera angles; augment aggressively |
| V5 paper details insufficient for implementation | Low | Medium | V4 motion attention as fallback; TSATTHead hyperparameters estimated in spec |
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
