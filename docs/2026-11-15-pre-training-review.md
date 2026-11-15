# Pre-Training Review — Data, Literature, and Code Readiness

**Date:** 2026-11-15
**Status:** Research complete. First-run decision made; code fixes not yet started.
**Scope:** What changed between the April 2026 review and the first real training run.
**Related:**
- [data-sources.md](data-sources.md) — full dataset details and verification log
- [2026-03-21 design spec](superpowers/specs/2026-03-21-tracknet-ball-tracking-design.md) — carries revision notes pointing here
- [2026-04-14 architecture review synthesis](2026-04-14-architecture-review-synthesis.md) — corrected in its §13

---

## 1. Summary

Four research agents surveyed new training data, known-source status, model architectures, and training recipes. Three conclusions:

1. **Nothing published since April justifies changing the backbone before a first run.** Dedicated multi-frame heatmap models still win, and all reported gains over TrackNetV3/V4 are between 0.1 and 1.5 F1.
2. **The code is not ready to train.** It has defects that would break or invalidate a real run (§4). The test suite passes because it uses synthetic 512x288 fixtures and stub models.
3. **RacketVision is released, but pickleball data is still missing** (§2).

## 2. Training data

Details, links and licences are in [data-sources.md](data-sources.md). Headlines:

- **RacketVision released 2026-04-11** on Hugging Face (7.54 GB). Only 64,042 of 435,179 frames carry ball labels (every 5th frame at 25 fps), so it needs a per-frame loss mask with our 3-in/3-out model. Commercial rights to its YouTube footage are unresolved.
- **Tennis and badminton core sets are unchanged.** Both download from the command line; the official tennis source is an NYCU SharePoint zip we had not listed.
- **New candidates:** BlurBall (table tennis, ~64K frames, our CSV format) and an ETH badminton set (20,510 frames, stationary non-broadcast camera).
- **Pickleball:** no new public data. The one existing set has labels and preprocessed arrays only; its raw video was never published.

## 3. Literature

### 3.1 Architecture

| Model | Source | Params | Evidence | Code and weights |
|---|---|---|---|---|
| TrackNetV5 (MDD + R-STR) | arXiv 2512.02789 | 14.77M | F1 .9859 on its own 7:3 split with no test set. Independent retrains reached .915 and .773 | Self-described official SDK, study-only terms, no weights |
| TrackNetV5 (NYCU, different paper) | ICIP 2026 | 10.09M; Lite 2.58M | F1 98.47 (Lite 98.06) vs V4 97.99 and V3 97.64, same test protocol | Weights released, no licence stated |
| TrackNetV6 | ICMR 2026 | 35.8–58M | Best reported numbers, appendix only | Demo and weights only |
| BlurBall | arXiv 2509.18387, CVPRW 2026 | 1.50M | F1 97.17 vs WASB 96.00 and V3 95.93 on table tennis | MIT |
| WASB-HRNet | arXiv 2311.05237, BMVC 2023 | 1.5M | Beats V2 in four of five later comparisons; roughly level with V3, sometimes below | Code MIT; weights licence not stated |

Findings:

- **The arXiv TrackNetV5 is weaker than it looked in April.** Its own ablation credits MDD with about 0.2 F1, and backbone plus R-STR alone (.9866) scores above the full model (.9859).
- **Numbers across papers are not comparable.** The same TrackNetV3 scores 97.64 in one paper and 94.69 in another. Rankings only hold within one paper's protocol.
- **Median background frame is the best-supported addition.** In RacketVision's tennis ablation, adding it to TrackNetV3 took recall from .725 to .830 and mean localisation error from 7.95 to 3.07 px. TrackNetV3's own ablation reports +0.41 F1. It assumes a static camera.
- **General-purpose detectors have not caught up.** YOLO11 on RacketVision tennis: recall .416, error 17.5 px, against .797–.830 and 1.7–3.1 px for TrackNetV3. No paper benchmarks SAM-family, point-tracker or DINO-feature approaches on these datasets, so that gap is untested, not closed.
- **On-device evidence is thin.** WASB is estimated at about 28 GMACs per 3-frame pass against roughly 112 for V2. The only device number found is a hobby repo reporting 3–5 ms per frame on an A17 Pro, with no method given. No peer-reviewed Neural Engine measurement exists for HRNet or U-Net trackers.

### 3.2 Our V5 implementation versus the authors' SDK

The SDK (`thaonan/TrackNetV5-SDK`) appeared after we wrote ours from the paper. Its terms forbid reuse beyond study, so these are differences to reimplement, not copy.

| Component | Ours | SDK |
|---|---|---|
| MDD init | alpha = beta = 0 | a = 0.2, b = 0.15 |
| MDD colour handling | Sigmoid per RGB channel, then mean | BT.601 luminance before differencing |
| MDD epsilon | 1e-8 | 1e-6 |
| R-STR fusion | Learnable 1x1 conv over 3 logits + 4 attention channels | Parameter-free gating of logits by the "brightening" attention |
| R-STR attention | Factorised temporal then spatial; 2 layers, width 128, 4 heads | Joint over all 1,728 tokens; 4 layers, width 256, 2 heads |
| R-STR residual | Added to the draft before dropout | Added to the dropped-out draft |
| Head size | 0.54M (V5 total 10.69M) | ~3.4M (V5 total 14.77M) |
| Backbone | 2 convs in Down3 and Up1; 10.15M | 3 convs in Down3 and Up1; 11.33M |

The spec's "PixelShuffle(4)" note is a misreading: the SDK uses PixelShuffle(16), equal to the patch size, which our Linear-plus-unpatchify already matches.

### 3.3 Training recipe versus published code

| Topic | Ours | Reference codebases |
|---|---|---|
| Heatmap target | Binary disc, radius 30 px | Radius 2.5 at 512x288 (V3, V4, WASB, BlurBall); Gaussian σ = 3.5 (RacketVision). Only the arXiv V5 paper says 30 |
| Training window stride | 3 (non-overlapping) | 1 (V3, V4, WASB, RacketVision) |
| Colour jitter | Independent per frame | Once per window (WASB) |
| Mixup | Implemented, not wired in | Batch-level, α = 0.5 (V3: +0.29 F1) |
| Background frame | None | Median frame concatenated (V3, RacketVision, NYCU V5) |
| Optimiser | AdamW 1e-4, batch 2 | Adam 1e-3, batch 8–10 (V3, WASB) |
| Far detection in metrics | Counted as FP and FN | FP only (V3, V4, WASB); RacketVision matches ours |
| Tolerance space | 4 px at 512x288 in config; "original resolution" in spec | 4 px at 512x288 (V3, V4, RacketVision); original resolution (WASB, arXiv V5) |
| Post-processing | Spline gap fill with no gate; Kalman replaces detections | Learned inpainting (V3); multi-candidate tracker (WASB); nobody uses a Kalman filter |

Batch size 2 traces back to the Keras setup of TrackNetV2/V4. With GroupNorm it affects only gradient noise.

## 4. Code readiness

**Verified** means reproduced or read in the code during this review. **Reported** means an agent's finding, not independently re-checked.

### 4.1 Blocking the first run

| # | Issue | Where | Status |
|---|---|---|---|
| 1 | **Loss returns NaN under bfloat16.** `1 - 1e-7` rounds to 1.0 in bf16, so the clamp does nothing and a confident positive pixel gives `0 * log(0)` | `models/losses.py:18`, run under autocast in `training/trainer.py` | Verified on CPU and CUDA |
| 2 | **Labels are not rescaled.** Frames are resized to 512x288 but heatmaps are drawn at raw CSV coordinates (1280x720 or 1920x1080 in every real dataset) | `data/dataset.py` | Verified by reading |
| 3 | **Label loader assumes our CSV layout.** `int(row["X"])` and fixed column names will not parse the tennis `Label.csv` (different headers, NaNs) or the pickleball CSV (normalised coordinates) | `data/dataset.py` `_load_labels` | Code verified; dataset layouts reported |
| 4 | **Inference feeds BGR; training uses RGB** | `inference/video_preprocess.py` has no colour conversion; `data/dataset.py:86` has one | Verified by reading |
| 5 | **`infer` cannot load a trainer checkpoint.** It loads the file as a bare state dict, but the trainer saves `{"model_state_dict": ...}`, with `_orig_mod.` key prefixes when the model is compiled | `main.py`, `training/trainer.py` | Verified by reading |
| 6 | **`train` and `evaluate` commands are placeholders** | `main.py` | Verified |
| 7 | **No train/val/test split logic.** One dataset object per frames directory | `data/dataset.py` | Reported |
| 8a | **`torch.compile` fails on the local machine.** No C compiler is installed, so the first compiled forward pass raises `Failed to find C compiler`. The default config enables compilation | `configs/default.yaml` `compile_model: true`; fix by installing `build-essential` or setting it false | Verified on Python 3.12 and 3.14 |

### 4.2 Recipe corrections agreed for the first run

| # | Change | Where |
|---|---|---|
| 8 | Heatmap radius 30 → 2.5 | `configs/default.yaml`, `data/heatmap.py` |
| 9 | Training stride 3 → 1 | `data/dataset.py` |
| 10 | Colour jitter sampled once per window | `data/transforms.py` |
| 11 | Evaluation: largest-blob prediction, FP/FN reported under both conventions, tolerance space stated. Ground truth stays derived from the target heatmap, which is what TrackNetV3's own evaluation does (`test.py`, checked 2026-11-15); with a 2.5 px target the centroid is within 1 px of the label | `training/evaluate.py` |

### 4.3 V5 only (deferred)

| # | Issue | Status |
|---|---|---|
| 12 | **MDD does not train from its initial state.** With alpha = 0 the gradient on alpha is exactly 0 and the attention collapses to four discrete values; beta's gradient is of order 1e7 or larger with no gradient clipping | Verified locally |
| 13 | R-STR and backbone differ from the SDK (§3.2) | Reported |

## 5. Decisions

**Done (2026-11-15):** the project moved from Python 3.12 to 3.14. All 166 tests, lint and type check give the same results on both, and uncompiled bf16 training steps run on the GPU for V2 and V5.

**Decided (2026-11-15):**

- First run is **V2 on tennis**, after the fixes in §4.1 and §4.2.
- The first experiment after a baseline is the **median background frame** input.
- **V5 is deferred.** If revisited, fix MDD and realign R-STR first (§4.3).

**Open:**

1. **Where training runs.** The local GPU is an RTX 3070 Laptop with 8 GB: enough for V2, limiting for batch size and multi-sport pre-training. Measured peak VRAM for one uncompiled bf16 training step at batch 2: about 1.8 GB for V2 and 1.9 GB for V5.
2. **Commercial intent.** If the iOS product is the goal, RacketVision, BlurBall and the ETH set need licence confirmation before entering the training mix.
3. **On-device model.** The April recommendation to swap to WASB is not settled. A fair bake-off has three candidates: V2 with background frame, WASB-HRNet, and a width-scaled MBConv U-Net in the style of NYCU's Lite model.
4. **Pickleball data.** Requires our own recordings. If we annotate, label the centre of the blur streak, which BlurBall reports is worth +0.4 to +1.5 F1 across models.

## 6. What was not verified

- Exact column headers of the tennis `Label.csv` (inferred from WASB's loader).
- What "r = 30" means in the arXiv TrackNetV5 paper; there is no ablation.
- Whether the TrackNetV5 SDK is really official (self-described; consistent with the paper's recipe and parameter count).
- Licences for the WASB weights and the NYCU TrackNetV5 code and weights.
- FLOPs for WASB and NYCU-Lite are hand estimates, not measurements.
- TrackNetV6's main paper (publisher returned 403); only its appendix was read.
- BlurBall's download host and the Kaggle `tenis_backview` page.
- Whether our Kalman filter or spline gap fill raise or lower F1.

## 7. Sources

- TrackNetV5: arXiv 2512.02789; `github.com/thaonan/TrackNetV5-SDK`
- TrackNetV5 (NYCU): ICIP 2026; `github.com/gino6178/TrackNetV5`
- TrackNetV6: ICMR 2026, DOI 10.1145/3805622.3810690; `github.com/Gi-gigi/TrackNetV6`
- TrackNetV3: `github.com/qaz812345/TrackNetV3`
- TrackNetV4: arXiv 2409.14543; `github.com/TrackNetV4/TrackNetV4`
- WASB: arXiv 2311.05237; `github.com/nttcom/WASB-SBDT`
- BlurBall: arXiv 2509.18387; `github.com/cogsys-tuebingen/BlurBall`
- RacketVision: arXiv 2511.17045; `github.com/OrcustD/RacketVision`
- TOTNet: arXiv 2508.09650
- Independent V5 retrains: `github.com/panditamey/TrackMem`, `github.com/AnInsomniacy/tracknet-series-pytorch`
- PyTorch AMP documentation (autocast and `binary_cross_entropy`)
