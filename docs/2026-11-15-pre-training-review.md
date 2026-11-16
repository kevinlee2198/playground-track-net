# Pre-Training Review — Data, Literature, and Code Readiness

**Date:** 2026-11-15
**Status:** Research complete. First-run decision made. The fixes in §4.1 and §4.2 are done (branch `fix/pre-training-blockers`) and a preflight on the real tennis dataset has passed (§4.4). No full training run yet; the open decisions are in §5.
**Scope:** What changed between the April 2026 review and the first real training run.
**Related:**
- [data-sources.md](data-sources.md) — full dataset details and verification log
- [2026-03-21 design spec](superpowers/specs/2026-03-21-tracknet-ball-tracking-design.md) — carries revision notes pointing here
- [2026-04-14 architecture review synthesis](2026-04-14-architecture-review-synthesis.md) — corrected in its §13

---

## 1. Summary

Four research agents surveyed new training data, known-source status, model architectures, and training recipes. Three conclusions:

1. **Nothing published since April justifies changing the backbone before a first run.** Dedicated multi-frame heatmap models still win, and all reported gains over TrackNetV3/V4 are between 0.1 and 1.5 F1.
2. **The code is not ready to train.** It has defects that would break or invalidate a real run (§4). The test suite passes because it uses synthetic 512x288 fixtures and stub models. *Since fixed: §4 gives the fixing commit for each defect and the real-data preflight.*
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

The "Ours" column is the code as reviewed. Heatmap radius, training stride, colour jitter, far-detection counting and tolerance space have since changed (§4.2); mixup, background frame, optimiser and post-processing have not.

## 4. Code readiness

**Verified** means reproduced or read in the code during this review. **Reported** means an agent's finding, not independently re-checked. Every issue in §4.1 and §4.2 is now fixed on branch `fix/pre-training-blockers`; the Fixed column gives the commits. File and line references describe the code as reviewed.

### 4.1 Blocking the first run

| # | Issue | Where | Status | Fixed |
|---|---|---|---|---|
| 1 | **Loss returns NaN under bfloat16.** `1 - 1e-7` rounds to 1.0 in bf16, so the clamp does nothing and a confident positive pixel gives `0 * log(0)` | `models/losses.py:18`, run under autocast in `training/trainer.py` | Verified on CPU and CUDA | `cd64d79`, `93889ed`: the model returns logits in training (`return_logits=True` on backbone, R-STR head and TrackNet) and the loss runs in float32 from logits; a non-finite loss stops training |
| 2 | **Labels are not rescaled.** Frames are resized to 512x288 but heatmaps are drawn at raw CSV coordinates (1280x720 or 1920x1080 in every real dataset) | `data/dataset.py` | Verified by reading | `e917bb1`; mapping through pixel centres in `ac4d449` (§4.5) |
| 3 | **Label loader assumes our CSV layout.** `int(row["X"])` and fixed column names will not parse the tennis `Label.csv` (different headers, NaNs) or the pickleball CSV (normalised coordinates) | `data/dataset.py` `_load_labels` | Code verified; tennis layout now confirmed on the downloaded files (§4.4) | `cae52b4`, `c403f3e`: `data/labels.py` detects our layout or the tennis one from the header. The pickleball layout is still not parsed (out of scope for the first run) |
| 4 | **Inference feeds BGR; training uses RGB** | `inference/video_preprocess.py` has no colour conversion; `data/dataset.py:86` has one | Verified by reading | `a452786`; a parity test in `5663364` |
| 5 | **`infer` cannot load a trainer checkpoint.** It loads the file as a bare state dict, but the trainer saves `{"model_state_dict": ...}`, with `_orig_mod.` key prefixes when the model is compiled | `main.py`, `training/trainer.py` | Verified by reading | `a452786`: `models.checkpoint.load_model_weights` takes either form; the Trainer compiles in place, so keys carry no prefix |
| 6 | **`train` and `evaluate` commands are placeholders** | `main.py` | Verified | `fa8ce2c`, `7fc0695`: `train --config [--max-steps] [--resume]`, `evaluate --config --weights [--split]`; `check-data` added in `e9663e7`, `e0e6b59` |
| 7 | **No train/val/test split logic.** One dataset object per frames directory | `data/dataset.py` | Reported | `7ed3833`: clips discovered under `root/<game>/<clip>/Label.csv` and split by game (train game1–6, val game7, test game8–10) |
| 8a | **`torch.compile` fails on the local machine.** No C compiler is installed, so the first compiled forward pass raises `Failed to find C compiler`. The default config enables compilation | `configs/default.yaml` `compile_model: true`; fix by installing `build-essential` or setting it false | Verified on Python 3.12 and 3.14 | `fa8ce2c`: with `compile_model: true` on CUDA, the Trainer probes compilation at startup and stops with a message naming both fixes. `37892b5`: the default is now `compile_model: false` |

### 4.2 Recipe corrections agreed for the first run

| # | Change | Where | Fixed |
|---|---|---|---|
| 8 | Heatmap radius 30 → 2.5 | `configs/default.yaml`, `data/heatmap.py` | `e917bb1`; default config in `7fc0695` |
| 9 | Training stride 3 → 1 | `data/dataset.py` | `e917bb1` (stride option); default `train_stride: 1` in `7fc0695`. Validation and test use stride 3 |
| 10 | Colour jitter sampled once per window | `data/transforms.py` | `c570368` |
| 11 | Evaluation: largest-blob prediction, FP/FN reported under both conventions, tolerance space stated. Ground truth stays derived from the target heatmap, which is what TrackNetV3's own evaluation does (`test.py`, checked 2026-11-15); with a 2.5 px target the centroid is within 1 px of the label | `training/evaluate.py` | `356dc49` (largest blob; `recall` and `recall_strict`); `7fc0695` (`evaluate` reports `model_res_4px`, 4 px at 512x288, and `source_res_4px`, 4 px at source resolution = 1.6 px at 512x288 for 1280-wide video). Measured centroid error against the label is at most 0.29 px. Per-epoch validation and `best.pt` selection use `f1` at 4 px, 512x288 |

### 4.3 V5 only (deferred)

| # | Issue | Status |
|---|---|---|
| 12 | **MDD does not train from its initial state.** With alpha = 0 the gradient on alpha is exactly 0 and the attention collapses to four discrete values; beta's gradient is of order 1e7 or larger with no gradient clipping | Verified locally |
| 13 | R-STR and backbone differ from the SDK (§3.2) | Reported |

### 4.4 Real-data preflight (2026-11-15)

Machine: laptop, NVIDIA GeForce RTX 3070 Laptop GPU (8192 MiB), WSL2, Python 3.14, 16 CPU threads, 15 GB RAM. No C compiler, so every run used `compile_model: false`. GPU numbers were taken at commit `fcc6169` with nothing else using the GPU.

**Dataset as downloaded.** The official NYCU SharePoint zip (2,564,111,322 bytes) extracts to a top-level `Dataset/` folder holding `game1` … `game10`, each with `ClipN/` folders of `NNNN.jpg` frames and one `Label.csv`, plus a `Readme.docx`. 95 clips, 19,835 frames, every image 1280x720, clips of 29 to 901 frames (median 143). Every clip is numbered 0000…n-1 with no gaps and has exactly one label row per image. All 95 `Label.csv` files have the header `file name,visibility,x-coordinate,y-coordinate,status` (example row `0000.jpg,1,599,423,0`). Visibility 0 rows never carry coordinates (empty fields). All 82 visibility 3 rows do carry coordinates; the loader treats 3 as no ball (`visible_flags=(1, 2)`, as in WASB's tennis config), so those 82 frames (0.4%) are trained and scored as ball-absent. The `status` column (`0` 18,067, `1` 516, `2` 523, empty 729) is unused. One visible label lies outside the frame: game8/Clip2 `0094.jpg`, visibility 2, (282, 720), one past the bottom edge; its target is clipped at the edge.

**Splits** (`configs/default.yaml`; train is every game not named):

| Split | Clips | Frames | Windows | Stride | visibility 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|
| train (game1–6) | 56 | 12,279 | 12,167 | 1 | 407 / 11,048 / 785 / 39 |
| val (game7) | 9 | 1,881 | 624 (1,872 frames scored) | 3 | 125 / 1,634 / 85 / 37 |
| test (game8–10) | 30 | 5,675 | 1,880 (5,640 frames scored) | 3 | 197 / 4,950 / 522 / 6 |

The test games match WASB's. WASB has no validation split, so we take game7 from its training games and train on one game fewer than WASB does. Windows are not padded at clip ends, so 9 frames in val and 35 in test are not scored.

**check-data** (`--count 24`, about 9 s) reports the table above, no frame-id gaps, source size 1280x720 in every split, and one visible label outside the frame (test). Eight overlays across all three splits were inspected by eye: the ball is inside the circle in every enlarged crop. The ball is about 1–2 px wide at 512x288.

**Compile probe.** `train` with `compile_model: true` stops before creating any file with `RuntimeError: torch.compile does not work on this machine (InductorError). Install a C compiler (on Debian/Ubuntu: sudo apt install build-essential) or set compile_model: false in the config.`

**Short runs (300 steps):**

| | batch 2 | batch 8 |
|---|---|---|
| loss at step 100 / 200 / 300 | 0.0224 / 0.0007 / 0.0005 | 0.0226 / 0.0006 / 0.0005 |
| steps per second (after warm-up) | 8.8–8.9 (5.9 over the first 100) | not measured cleanly |
| peak GPU memory (`torch.cuda.max_memory_allocated`) | 1,769 MiB | 6,656 MiB |
| validation on game7 | 22.0 s | crashed: `torch.AcceleratorError: CUDA error: out of memory` in the DataLoader pin-memory thread as validation started |

Batch 8 is not usable on this GPU as configured; batch 4 was not tried. After 300 steps the model predicts no ball anywhere (val: tp 0, tn 161, fp1 0, fp2 0, fn 1,711 under both protocols). `infer` on game8/Clip1 (118 frames) wrote 118 rows. `train --resume … --max-steps 400` continued from step 300 and logged the truncated-epoch warning.

**One full epoch** of the default recipe (batch 2, lr 1e-4, radius 2.5, stride 1, bf16 autocast, seed 42): 6,083 steps, 708.3 s of training (8.6–8.7 steps/s steady), 24.8 s of validation, peak 1,769 MiB. Mean step loss over the epoch 0.0006; last logged window (steps 5,501–6,000) 0.0001. Validation on game7:

| Protocol | precision | recall | F1 | recall (strict) | F1 (strict) | tp / tn / fp1 / fp2 / fn |
|---|---|---|---|---|---|---|
| 4 px at 512x288 (TrackNetV3/V4 convention) | 0.914 | 0.622 | 0.740 | 0.590 | 0.717 | 1,009 / 154 / 88 / 7 / 614 |
| 4 px at source resolution = 1.6 px at 512x288 (WASB convention) | 0.597 | 0.518 | 0.554 | 0.385 | 0.468 | 659 / 154 / 438 / 7 / 614 |

This is a sanity check that the model learns, not a result: one epoch of thirty, validation split only.

**Estimated full run** (extrapolated from the one epoch above, not measured). 12,167 windows / batch 2 = 6,083 steps per epoch; about 12.2 minutes per epoch including validation; 30 epochs ≈ 6.1 hours on this laptop without `torch.compile`.

**Not measured:** speed with `torch.compile` (needs a C compiler); batch 4; anything on the test split (deliberately untouched).

### 4.5 Further defects found and fixed during the fixes

| Issue | Fixed |
|---|---|
| Targets were drawn at `x * scale`, but `cv2.resize` maps source pixel `x` to `(x + 0.5) * scale - 0.5`, so at 1280→512 the target sat 0.3 px right of and below the ball, and horizontal flip turned that into −0.3 px on half the samples. `source_to_model` / `model_to_source` in `data/heatmap.py` are now the only rescaling arithmetic, used by training targets, inference output and `check-data`. This departs deliberately from the published TrackNet code: TrackNetV3's `dataset.py` scales by the plain `w / WIDTH` ratio | `ac4d449` |
| `latest.pt` was written before `best_f1` was updated, so a resumed run could overwrite `best.pt` with a worse model | `984809c` |
| A fresh `train` in an experiment that already had `latest.pt` restarted from step 0 and overwrote its checkpoints. It now stops unless `--resume` is given | `c027ef9` |
| A config value that `weights_only=True` cannot restore (a YAML date, a `Path`) made every checkpoint unloadable; a train set smaller than one batch trained zero steps; `log_every <= 0` crashed. The Trainer now rejects all three at startup. Resuming a checkpoint whose epoch was cut short by `--max-steps`, or with a config that differs from the stored one, now logs a warning | `d7f8dbf` |
| A visible label at ±inf loaded; a non-integer image name raised a bare error; an empty val or test split failed with `ConcatDataset`'s assertion. All now raise clear errors, and `check-data` reports frame-id gaps and source sizes per split | `f5eb5fa` |

## 5. Decisions

**Done (2026-11-15):**

- The project moved from Python 3.12 to 3.14. All 166 tests of the time, lint and type check gave the same results on both, and uncompiled bf16 training steps ran on the GPU for V2 and V5.
- The fixes in §4.1, §4.2 and §4.5 are in, and the real-data preflight in §4.4 passed: `check-data`, `train` (including `--resume`), `evaluate` and `infer` have all run on the tennis dataset, and one full epoch trained and validated cleanly. No full run exists, and nothing has been evaluated on the test split.

**Decided (2026-11-15):**

- First run is **V2 on tennis** with the default recipe: AdamW at 1e-4, batch 2, 30 epochs, MultiStepLR at 20 and 25, radius 2.5, training stride 1.
- Split by game: train game1–6, validate on game7, test on game8–10 (§4.4).
- Report both evaluation protocols (`model_res_4px` and `source_res_4px`), each with `recall` and `recall_strict`. `best.pt` is selected by `f1` at 4 px, 512x288.
- `compile_model: false` by default until a C compiler is installed.
- The first experiment after a baseline is the **median background frame** input.
- **V5 is deferred.** If revisited, fix MDD and realign R-STR first (§4.3).

**Open:**

1. **Where training runs.** The local GPU is an RTX 3070 Laptop with 8 GB: enough for V2, limiting for batch size and multi-sport pre-training. A 30-epoch run would take about 6.1 hours here without `torch.compile`, extrapolated from one measured epoch (§4.4). Measured peak memory for V2 training at batch 2: 1,769 MiB; an earlier single-step measurement gave about 1.9 GB for V5.
2. **Commercial intent.** If the iOS product is the goal, RacketVision, BlurBall and the ETH set need licence confirmation before entering the training mix.
3. **On-device model.** The April recommendation to swap to WASB is not settled. A fair bake-off has three candidates: V2 with background frame, WASB-HRNet, and a width-scaled MBConv U-Net in the style of NYCU's Lite model.
4. **Pickleball data.** Requires our own recordings. If we annotate, label the centre of the blur streak, which BlurBall reports is worth +0.4 to +1.5 F1 across models.
5. **Batch size and learning rate.** Batch 2 ran at 8.6–8.7 steps per second through the full epoch (8.8–8.9 after warm-up in the 300-step run) and peaks at 1,769 MiB. Batch 8 peaked at 6,656 MiB in training and then ran out of GPU memory as validation started; batch 4 was not tried. Reference codebases use Adam at 1e-3 with batch 8–10 (§3.3).
6. **Mixup** is implemented (`data/transforms.py`) but not wired into training. TrackNetV3 reports +0.29 F1 from it.
7. **Median background frame.** Not in the code. It needs a 12-channel input, and `infer` and `evaluate` must first read the architecture from the checkpoint's config (limitation below).
8. **Colour jitter strength.** 0.3 brightness, contrast and saturation, always applied: stronger than WASB's (0.1 at p = 0.5). Tuning it is an experiment, not a fix.
9. **`torch.compile` speed** is unmeasured here; it needs a C compiler (`sudo apt install build-essential`).

**Known limitations left in place:**

- Up to two frames at the end of each clip are neither trained on with full context nor scored, because windows are not padded (9 frames in val, 35 in test). This matches WASB's evaluation.
- The 4 px comparison uses `<=`; WASB uses `<`. The difference only matters for a distance of exactly 4.0.
- At the source-resolution tolerance (1.6 px at 512x288), ground-truth quantisation of up to 0.29 px is a meaningful share of the budget. If results sit near the boundary, switch that protocol to label coordinates.
- Mixed resolutions inside one clip are not supported. The source size is read from the first frame.
- `infer` (and `evaluate`) assume the default V2 architecture. Checkpoints now record the config, but neither reads it yet; they must before the 12-channel background-frame experiment.
- Resume is per epoch. An interrupted epoch restarts from its beginning.

## 6. What was not verified

- Speed with `torch.compile` (no C compiler on the machine), batch 4, and anything on the test split (deliberately untouched).
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
