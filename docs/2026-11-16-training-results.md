# Training Results — V2 on Tennis

Started 2026-11-16. A running record of full training runs on the TrackNet tennis dataset, what each one showed, and what was tried next. Model scores are on the **validation split** (game7, 9 clips, 1,872 scored frames); §4 also measures the training games, and says so where it does. The test split (game8–10) has not been touched, by a model or by any measurement here.

Protocols, as `evaluate` reports them: `model_res_4px` counts a detection within 4 px of the label at 512x288 as correct; `source_res_4px` allows 4 px in the 1280x720 source frame, which is 1.6 px at 512x288.

## 1. Runs

| Run | Change from the baseline | Best epoch | F1 `model_res_4px` | F1 `source_res_4px` |
|---|---|---|---|---|
| `tennis_v2_baseline` | none (configs/default.yaml at f759052; since 2026-11-18, configs/tennis_v2_baseline.yaml) | 20 | 0.949 | 0.771 |
| `tennis_v2_noflip` | `horizontal_flip: false` | 15 | 0.947 | 0.903 |
| `tennis_v2_bg` | `background_frame: true` on top of `tennis_v2_noflip` | 15 | 0.949 | 0.909 |
| `tennis_v2_mixup` | `mixup_alpha: 0.5` on top of `tennis_v2_noflip` | 15 | 0.945 | 0.898 |
| `tennis_v2_jitter` | `color_jitter` 0.1 at p 0.5 on top of `tennis_v2_noflip` | 18 | 0.941 | 0.896 |

## 2. Baseline

**Recipe:** V2, AdamW at 1e-4 with weight decay 0.01, batch 2, 30 epochs, MultiStepLR at 20 and 25, radius 2.5, training stride 1, flip and colour jitter, bf16 autocast, seed 42.

**Run:** 30 epochs, 182,490 steps, 655–673 s of training and 22 s of validation per epoch, 5.7 hours of compute, peak GPU memory 1,769 MiB. The laptop hibernated from 01:49 to 09:36 during epoch 17; the process continued from the same step when it woke, and the log shows no error or warning.

**Validation, `best.pt` (epoch 20) and `latest.pt` (epoch 30):**

| Checkpoint | Protocol | Precision | Recall | F1 | TP | TN | FP wrong place | FP no ball | FN |
|---|---|---|---|---|---|---|---|---|---|
| best | `model_res_4px` | 0.951 | 0.947 | 0.949 | 1,594 | 106 | 28 | 55 | 89 |
| best | `source_res_4px` | 0.660 | 0.926 | 0.771 | 1,107 | 106 | 515 | 55 | 89 |
| latest | `model_res_4px` | 0.971 | 0.923 | 0.946 | 1,561 | 134 | 19 | 27 | 131 |
| latest | `source_res_4px` | 0.676 | 0.892 | 0.770 | 1,087 | 134 | 493 | 27 | 131 |

**Learning curve (`model_res_4px` F1 per epoch):** 0.759 after epoch 1, 0.924 by epoch 4 (with a dip to 0.730 at epoch 3), then a slow climb to 0.948 at epoch 16. The learning-rate drops at epochs 20 and 25 did not raise F1: epochs 21–30 sit at 0.938–0.946, with precision up to 0.97 and recall down to 0.90–0.93. The last ten epochs bought nothing on this metric.

## 3. Where the baseline is wrong

Measured on `best.pt` with `scripts/analysis/val_errors.py`, against the label coordinates. `evaluate` uses the centroid of the target disc instead, so its counts differ by a few frames: at 1.6 px it reports F1 0.771 where this section has 0.767.

**Detection.**

- By the dataset's visibility value: clearly visible balls (1,627 frames) are detected 96.0% of the time; hard-to-see balls (84) 71.4%.
- 55 false alarms on frames without a labelled ball: 38 on frames marked absent and 17 on frames marked occluded, where the ball is in the picture but hidden. Their blobs are smaller (median 12 px against 18).
- 89 misses. In 83 of them the heatmap peaks between 0.1 and 0.5: the model responded but stayed under the threshold. Only one miss is within 8 px of the frame edge.
- Detection threshold: F1 is flat from 0.4 (0.950) to 0.5 (0.949); 0.3 gives 0.944 and 0.6 gives 0.942. The default 0.5 is fine.
- Per clip, F1 runs from 0.725 (Clip8, 54 frames) to 0.972 (Clip2). The 900-frame Clip4 scores 0.963.

**What the failures look like.** All 172 failing frames were cropped from the source video, with the frame before and after, and sorted by a model looking at the crops, not by a person. About four in five were clear-cut; treat the counts as approximate.

| Failure | Frames | What is there |
|---|---|---|
| Detection, label says no ball | 38 | racket or body 17, a ball held or bounced before a serve 10, court line 5, a ball in flight 3, unclear 3 |
| Detection, label says occluded | 17 | a ball held or bounced 9, racket or body 6, other 2 |
| Detection over 4 px from the label | 28 | racket or body 22, court line 3, a ball-like streak or the ball itself 2, nothing 1 |
| Miss | 89 | ball at a racket or player 46, plainly visible 13, blur streak 11, faint 8, on a line or the net 7, no ball visible at the label 3, unclear 1 |

- **Rackets and players are the main problem, in both directions.** 45 of the 83 wrong detections are on a racket, hand, shoe or shorts, most often the yellow-green strings, and 46 of the 89 misses are balls touching a racket or player.
- **The labels are inconsistent around serves.** A ball in the server's hand or on the strings is marked visible in some clips and absent or occluded in others. Roughly 30–40 of the 172 failures, counting both directions, come from this and are out of any model's reach.
- **Static structures are a small share:** 8 detections on court lines, all but one from Clip7, which is blurrier than the rest.
- **13 misses are plain balls in the open,** with heatmap peaks of 0.1–0.5.
- **No minimum blob size helps.** Dropping detections under 5 px removes 12 wrong detections and 12 correct ones (F1 0.946, from 0.949), and no other cut-off does better.

**Localisation.** This is where the two protocols part: 98.3% of detections on a visible ball are within 4 px of the label, but only 67.8% are within 1.6 px.

| | Median | 75th pct | 90th pct | 95th pct |
|---|---|---|---|---|
| Distance from the label, px at 512x288 | 1.30 | 1.74 | 2.16 | 2.53 |

The error is not noise around the label. Predictions sit a constant **1.08 px to the left** of it: over the 1,600 detections within 10 px of their label (22 others are on something else entirely), the mean horizontal error is −1.08 px and the mean vertical error 0.00. Every clip shows it (−0.76 to −1.18), and so does every quarter of the frame's width except the leftmost, where 32 detections average −0.58. Around the offset the spread is 0.74 px horizontally and 0.80 px vertically.

Shifting every prediction 1.08 px right, after the fact, takes `source_res_4px` F1 from 0.767 to 0.906 and leaves `model_res_4px` at 0.949. That number is optimistic, because the shift was fitted on the same frames it was scored on, but it sizes the problem: most of the gap between the two protocols is this one offset.

## 4. Why predictions sit left of the labels

Short answer: the horizontal-flip augmentation. Training without it removes the offset (measurement 4).

Four measurements. The first three are on the baseline's `best.pt`; 1, 2 and 4 use `scripts/analysis/prediction_offset.py` (mean of prediction minus label over detections within 10 px), and 3 uses `scripts/analysis/label_vs_ball.py`, whose docstring gives its parameters.

1. **The offset exists only on the validation game.** On un-augmented training clips (every 12th window of game1–6, 1,044 detections) the mean horizontal error is +0.14 px.
2. **Mirroring the validation frames removes it.** With the input flipped left to right, and the labels flipped to match, the mean horizontal error on game7 is +0.08 px instead of −1.07 (every 2nd window, about 800 detections). On the training games mirroring changes little (+0.06).
3. **The labels are not on the ball, in any of game1–7.** Without any model: crop the source frame around each label, remove the background with the median of the six neighbouring frames, and take the centroid of the moving blob. For a still ball at the centre of the frame, the blob is 1.3–1.7 px left of and 1.0–1.7 px above the label in each of the seven games (game7: −1.50, −1.32 source px). At 512x288 that is about 0.6 px. Two further effects sit on top of the constant and roughly average out over a clip. In five games the label runs ahead of a moving ball by 0.11–0.18 px for each px/frame of speed, which is 1–1.5 px either way at typical speeds; in game4 and game5 that slope is under 0.03. A flip does not disturb this, because it mirrors the ball's direction too. And in game4 and game6 the offset drifts across the frame: −1.3 and −2.6 px per 1,000 px horizontally, −1.4 and −1.2 vertically.
4. **Without the flip, the offset is gone.** After 4 epochs of `tennis_v2_noflip`, the mean horizontal error on game7 is +0.09 px (760 detections), and +1.23 with mirrored input. Its `source_res_4px` F1 at that point is already 0.864, against 0.771 for the finished baseline; `model_res_4px` is 0.922, level with the baseline's 0.924 at the same epoch. This does not show the baseline's offset at epoch 4 (no checkpoint was kept). The finished run stays near zero: on its `best.pt` (epoch 15) the mean horizontal error on game7 is −0.01 px (1,584 detections) and +1.14 with mirrored input; on the training games (every 12th window, 1,049 detections) it is +0.02 and +1.25. On `latest.pt` (epoch 30) it is −0.05 px on game7 (1,539 detections, from `scripts/analysis/val_errors.py`; mirrored input not measured).

The same numbers as horizontal position relative to the ball, at 512x288, taking the label to be 0.58 px right of the ball, which is game7's mean offset in measurement 3 (1.45 source px; the training games' means are 0.45–0.69 px):

| Model | Frames | Input as recorded | Input mirrored |
|---|---|---|---|
| Baseline (flip) | training games | +0.72 | −0.52 |
| Baseline (flip) | game7 | −0.50 | −0.50 |
| No flip, epoch 4 | game7 | +0.67 | +0.65 |
| No flip, best (epoch 15) | training games | +0.60 | +0.67 |
| No flip, best (epoch 15) | game7 | +0.57 | +0.56 |

**What this establishes.** The labels carry a constant offset from where the ball appears. A model trained without the flip learns it as a fixed rule, "0.6 px right of the ball", and the rule carries over to a game it has not seen (the no-flip rows on game7). The vertical offset is the control: mirroring left to right does not touch it, and the baseline and both no-flip checkpoints reproduce it on game7 (vertical error 0.00, −0.08 and +0.02). The flip mirrors the horizontal offset in half the training windows, so the right answer becomes "right of the ball" in some windows and "left of it" in others.

**What is not explained.** On training frames the baseline gets both cases right (first row). Having seen each frame about 30 times, the simplest account is that it memorised them, not that it reads the orientation from the scene. On game7 it puts the ball 0.5 px to the left whichever way the input faces (second row). A model that simply could not tell would be expected to settle on the ball itself; why this one settles on the mirrored rule is not known.

**Caveats on measurement 3.** The blob centroid is a crude estimate of the ball's centre, and lighting could bias it. Its horizontal value agrees with what the models imply, 0.5–0.7 px in the table above. The test split was not measured.

**Two ways to fix it:**

- **Turn the flip off** (`horizontal_flip: false`). No assumption about the size of the offset; costs one augmentation. This is run `tennis_v2_noflip`; its final numbers follow.
- **Keep the flip and move the labels onto the ball** by subtracting the measured offset when labels are loaded. Keeps the augmentation, and `infer` would then report the ball's visual centre. It depends on the offset estimate, departs from the published label convention, and corrects only the constant part: the drift across the frame in game4 and game6 is also mirrored by a flip.

**The no-flip run, finished.** The baseline's recipe with `horizontal_flip: false` (configs/tennis_v2_noflip.yaml), same seed. 30 epochs, 182,490 steps, 5.9 hours from start to finish (25 epochs took 653–682 s of training; five, epochs 2 and 5–8, took 701–881 s while tests and analysis shared the GPU), peak GPU memory 1,769 MiB, no warning or error in the log.

| Checkpoint | Protocol | Precision | Recall | F1 | TP | TN | FP wrong place | FP no ball | FN |
|---|---|---|---|---|---|---|---|---|---|
| best (epoch 15) | `model_res_4px` | 0.952 | 0.942 | 0.947 | 1,581 | 115 | 33 | 46 | 97 |
| best (epoch 15) | `source_res_4px` | 0.872 | 0.937 | 0.903 | 1,447 | 115 | 167 | 46 | 97 |
| latest (epoch 30) | `model_res_4px` | 0.964 | 0.913 | 0.938 | 1,535 | 133 | 29 | 28 | 147 |
| latest (epoch 30) | `source_res_4px` | 0.887 | 0.906 | 0.896 | 1,412 | 133 | 152 | 28 | 147 |

- **At the source-resolution tolerance, F1 goes from 0.771 to 0.903.** Detections more than 1.6 px from the target fall from 515 to 167. This is what shifting the baseline's predictions after the fact predicted in §3: 0.906 by `val_errors.py`'s count, against 0.907 measured the same way on this run.
- **At 4 px, 512x288, the two best checkpoints are level:** 0.947 against 0.949, which is 13 fewer correct detections in 1,872 frames. Each was picked on this same validation game. At epoch 30 the gap is wider, 0.938 against 0.946 (1,535 against 1,561 correct detections). With one seed each and one validation game, neither is a difference to read much into, in either direction.
- **Localisation** (`scripts/analysis/val_errors.py`, against the label coordinates, as in §3): the median distance from the label is 0.76 px (baseline 1.30) and the 90th percentile 1.57 (2.16); 90.4% of detections on a visible ball are within 1.6 px (67.8%). The mean error is −0.01 px horizontally and +0.02 vertically, with a spread of 0.71 and 0.76 px. Fitting a shift now gains nothing (F1 0.9073 to 0.9077 by that script's count).
- **Detection is about where the baseline's was.** Clearly visible balls are detected 95.4% of the time (96.0%) and hard-to-see balls 73.8% (71.4%). There are 46 false alarms on frames without a labelled ball (55) and 97 misses (89), 79 of them with a heatmap peak between 0.1 and 0.5. The detection threshold is again flat around the default: 0.946 at 0.4, 0.947 at 0.5, 0.946 at 0.6.
- **Learning curve (`model_res_4px` F1 per epoch):** 0.764 after epoch 1, 0.922 at epoch 4, 0.938 at epoch 8, the best 0.947 at epoch 15 and 0.946 at epoch 19. After the learning-rate drops at epochs 20 and 25, epochs 21–30 sit at 0.937–0.944, ending at 0.938 with precision 0.96 and recall 0.91. As with the baseline, the last ten epochs bought nothing on this metric.

Not done for this run: the failing frames were not cropped and sorted as in §3, so whether rackets and players are still the main failure is not known.

`tennis_v2_noflip` is the run the later experiments are compared with. `configs/default.yaml` still has the flip on, so it still reproduces the baseline; whether the default should change is open.

*Update, 2026-11-18: the default has the flip off.* `configs/default.yaml` now has `horizontal_flip: false` and `experiment_name: tennis_v2`, so it trains the no-flip recipe under a name of its own. Without the rename, resuming the baseline's `latest.pt` under the default config would pass the resume guard and, with `epochs` raised past the checkpoint's 30, continue the baseline run with the flip off, with only a warning. configs/tennis_v2_baseline.yaml is default.yaml with the flip on under the baseline's name: it has every setting of default.yaml at f759052 (the keys added since hold the values the code used before them), so it reproduces the baseline and resumes its checkpoints. configs/tennis_v2_noflip.yaml is now default.yaml under the no-flip run's name, with the flip off still set in it. No finished run's numbers change.

*Update, 2026-11-18: `infer` gets no offset correction for now.* Only the horizontal label offset has two independent estimates, the blob centroid of measurement 3 and what the no-flip models imply in the table above; the vertical one rests on the blob centroid alone. The label also leads a moving ball by a similar amount (measurement 3), which no constant correction removes, and nothing in the next phase needs 1.5 source px. When a correction is wanted, it should be one config key on the tennis data (`data.label_offset`, 0 by default), stored in the checkpoint and applied once by `infer` to the CSV and the overlay together, only for models trained without the flip.

**For the app:** a model trained without the flip reports positions in the label convention, about 1.5 source px right of and 1.4 px below the ball as it appears. `infer` does not correct for that (update above).

## 5. The median background frame

Run `tennis_v2_bg` (configs/tennis_v2_bg.yaml) differs from `tennis_v2_noflip` by one key, `background_frame: true`, so the comparison is with that run's final numbers. It finished on 2026-11-17 and scored level with `tennis_v2_noflip`: 0.949 against 0.947 at 4 px, 512x288, and 0.909 against 0.903 at the source-resolution tolerance, differences one run cannot tell from noise. The numbers are at the end of this section.

**The input.** For each clip, the per-pixel median of up to 50 evenly spaced frames, at model resolution, is appended to every window as a fourth RGB image (12 input channels). A moving ball is in few of those frames at any one place, so the median leaves it out. Players who stand still through a short clip partly remain in it: in the background image of game7 Clip8 (55 frames, 54 of them scored) both players are visible.

**Why it was tried.** It is the best-supported addition in the literature ([pre-training review](2026-11-15-pre-training-review.md) §3.1). The failure analysis in §3 does not make a strong case for it on its own. About half of the baseline's failures are at rackets and players (45 of 83 wrong detections, 46 of 89 misses), and those move, so they are missing from the background just as the ball is. Static structures are a small share: 8 wrong detections on court lines. Where a static reference could plausibly help is with balls that are faint or in front of a line or the net (15 of the 89 misses). None of this is tested here.

**A check of the input, before any training.** No model involved; game1–7 only, every 5th frame of each clip, against that clip's median background.

- The camera is close to static within a clip: the mean absolute difference between a frame and its clip's background is 2.5–4.4 of 255 by game, and 6.4 in the worst clip (game2 Clip1). Differences here are at 512x288, taking the largest of the three colour channels at each pixel.
- The ball has not been absorbed into the background: at a visible ball, the largest difference from the background in the 5x5 patch at the label has a median of 83–119 of 255 by game, and is under 20 in 0–2.7% of frames by game (1.1% on game7). A racket or player next to the ball would score high too, so this does not show that the ball can be told apart from them.

The script for this check is not in the repository.

**The run, finished.** `tennis_v2_noflip`'s recipe with `background_frame: true`, same seed, run from a worktree at commit 0dd09a4. 30 epochs, 182,490 steps, 6.0 hours from start to finish (2026-11-16 18:24 to 2026-11-17 00:22). 24 epochs took 640–677 s of training; epoch 1 took 803 s and epochs 16–20 took 709–818 s, for a reason that was not looked into. Peak GPU memory 1,777 MiB (1,769 without the background), no warning or error in the log.

| Checkpoint | Protocol | Precision | Recall | F1 | TP | TN | FP wrong place | FP no ball | FN |
|---|---|---|---|---|---|---|---|---|---|
| best (epoch 15) | `model_res_4px` | 0.948 | 0.949 | 0.949 | 1,590 | 110 | 36 | 51 | 85 |
| best (epoch 15) | `source_res_4px` | 0.876 | 0.945 | 0.909 | 1,469 | 110 | 157 | 51 | 85 |
| latest (epoch 30) | `model_res_4px` | 0.970 | 0.919 | 0.944 | 1,556 | 130 | 17 | 31 | 138 |
| latest (epoch 30) | `source_res_4px` | 0.892 | 0.912 | 0.902 | 1,431 | 130 | 142 | 31 | 138 |

- **No difference that one run can show.** Against `tennis_v2_noflip`, the best checkpoints score 0.949 against 0.947 at 4 px, 512x288 (1,590 against 1,581 correct detections in 1,872 frames) and 0.909 against 0.903 at the source-resolution tolerance (1,469 against 1,447). At epoch 30 it is 0.944 against 0.938 and 0.902 against 0.896. All four comparisons favour the background frame, by 0.001 to 0.006. Within this run alone, F1 at 4 px moves between 0.934 and 0.949 over epochs 10–20, both best checkpoints were picked on this same validation game, and there is one seed each, so this is not evidence of a gain.
- **Localisation is the same** (`scripts/analysis/val_errors.py`, against the label coordinates): the median distance from the label is 0.76 px (no-flip 0.76) and the 90th percentile 1.56 (1.57); 90.7% of detections on a visible ball are within 1.6 px (90.4%). The mean error is +0.00 px horizontally and −0.02 vertically at the best epoch, and −0.00 and −0.04 at epoch 30. With mirrored input it is +1.14 px, as for the no-flip run: this model has learned the same label convention.
- **Detection is about the same.** Clearly visible balls are detected 96.2% of the time (no-flip 95.4%) and hard-to-see balls 72.6% (73.8%). There are 85 misses (97), every one with a heatmap peak between 0.1 and 0.5, and 51 false alarms on frames without a labelled ball (46). The detection threshold is flat around the default: 0.950 at 0.4, 0.949 at 0.5, 0.949 at 0.6.
- **Learning curve (`model_res_4px` F1 per epoch):** 0.8265 after epoch 1 (no-flip 0.7639), 0.925 at epoch 4, 0.938 at epoch 8, the best 0.949 at epoch 15 and 0.948 at epoch 16. After the learning-rate drops at epochs 20 and 25, epochs 21–30 sit at 0.9436–0.9455, ending at 0.944 with precision 0.97 and recall 0.92. The training loss ends at an epoch mean of 1.97e-5.

Not done for this run: the failing frames were not cropped and sorted as in §3, so whether the 12 fewer misses are the faint balls and the balls in front of a line or the net, where a static reference was expected to help, is not known.

**What this does and does not say.** The gains reported for a background frame elsewhere (review §3.1) did not appear here at a size one run can show. Reasons that would fit, none of them tested: the no-flip run already detects 95% of clearly visible balls on this game, which leaves little room; about half of the failures found in §3 were at rackets and players, which the background does not contain; and the validation game is one broadcast match with the same camera as the training games. On this evidence there is no reason to turn `background_frame` on: it adds 28 s of start-up and makes `infer` need a single static-camera clip. It stays off by default, and the runs in §6 stay based on `tennis_v2_noflip`.

## 6. Mixup and colour-jitter strength

Runs `tennis_v2_mixup` (configs/tennis_v2_mixup.yaml) and `tennis_v2_jitter` (configs/tennis_v2_jitter.yaml) each change one setting from `tennis_v2_noflip`, as `tennis_v2_bg` does, so all three runs are compared against the same run: `tennis_v2_noflip`'s final numbers (§4). `tennis_v2_mixup` finished on 2026-11-17 and scored level with `tennis_v2_noflip`, slightly below it: 0.945 against 0.947 at 4 px, 512x288, and 0.898 against 0.903 at the source-resolution tolerance. Its numbers follow the description of mixup below. `tennis_v2_jitter` finished on 2026-11-18 and scored below `tennis_v2_noflip` by a little more: 0.941 against 0.947, and 0.896 against 0.903. That is still not a difference one run can show. Its numbers are at the end of this section.

**Mixup.** `mixup_alpha: 0.5` mixes each training batch before the forward pass; validation is not mixed. Each sample draws a weight `lam` from Beta(0.5, 0.5), keeps `max(lam, 1 - lam)`, and is blended with a partner chosen by one random permutation of the batch, frames and heatmaps alike. This is TrackNetV3's sample mixup; their ablation (Table 2) reports F1 rising from 97.44% to 97.73% when it is added, +0.29 percentage points, on their badminton shuttlecock dataset and with the background image already in the input ([pre-training review](2026-11-15-pre-training-review.md) §3.3). `tennis_v2_mixup` has no background image, so it is not the same comparison. Three things were known before training:

- With batch size 2, the permutation leaves the batch as it is half the time, so about half the batches are not mixed.
- The mixed heatmaps are soft targets: at a sample's own ball the target is `lam`, between 0.5 and 1. This loss's per-pixel optimum for a soft target is pulled towards 0.5: a target of 0.75 is best fit by a prediction of 0.60. TrackNetV3 trains with the same loss formula (`WBCELoss` in its `utils/metric.py`), so its mixup has the same property.
- The logged training loss of the mixup run is measured against the mixed (soft) targets, and on a soft target this loss's best achievable value is above zero, so `Loss/train` of the mixup run has a floor that the other runs do not have. It is not comparable with the other runs' training loss: compare runs on validation F1, which is never mixed. The size of the floor is a calculation from the loss formula, made before the run. For a mixed sample with both balls visible (two 21-pixel discs at 512x288, `lam` from Beta(0.5, 0.5) kept as `max(lam, 1 - lam)`), the per-pixel optimum leaves about 2.8e-5 on that sample's mean loss. With about half the batches unmixed at batch size 2, that is about 1.4e-5 on the epoch mean, against the baseline's 2.36e-5 at epoch 30.

**The mixup run, finished.** `tennis_v2_noflip`'s recipe with `mixup_alpha: 0.5`, same seed, run from a worktree at commit b9c7744. 30 epochs, 182,490 steps, 5.7 hours from start to finish (2026-11-17 10:04 to 15:47). Every epoch took 661–684 s of training. Peak GPU memory 1,773 MiB (1,769 without mixup), no warning or error in the log.

| Checkpoint | Protocol | Precision | Recall | F1 | TP | TN | FP wrong place | FP no ball | FN |
|---|---|---|---|---|---|---|---|---|---|
| best (epoch 15) | `model_res_4px` | 0.956 | 0.934 | 0.945 | 1,569 | 120 | 32 | 41 | 110 |
| best (epoch 15) | `source_res_4px` | 0.869 | 0.928 | 0.898 | 1,427 | 120 | 174 | 41 | 110 |
| latest (epoch 30) | `model_res_4px` | 0.983 | 0.893 | 0.936 | 1,520 | 143 | 9 | 18 | 182 |
| latest (epoch 30) | `source_res_4px` | 0.906 | 0.885 | 0.895 | 1,401 | 143 | 128 | 18 | 182 |

- **No difference that one run can show, and what there is goes against mixup.** Against `tennis_v2_noflip`, the best checkpoints score 0.945 against 0.947 at 4 px, 512x288 (1,569 against 1,581 correct detections in 1,872 frames) and 0.898 against 0.903 at the source-resolution tolerance (1,427 against 1,447). At epoch 30 it is 0.936 against 0.938 and 0.895 against 0.896. All four comparisons are lower with mixup, by 0.001 to 0.005. Within this run alone, F1 at 4 px moves between 0.923 and 0.945 over epochs 10–20, both best checkpoints were picked on this same validation game, there is one seed each, and after the first epoch the two runs do not see the training windows in the same order. This is not evidence of a loss, and it is not the gain TrackNetV3 reports.
- **One clip carries the gap in misses.** At the best epoch the mixup model misses 40 balls in game7 Clip7 where the no-flip model misses 13; over the other eight clips it misses 70 against 84. At epoch 30 it is 60 against 28 in Clip7 and 122 against 119 elsewhere. Why Clip7 differs was not looked into.
- **Localisation is about the same** (`scripts/analysis/val_errors.py`, against the label coordinates): at the best epoch the median distance from the label is 0.78 px (no-flip 0.76) and the 90th percentile 1.62 (1.57); 89.5% of detections on a visible ball are within 1.6 px (90.4%). At epoch 30 it is 0.76 px, 1.50 and 92.3% (no-flip 0.77, 1.58 and 90.5%). The mean error is +0.00 px horizontally and +0.06 vertically at the best epoch, and −0.02 and +0.02 at epoch 30. With mirrored input it is +1.17 px (+1.14 at epoch 30): this model has learned the same label convention.
- **Detection is about the same at the default threshold.** Clearly visible balls are detected 94.5% of the time (no-flip 95.4%) and hard-to-see balls 75.0% (73.8%). There are 110 misses (97), 95 of them with a heatmap peak between 0.1 and 0.5, and 41 false alarms on frames without a labelled ball (46). The detection threshold is flat around the default: 0.945 at 0.4, 0.945 at 0.5, 0.941 at 0.6.
- **The heatmap peaks look lower, as the soft targets would predict.** Peak values were not measured directly; the evidence is recall at a high threshold. With the threshold raised to 0.9, recall at the best epoch is 0.40 against the no-flip model's 0.58 (687 against 996 correct detections); at epoch 30 it is 0.72 against 0.78. At the default threshold the epoch-30 model is more cautious than the no-flip model at epoch 30: precision 0.98 and recall 0.89, against 0.96 and 0.91. Lowering its threshold to 0.3 gives 0.944 in place of 0.936 on this game (the no-flip model at epoch 30: 0.939 at 0.3, 0.938 at 0.5), but that threshold is chosen on the frames it is scored on.
- **Learning curve (`model_res_4px` F1 per epoch):** 0.794 after epoch 1 (no-flip 0.764), 0.919 at epoch 4, 0.939 at epoch 8, the best 0.945 at epoch 15 and 0.943 at epoch 20. After the learning-rate drops at epochs 20 and 25, epochs 21–30 sit at 0.931–0.941, ending at 0.936.
- **Training loss.** The epoch mean ends at 3.58e-5, against 1.95e-5 for `tennis_v2_noflip` (from its TensorBoard log; the text logs of that run and of the baseline predate the scientific notation), 1.97e-5 for `tennis_v2_bg` and 2.36e-5 for the baseline. The gap to `tennis_v2_noflip`, which differs only by mixup, is 1.6e-5, about the size of the floor calculated above (1.4e-5). It agrees with the calculation without isolating it: after the first epoch the two runs see the training windows in a different order.

Not done for this run: the failing frames were not cropped and sorted as in §3, and Clip7 was not looked at.

**What this does and does not say.** TrackNetV3's +0.29 percentage points did not appear: on this game the difference is −0.2 points at 4 px, 512x288, inside what one run can show. Reasons that would fit, none of them tested: at batch size 2 about half the batches are not mixed and a mixed batch blends its two windows with each other, once each way and each with its own weight, where the reference codebases train at batch 8–10 (review §3.3); TrackNetV3 measured it on badminton with the background image in the input; and soft targets would lower the heatmap peaks, which costs detections at a fixed threshold. That last one fits the epoch-30 comparison (0.944 at threshold 0.3 against the no-flip model's best of 0.939) but not the best checkpoints, where no threshold in the sweep takes the mixup model above 0.945. On this evidence there is no reason to turn mixup on. `mixup_alpha` stays 0 by default.

**Colour-jitter strength.** `tennis_v2_jitter` uses WASB's strength, 0.1 brightness, contrast and saturation applied to half the windows (`p: 0.5`), against ours: 0.3, applied to every window. Either way one draw covers every image in a window.

**The jitter run, finished.** `tennis_v2_noflip`'s recipe with `color_jitter` at 0.1 and `p: 0.5`, same seed, run from a worktree at commit b9c7744. 30 epochs, 182,490 steps, 5.7 hours from start to finish (2026-11-17 20:05 to 2026-11-18 01:48). By the log's timestamps, epochs 4–30 took 674–683 s each, validation included, and epochs 1–3 took 847, 759 and 696 s, while test-suite runs shared the GPU. The trainer's own timer reads 53–62 s more per epoch than the timestamps over epochs 1–11 and 32 s more at epoch 12 (it logs 706–880 s of training for epochs 1–11, against 654–662 s for epochs 13–30, where the two clocks agree within 3 s). Why the clocks disagree was not looked into. Peak GPU memory 1,769 MiB, no warning or error in the log.

| Checkpoint | Protocol | Precision | Recall | F1 | TP | TN | FP wrong place | FP no ball | FN |
|---|---|---|---|---|---|---|---|---|---|
| best (epoch 18) | `model_res_4px` | 0.948 | 0.935 | 0.941 | 1,568 | 109 | 34 | 52 | 109 |
| best (epoch 18) | `source_res_4px` | 0.865 | 0.929 | 0.896 | 1,431 | 109 | 171 | 52 | 109 |
| latest (epoch 30) | `model_res_4px` | 0.959 | 0.902 | 0.929 | 1,520 | 121 | 25 | 40 | 166 |
| latest (epoch 30) | `source_res_4px` | 0.878 | 0.893 | 0.886 | 1,392 | 121 | 153 | 40 | 166 |

- **Lower than the no-flip run in all four comparisons, by more than mixup was, and still not by an amount one run can show.** Against `tennis_v2_noflip`, the best checkpoints score 0.941 against 0.947 at 4 px, 512x288 (1,568 against 1,581 correct detections in 1,872 frames) and 0.896 against 0.903 at the source-resolution tolerance (1,431 against 1,447). At epoch 30 it is 0.929 against 0.938 (1,520 against 1,535) and 0.886 against 0.896 (1,392 against 1,412). The four differences are 0.006 to 0.010. Within this run alone, F1 at 4 px moves between 0.904 and 0.941 over epochs 10–20, both best checkpoints were picked on this same validation game, and there is one seed each.
- **The gap is there through most of the run, not only at the chosen epochs.** This run's per-epoch F1 at 4 px is below the no-flip run's in 24 of the 30 epochs, and in 22 of the 24 from epoch 7 on (all but 13 and 18); in epochs 1–6 it is above in four. After the learning-rate drops, epochs 21–30 sit at 0.928–0.932, against 0.937–0.944 for the no-flip run. Successive epochs of one run are not independent measurements, so this is one trajectory against another, not 24 comparisons.
- **The extra misses are spread over the clips.** At the best epoch there are 109 misses against 97. The largest differences by clip are 15 against 9 in game7 Clip2, 16 against 13 in Clip5 and 10 against 7 in Clip8. Clip7, which carried the mixup run's gap, has 13 in both.
- **Localisation is about the same** (`scripts/analysis/val_errors.py`, against the label coordinates): at the best epoch the median distance from the label is 0.77 px (no-flip 0.76) and the 90th percentile 1.61 (1.57); 89.8% of detections on a visible ball are within 1.6 px (90.4%). At epoch 30 it is 0.78 px, 1.55 and 90.6% (no-flip 0.77, 1.58 and 90.5%). The mean error is −0.03 px horizontally and −0.07 vertically at the best epoch, and −0.06 and −0.02 at epoch 30. With mirrored input it is +1.13 px (+1.10 at epoch 30): this model has learned the same label convention.
- **Detection is a little lower.** Clearly visible balls are detected 94.9% of the time (no-flip 95.4%) and hard-to-see balls 69.0% (73.8%, which is 58 against 62 of 84 frames). Of the 109 misses, 96 have a heatmap peak between 0.1 and 0.5, and there are 52 false alarms on frames without a labelled ball (46). The detection threshold is flat around the default: 0.940 at 0.4, 0.942 at 0.5, 0.937 at 0.6 (scored against the label coordinates, which is why 0.5 reads 0.942 here and 0.941 in the table). There is no consistent sign of lower heatmap peaks: with the threshold at 0.9, recall is 0.64 against the no-flip model's 0.58 at the best epoch, and 0.76 against 0.78 at epoch 30.
- **Learning curve (`model_res_4px` F1 per epoch):** 0.790 after epoch 1 (no-flip 0.764), 0.924 at epoch 4, 0.918 at epoch 8, a dip to 0.904 at epoch 16, the best 0.941 at epoch 18 and 0.938 at epoch 20. After the learning-rate drops at epochs 20 and 25 it ends at 0.929 with precision 0.96 and recall 0.90.
- **Training loss.** The epoch mean ends at 1.91e-5, against 1.95e-5 for `tennis_v2_noflip`. It is lower than the no-flip run's in 28 of the 30 epochs, never by more than about 4%. The jitter changes the inputs and not the targets, so this loss has no floor of the kind mixup's has, but it is measured on more lightly augmented inputs. A lower training loss is expected from a weaker augmentation in any case. Only the lower validation score bears on whether the stronger jitter helps, and it is small, with one run each.

Not done for this run: the failing frames were not cropped and sorted as in §3. Strength and probability were changed together, so their effects are not separated, and no run had the jitter off.

**What this does and does not say.** WASB's weaker jitter did not do better than ours on this game: it is 0.6 points lower at 4 px, 512x288, with the gap present through most of the run. That is the largest of the three differences in §5 and §6, and it is still one seed scored on one validation game, so it is not proof that 0.3 on every window is the better setting. The validation game is one broadcast match with the same camera and colours as the training games, so nothing here says how either setting does on footage that looks different, which is what colour augmentation is for. On this evidence there is no reason to change it: the `color_jitter` default stays at 0.3 with `p: 1.0`.

**After these three experiments** (background frame, mixup, jitter strength), none has beaten `tennis_v2_noflip` by an amount one run can show, and it remains the recipe the tennis results stand on. The per-epoch record points the same way as each run's chosen checkpoints: from epoch 7 on, F1 at 4 px is above the no-flip run's in 21 of 24 epochs for `tennis_v2_bg`, 7 of 24 for `tennis_v2_mixup` and 2 of 24 for `tennis_v2_jitter`. Those are single trajectories, one seed each, on one validation game. If any of the three is worth a repeat with other seeds, it is the background frame.

## 7. Small things noticed

- The training loss is logged with four decimals and reads `0.0000` from epoch 5 on (the epoch mean is 4.96e-5 at epoch 5 and 2.36e-5 at epoch 30). Fixed: it is now logged in scientific notation.
- `evaluate` builds its ground-truth position from the centroid of the target disc rather than from the label, which moves a few frames across the 1.6 px line (1,107 true positives against 1,099 when scored on label coordinates).
- Resuming a checkpoint made before `horizontal_flip` existed warns `horizontal_flip: None -> True`, although that run did flip. Fixed: a key the checkpoint predates is compared using the value the code used before the key existed.
- Resuming a checkpoint made before `background_frame` existed warns `background_frame: None -> False`, although nothing changed. Fixed in the same way.
- A test wrote to the default checkpoint directory and failed once a real run existed there (fixed with the flip switch).
