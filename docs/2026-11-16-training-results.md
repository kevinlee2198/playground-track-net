# Training Results — V2 on Tennis

Started 2026-11-16. A running record of full training runs on the TrackNet tennis dataset, what each one showed, and what was tried next. Model scores are on the **validation split** (game7, 9 clips, 1,872 scored frames); §4 also measures the training games, and says so where it does. The test split (game8–10) has not been touched, by a model or by any measurement here.

Protocols, as `evaluate` reports them: `model_res_4px` counts a detection within 4 px of the label at 512x288 as correct; `source_res_4px` allows 4 px in the 1280x720 source frame, which is 1.6 px at 512x288.

## 1. Runs

| Run | Change from the baseline | Best epoch | F1 `model_res_4px` | F1 `source_res_4px` |
|---|---|---|---|---|
| `tennis_v2_baseline` | none (configs/default.yaml at f759052) | 20 | 0.949 | 0.771 |
| `tennis_v2_noflip` | `horizontal_flip: false` | running; epoch 4 | 0.922 | 0.864 |
| `tennis_v2_bg` | `background_frame: true` on top of `tennis_v2_noflip` | planned | planned | planned |
| `tennis_v2_mixup` | `mixup_alpha: 0.5` on top of `tennis_v2_noflip` | planned | planned | planned |
| `tennis_v2_jitter` | `color_jitter` 0.1 at p 0.5 on top of `tennis_v2_noflip` | planned | planned | planned |

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
4. **Without the flip, the offset is gone.** After 4 epochs of `tennis_v2_noflip`, the mean horizontal error on game7 is +0.09 px (760 detections), and +1.23 with mirrored input. Its `source_res_4px` F1 at that point is already 0.864, against 0.771 for the finished baseline; `model_res_4px` is 0.922, level with the baseline's 0.924 at the same epoch. Two things this does not show: the baseline's offset at epoch 4 (no checkpoint was kept), and whether the no-flip model stays near zero through epoch 30 (pending).

The same numbers as horizontal position relative to the ball, at 512x288, taking the label to be 0.58 px right of the ball, which is game7's mean offset in measurement 3 (1.45 source px; the training games' means are 0.45–0.69 px):

| Model | Frames | Input as recorded | Input mirrored |
|---|---|---|---|
| Baseline (flip) | training games | +0.72 | −0.52 |
| Baseline (flip) | game7 | −0.50 | −0.50 |
| No flip, epoch 4 | game7 | +0.67 | +0.65 |

**What this establishes.** The labels carry a constant offset from where the ball appears. A model trained without the flip learns it as a fixed rule, "0.6 px right of the ball", and the rule carries over to a game it has not seen (third row). The vertical offset is the control: mirroring left to right does not touch it, and both models reproduce it on game7 (vertical error 0.00 and −0.08). The flip mirrors the horizontal offset in half the training windows, so the right answer becomes "right of the ball" in some windows and "left of it" in others.

**What is not explained.** On training frames the baseline gets both cases right (first row). Having seen each frame about 30 times, the simplest account is that it memorised them, not that it reads the orientation from the scene. On game7 it puts the ball 0.5 px to the left whichever way the input faces (second row). A model that simply could not tell would be expected to settle on the ball itself; why this one settles on the mirrored rule is not known.

**Caveats on measurement 3.** The blob centroid is a crude estimate of the ball's centre, and lighting could bias it. Its horizontal value agrees with what the models imply, 0.5–0.7 px in the table above. The test split was not measured.

**Two ways to fix it:**

- **Turn the flip off** (`horizontal_flip: false`). No assumption about the size of the offset; costs one augmentation. This is run `tennis_v2_noflip`; measurement 4 is its early result and §1 will carry its final numbers.
- **Keep the flip and move the labels onto the ball** by subtracting the measured offset when labels are loaded. Keeps the augmentation, and `infer` would then report the ball's visual centre. It depends on the offset estimate, departs from the published label convention, and corrects only the constant part: the drift across the frame in game4 and game6 is also mirrored by a flip.

**For the app:** a model trained without the flip reports positions in the label convention, about 1.5 source px right of and 1.4 px below the ball as it appears. An overlay should subtract that.

## 5. The median background frame

Run `tennis_v2_bg` (configs/tennis_v2_bg.yaml) is planned. It differs from `tennis_v2_noflip` by one key, `background_frame: true`, so the comparison is with that run's final numbers.

**The input.** For each clip, the per-pixel median of up to 50 evenly spaced frames, at model resolution, is appended to every window as a fourth RGB image (12 input channels). A moving ball is in few of those frames at any one place, so the median leaves it out. Players who stand still through a short clip partly remain in it: in the background image of game7 Clip8 (55 frames, 54 of them scored) both players are visible.

**Why it is next.** It is the best-supported addition in the literature ([pre-training review](2026-11-15-pre-training-review.md) §3.1). The failure analysis in §3 does not make a strong case for it on its own. About half of the baseline's failures are at rackets and players (45 of 83 wrong detections, 46 of 89 misses), and those move, so they are missing from the background just as the ball is. Static structures are a small share: 8 wrong detections on court lines. Where a static reference could plausibly help is with balls that are faint or in front of a line or the net (15 of the 89 misses). None of this is tested here.

**A check of the input, before any training.** No model involved; game1–7 only, every 5th frame of each clip, against that clip's median background.

- The camera is close to static within a clip: the mean absolute difference between a frame and its clip's background is 2.5–4.4 of 255 by game, and 6.4 in the worst clip (game2 Clip1). Differences here are at 512x288, taking the largest of the three colour channels at each pixel.
- The ball has not been absorbed into the background: at a visible ball, the largest difference from the background in the 5x5 patch at the label has a median of 83–119 of 255 by game, and is under 20 in 0–2.7% of frames by game (1.1% on game7). A racket or player next to the ball would score high too, so this does not show that the ball can be told apart from them.

The script for this check is not in the repository.

## 6. Mixup and colour-jitter strength

Runs `tennis_v2_mixup` (configs/tennis_v2_mixup.yaml) and `tennis_v2_jitter` (configs/tennis_v2_jitter.yaml) are planned and not started. Each changes one setting from `tennis_v2_noflip`, as `tennis_v2_bg` does, so all three planned runs are compared against the same run: `tennis_v2_noflip`'s final numbers.

**Mixup.** `mixup_alpha: 0.5` mixes each training batch before the forward pass; validation is not mixed. Each sample draws a weight `lam` from Beta(0.5, 0.5), keeps `max(lam, 1 - lam)`, and is blended with a partner chosen by one random permutation of the batch, frames and heatmaps alike. This is TrackNetV3's sample mixup; their ablation (Table 2) reports F1 rising from 97.44% to 97.73% when it is added, +0.29 percentage points, on their badminton shuttlecock dataset and with the background image already in the input ([pre-training review](2026-11-15-pre-training-review.md) §3.3). `tennis_v2_mixup` has no background image, so it is not the same comparison. Three things are known before training:

- With batch size 2, the permutation leaves the batch as it is half the time, so about half the batches are not mixed.
- The mixed heatmaps are soft targets: at a sample's own ball the target is `lam`, between 0.5 and 1. This loss's per-pixel optimum for a soft target is pulled towards 0.5: a target of 0.75 is best fit by a prediction of 0.60. TrackNetV3 trains with the same loss formula (`WBCELoss` in its `utils/metric.py`), so its mixup has the same property.
- The logged training loss of the mixup run is measured against the mixed (soft) targets, and on a soft target this loss's best achievable value is above zero, so `Loss/train` of the mixup run has a floor that the other runs do not have. It is not comparable with the other runs' training loss: compare runs on validation F1, which is never mixed. The size of the floor is a calculation from the loss formula; nothing has been trained. For a mixed sample with both balls visible (two 21-pixel discs at 512x288, `lam` from Beta(0.5, 0.5) kept as `max(lam, 1 - lam)`), the per-pixel optimum leaves about 2.8e-5 on that sample's mean loss. With about half the batches unmixed at batch size 2, that is about 1.4e-5 on the epoch mean, against the baseline's 2.36e-5 at epoch 30.

**Colour-jitter strength.** `tennis_v2_jitter` uses WASB's strength, 0.1 brightness, contrast and saturation applied to half the windows (`p: 0.5`), against ours: 0.3, applied to every window. Either way one draw covers every image in a window.

## 7. Small things noticed

- The training loss is logged with four decimals and reads `0.0000` from epoch 5 on (the epoch mean is 4.96e-5 at epoch 5 and 2.36e-5 at epoch 30). Fixed: it is now logged in scientific notation.
- `evaluate` builds its ground-truth position from the centroid of the target disc rather than from the label, which moves a few frames across the 1.6 px line (1,107 true positives against 1,099 when scored on label coordinates).
- Resuming a checkpoint made before `horizontal_flip` existed warns `horizontal_flip: None -> True`, although that run did flip. Fixed: a key the checkpoint predates is compared using the value the code used before the key existed.
- Resuming a checkpoint made before `background_frame` existed warns `background_frame: None -> False`, although nothing changed. Fixed in the same way.
- A test wrote to the default checkpoint directory and failed once a real run existed there (fixed with the flip switch).
