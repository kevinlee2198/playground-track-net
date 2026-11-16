# Training Results — V2 on Tennis

Started 2026-11-16. A running record of full training runs on the TrackNet tennis dataset, what each one showed, and what was tried next. All numbers are on the **validation split** (game7, 9 clips, 1,872 scored frames). The test split (game8–10) has not been touched.

Protocols, as `evaluate` reports them: `model_res_4px` counts a detection within 4 px of the label at 512x288 as correct; `source_res_4px` allows 4 px in the 1280x720 source frame, which is 1.6 px at 512x288.

## 1. Runs

| Run | Change from the baseline | Best epoch | F1 `model_res_4px` | F1 `source_res_4px` |
|---|---|---|---|---|
| `tennis_v2_baseline` | none (configs/default.yaml at f759052) | 20 | 0.949 | 0.771 |
| `tennis_v2_noflip` | `horizontal_flip: false` | running | | |

## 2. Baseline

**Recipe:** V2, AdamW at 1e-4 with weight decay 0.01, batch 2, 30 epochs, MultiStepLR at 20 and 25, radius 2.5, training stride 1, flip and colour jitter, bf16 autocast, seed 42.

**Run:** 30 epochs, 182,490 steps, 655–673 s of training and 22 s of validation per epoch, 5.6 hours of compute, peak GPU memory 1,769 MiB. The laptop hibernated from 01:49 to 09:36 during epoch 17; the process continued from the same step when it woke, and the log shows no error or warning.

**Validation, `best.pt` (epoch 20) and `latest.pt` (epoch 30):**

| Checkpoint | Protocol | Precision | Recall | F1 | TP | TN | FP wrong place | FP no ball | FN |
|---|---|---|---|---|---|---|---|---|---|
| best | `model_res_4px` | 0.951 | 0.947 | 0.949 | 1,594 | 106 | 28 | 55 | 89 |
| best | `source_res_4px` | 0.660 | 0.926 | 0.771 | 1,107 | 106 | 515 | 55 | 89 |
| latest | `model_res_4px` | 0.971 | 0.923 | 0.946 | 1,561 | 134 | 19 | 27 | 131 |
| latest | `source_res_4px` | 0.676 | 0.892 | 0.770 | 1,087 | 134 | 493 | 27 | 131 |

**Learning curve (`model_res_4px` F1 per epoch):** 0.759 after epoch 1, 0.924 by epoch 4 (with a dip to 0.730 at epoch 3), then a slow climb to 0.948 at epoch 16. The learning-rate drops at epochs 20 and 25 did not raise F1: epochs 21–30 sit at 0.938–0.946, with precision up to 0.97 and recall down to 0.92. The last ten epochs bought nothing on this metric.

## 3. Where the baseline is wrong

Measured on `best.pt` against the label coordinates (not the centroid of the target disc, so counts differ from `evaluate` by a few frames).

**Detection.**

- By the dataset's visibility value: clearly visible balls (1,627 frames) are detected 96.0% of the time; hard-to-see balls (84) 71.4%.
- 55 false alarms on frames without a labelled ball: 38 on frames marked absent and 17 on frames marked occluded, where the ball is in the picture but hidden. Their blobs are smaller (median 12 px against 18).
- 89 misses. In 83 of them the heatmap peaks between 0.1 and 0.5: the model responded but stayed under the threshold. Only one miss is within 8 px of the frame edge.
- Detection threshold: F1 is flat from 0.4 (0.950) to 0.5 (0.949); 0.3 gives 0.944 and 0.6 gives 0.942. The default 0.5 is fine.
- Per clip, F1 runs from 0.725 (Clip8, 54 frames) to 0.972 (Clip2). The 900-frame Clip4 scores 0.963.

**Localisation.** This is where the two protocols part: 98.3% of detections on a visible ball are within 4 px of the label, but only 67.8% are within 1.6 px.

| | Median | 75th pct | 90th pct | 95th pct |
|---|---|---|---|---|
| Distance from the label, px at 512x288 | 1.30 | 1.74 | 2.16 | 2.53 |

The error is not noise around the label. Predictions sit a constant **1.08 px to the left** of it (mean horizontal error −1.08, vertical 0.00), in every clip (−0.76 to −1.18) and every part of the frame. Around that offset the spread is 0.74 px horizontally and 0.80 px vertically.

Shifting every prediction 1.08 px right, after the fact, takes `source_res_4px` F1 from 0.767 to 0.906 and leaves `model_res_4px` at 0.949. That number is optimistic, because the shift was fitted on the same frames it was scored on, but it sizes the problem: most of the gap between the two protocols is this one offset.

## 4. Why predictions sit left of the labels

Three measurements:

1. **The offset exists only on the validation game.** On un-augmented training clips (every 12th window of game1–6, 1,044 detections) the mean horizontal error is +0.14 px.
2. **Mirroring the validation frames removes it.** With the input flipped left to right, and the labels flipped to match, the mean horizontal error on game7 is +0.08 px instead of −1.07. On the training games flipping changes little (+0.06).
3. **The labels are not on the centre of the ball, in any game.** Without any model: crop the source frame around each label, remove the background with a temporal median of the six neighbouring frames, and take the centroid of the moving blob. The blob is 1.1–1.7 px left of and 1.0–1.5 px above the label in every game, game7 included (−1.45, −1.18). At 512x288 that is about 0.6 px horizontally.

Reading them together: the labels carry a constant offset of about +0.6 px (at model resolution) from where the ball appears. Vertically that is harmless, because the model learns it. Horizontally, the flip augmentation mirrors the offset in half the training windows, so the correct answer is "0.6 px right of the ball" in an unflipped frame and "0.6 px left" in a flipped one. On the training games the model gets both right, so it has learned to tell the two apart from the surroundings. On a game it has never seen it cannot, and on game7 it answers as if every frame were flipped: 0.6 px left of the ball, which is 1.1–1.2 px left of the label.

What is measured and what is inferred: the three numbered results are measurements. That the model tells orientation from context is an inference that fits all three; it was not observed directly. The blob centroid is a crude estimate of the ball's centre (lighting could bias it), but the horizontal offset it gives, 0.58 px at model resolution, agrees with the 0.5–0.6 px the model's behaviour implies.

**Two ways to fix it:**

- **Turn the flip off** (`horizontal_flip: false`). No assumption about the size of the offset; costs one augmentation. Run `tennis_v2_noflip` tests this.
- **Keep the flip and move the labels onto the ball** by subtracting the measured offset when labels are loaded. Keeps the augmentation, and `infer` would then report the ball's visual centre; depends on the offset estimate, and departs from the published label convention.

## 5. Small things noticed

- The training loss is logged with four decimals and reads `0.0000` from epoch 5 on.
- `evaluate` builds its ground-truth position from the centroid of the target disc rather than from the label, which moves a few frames across the 1.6 px line (1,107 true positives against 1,099 when scored on label coordinates).
