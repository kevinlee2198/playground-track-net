# Training Results — V2 on Badminton

Started 2026-11-18. The first run on the CoachAI shuttlecock dataset ([data-sources.md](data-sources.md)), with the checks made on the real data before it. Model scores are on the **validation split**: `Professional/match22`, `Professional/match23` and `Amateur/match3`, 18 rallies, 9,084 scored frames. The dataset's `Test` part is held out like tennis game8–10: its frames were extracted along with the rest, no `Test` frame has been looked at, and no model or analysis has been run on it. Only its row and frame counts are recorded.

Protocols, as `evaluate` reports them: `model_res_4px` counts a detection within 4 px of the label at 512x288 as correct; `source_res_4px` allows 4 px in the 1280x720 source frame, which is 1.6 px at 512x288.

## 1. Runs

| Run | Recipe | Best epoch | F1 `model_res_4px` | F1 `source_res_4px` |
|---|---|---|---|---|
| `badminton_v2` | configs/badminton_v2.yaml: the tennis no-flip recipe, 10 epochs | 7 | 0.951 | 0.914 |

One run, one seed. These numbers are not comparable with the tennis results (a different dataset and validation set) or with published badminton results, which are scored on the `Test` part.

## 2. The data as trained on

- **Train:** `Professional/match1`–`match21` and `Amateur/match1`–`match2`: 23 matches, 154 rallies, 69,455 label rows, 69,147 training windows (stride 1), 34,573 steps per epoch at batch 2. A tennis epoch is 6,083 steps.
- **Validation:** 3 matches, 18 rallies, 9,103 label rows, 3,028 windows (stride 3), 9,084 frames scored. Windows are never padded, so up to two frames at the end of each rally are not scored: 19 here.
- **Frames:** `main.py extract-frames --root datasets/badminton` decoded all 201 rallies in 5 min 16 s: 91,214 frames written, one per label row, and 365 surplus trailing video frames dropped, in 25 rallies. In no rally did the container's frame count differ from the decoded count. `datasets/badminton` is 21 GB with the frames.

## 3. Checks on the real data before training

All of these used `Professional` and `Amateur` only. The scripts are not in the repository.

- **The datasets build as the config says.** Train and validation built with the sizes above, and no clip from `Test` was in either.
- **No repeated frames.** In the first rally of each of the 26 matches (first 200 frames), no repeated consecutive frames were found, so the varying frame rates (25 to 30 fps) do not appear to come with duplicated frames.
- **The shuttle is at the label.** In a sheet of 12 label-centred crops from random training frames, the shuttle is at the label wherever it can be made out, usually as a motion-blurred streak.
- **Frame i matches label row i.** For each rally, a score was computed for shifts of −4 to +4 frames: how much frame i+k differs from the rally's median background in the 7x7 patch at label i, averaged over up to 40 visible labels. The right alignment should score highest at shift 0.
  - 169 of the 172 rallies score highest at shift 0. The score at shift 0 is 1.62 times the best other shift at the median.
  - The other three score highest at shift +1 by a narrow margin: `Amateur/match1/1_05_05` (88.3 at shift 0, 89.6 at +1), `Professional/match15/1_21_12` (106.4, 111.2) and `Professional/match15/2_16_12` (101.7, 105.4). None of them has surplus video frames. They were not looked at further.
  - No rally shows a clean one-frame shift, which would score low at shift 0.
  - The 24 rallies with surplus video frames outside `Test` all score highest at shift 0, which supports treating the surplus as trailing frames.
  - In nearly all of the 27 rallies whose scores were printed (the 24 with surplus frames and the three near-ties), shift +1 scores well above shift −1. A motion-blur streak that still covers the previous position would do that; this was not tested.

## 4. The run

**Recipe:** configs/badminton_v2.yaml. V2, AdamW at 1e-4 with weight decay 0.01, batch 2, 10 epochs, MultiStepLR at 6 and 8, radius 2.5, training stride 1, colour jitter, no flip, no background frame, no mixup, bf16 autocast, seed 42. Apart from the dataset, the epoch count and the milestones, this is `tennis_v2_noflip`.

**Run:** from a worktree at commit 1c5421b. 10 epochs, 345,730 steps, 10 h 45 min from start to finish (2026-11-18 01:49 to 12:34). By the log's timestamps, epochs 2–10 took 63.7–64.2 minutes each, validation included; epoch 1 took 70.1 minutes, slowed in its first 15 minutes, mostly while `tennis_v2_jitter` was being scored on the same GPU. Peak GPU memory 1,769 MiB, no warning or error in the log. The trainer's own epoch timer reads 1.5 to 5.3 minutes more per epoch than the timestamps, with the gap growing through the run; the same disagreement showed in the first epochs of `tennis_v2_jitter` and has not been looked into, so the times here are from the timestamps.

**Validation, `best.pt` (epoch 7) and `latest.pt` (epoch 10):**

| Checkpoint | Protocol | Precision | Recall | F1 | TP | TN | FP wrong place | FP no ball | FN |
|---|---|---|---|---|---|---|---|---|---|
| best | `model_res_4px` | 0.952 | 0.950 | 0.951 | 7,655 | 645 | 104 | 278 | 402 |
| best | `source_res_4px` | 0.884 | 0.946 | 0.914 | 7,107 | 645 | 652 | 278 | 402 |
| latest | `model_res_4px` | 0.955 | 0.945 | 0.950 | 7,619 | 657 | 97 | 266 | 445 |
| latest | `source_res_4px` | 0.884 | 0.941 | 0.911 | 7,056 | 657 | 660 | 266 | 445 |

**By validation match, `best.pt`:**

| Match | Rallies | Frames | F1 `model_res_4px` | F1 `source_res_4px` | TP | FP wrong place | FP no ball | FN |
|---|---|---|---|---|---|---|---|---|
| `Professional/match22` | 5 | 2,997 | 0.965 | 0.941 | 2,495 | 38 | 48 | 93 |
| `Professional/match23` | 6 | 3,078 | 0.940 | 0.874 | 2,545 | 30 | 117 | 177 |
| `Amateur/match3` | 7 | 3,009 | 0.949 | 0.927 | 2,615 | 36 | 113 | 132 |

The TP, FP and FN columns are at 4 px, 512x288. Each row comes from running `evaluate` with that match as the only validation group.

- **Learning curve (`model_res_4px` F1 per epoch):** 0.923, 0.939, 0.941, 0.944, 0.945, 0.950, 0.951, 0.949, 0.949, 0.950. It reaches 0.950 at epoch 6 and stays within 0.002 of that; epoch 7 is the first epoch after the first learning-rate drop. From one epoch to the next it moves much less than the tennis runs did; a badminton epoch has 5.7 times as many steps, and the validation set is almost five times the size.
- **Training loss:** the epoch mean falls from 1.58e-4 in epoch 1 to 2.81e-5 in epoch 6, then to 2.40e-5 after the first drop, and ends at 2.25e-5.
- **Localisation.** At the best epoch, 7,107 of the 7,759 detections on a frame with a labelled shuttle are within 1.6 px of the label at 512x288 (91.6%), and 7,655 are within 4 px (98.7%).
- **False alarms.** Of the 923 scored frames without a labelled shuttle, the best model reports a shuttle in 278 (30%). Whether those are wrong detections, or shuttles that are in the picture and not labelled (in a hand before a serve, on the floor after a rally), was not looked at.
- **Match23 is the weakest of the three**, mostly at the source-resolution tolerance: 0.874 against 0.941 and 0.927. Its detections are also the furthest from the labels vertically (0.18 px above them on average, §5). Why was not looked into.

## 5. Prediction offset, and what it suggests about the flip

`scripts/analysis/prediction_offset.py` on the validation split, every window, prediction minus label in pixels at 512x288, for detections within 10 px of the label:

| Checkpoint | Input | Detections | Mean dx | Mean dy |
|---|---|---|---|---|
| best | as it is | 7,704 | −0.07 | −0.06 |
| best | mirrored | 7,568 | +0.15 | −0.03 |
| latest | as it is | 7,670 | −0.08 | −0.08 |
| latest | mirrored | 7,554 | +0.15 | −0.05 |

By match, at the best epoch with the input as it is: `Professional/match22` −0.11 and −0.05, `Professional/match23` −0.13 and −0.18, `Amateur/match3` +0.02 and +0.07.

- **No offset to speak of.** The mean error is under 0.1 px in both directions over the validation set, and at the best epoch at most 0.18 px in any one match and direction (0.21 at epoch 10).
- **Mirroring the input moves the mean by 0.22 px.** On tennis the same test moved it by 1.14 to 1.17 px in the four runs trained without the flip (mirrored minus unmirrored mean), because the tennis labels sit about 0.6 px (at 512x288) to the right of the ball and a model trained without the flip learns that convention ([tennis results](2026-11-16-training-results.md) §4–§6). Read the same way, the badminton labels would be about 0.1 px off horizontally at 512x288, which is 0.3 px in the source frame. That is an inference from the model's behaviour, not a measurement of the labels against the shuttle.
- **So the flip would probably cost little here.** `horizontal_flip: false` was chosen for this run because the offset had not been measured. If the offset is as small as this suggests, turning the flip on would not produce the 1.1 px error it produced on tennis. Whether it would help is a separate question and needs a run.

## 6. What this does and does not say

- The data loading, the frame extraction and the recipe work on the real badminton dataset: one run reaches 0.951 at 4 px, 512x288 on three held-out matches, with no tuning for badminton beyond scaling the schedule to the larger dataset.
- More epochs at the reduced learning rates added nothing: F1 reached 0.950 at epoch 6 and stayed within 0.002 of it through epochs 7–8 (learning rate 1e-5) and 9–10 (1e-6). The curve flattening after the drops is what the schedule produces anyway, so this does not show that a longer schedule with later drops would not help. That was not tried.
- It is one seed, and the validation matches come from the same dataset as the training matches, so it says nothing about other footage.
- Not done: no per-frame error analysis of the kind made for tennis (`scripts/analysis/val_errors.py` reads the tennis label format only), no look at the failing frames, no measurement of the labels against the shuttle, and no look at the three rallies whose alignment check was a near-tie.
