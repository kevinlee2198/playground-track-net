# Augmentation Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make mixup and the colour-jitter strength config switches, so each can be trained as a one-setting change from the no-flip recipe, and close three small gaps the first runs exposed (a resume that can overwrite another run, misleading resume warnings, a loss log that reads 0.0000).

**Architecture:** Two new config keys, both defaulting to today's behaviour: `mixup_alpha` (0 = off; applied to each training batch on the device, as TrackNetV3 does) and `color_jitter` (strengths and probability, read where the training transform is built). Experiment configs inherit from `tennis_v2_noflip.yaml` and change one key each.

**Tech Stack:** Python 3.14, PyTorch, torchvision.transforms.v2, pytest; `uv run` for everything.

**Spec:** `docs/2026-11-15-pre-training-review.md` §3.3 (reference recipes) and §5 items 6 (mixup) and 8 (colour jitter); `docs/2026-11-16-training-results.md`, "Small things noticed" (§6 when this plan was written, §7 since).

## Global Constraints

- **Defaults must not change behaviour.** With `mixup_alpha: 0.0` and the default `color_jitter`, a training run gives the same batches, the same random-number stream and the same checkpoints as before for a given seed. No new random number may be drawn on a default path.
- Every key in `configs/default.yaml`, nested ones included, must be read by the code as a `"key"` string literal (`tests/test_training.py::test_every_default_config_key_is_read_by_the_code`).
- The config stored in checkpoints holds only plain YAML types.
- Project conventions in `CLAUDE.md`: ruff format and check, `ty` (the count of diagnostics must stay at 12), `torchvision.transforms.v2`, tests on synthetic data only, a test that trains uses `_fast_training_config` so it writes under `tmp_path`.
- Verification before each commit: `uv run ruff format .`, `uv run ruff check --fix .`, `uv run pytest tests/ -q -n 4`, `uv run ty check`.
- A real training run is using the GPU and writing under `checkpoints/` and `runs/`. Do not start training on real data, do not write to `checkpoints/`, `runs/`, `datasets/` or `/home/kevinlee/workspace/playground/playground-track-net-runs`. Never read `datasets/tennis/game8`, `game9` or `game10`.
- Change files only with the Write and Edit tools. No `sed -i`, no shell redirection into repo files, no `git stash`, no `git commit --amend`, no `git add -A`: stage by path.
- One commit per task unless the task says otherwise, message in the repo's style, ending with the session's `Co-Authored-By` trailer.

## Review Focus

- `train --resume A/latest.pt` with a config whose experiment directory already holds a different `latest.pt` refuses before writing anything. Resuming a run's own `latest.pt`, and resuming into an experiment name that has no checkpoint yet, both still work.
- With `mixup_alpha` above 0: validation batches are never mixed; a batch of one sample comes back unchanged; a 12-channel batch (background frame) is mixed across all channels; targets stay within [0, 1].
- `color_jitter.p: 0` never changes a frame; `p: 1` always draws exactly what it drew before this plan.
- A checkpoint written before a key existed resumes without a warning about that key when the current config holds the value the code used before.
- Negative `mixup_alpha`, or `color_jitter.p` outside [0, 1], is rejected with a clear error before any file is written.

---

### Task 1: Resume safety, quieter resume warnings, readable loss

**Files:**
- Modify: `training/run.py` (`run_training`, `_warn_config_changes`), `training/trainer.py` (the three loss log lines)
- Test: `tests/test_pipeline.py`, `tests/test_training.py`

**Interfaces:**
- Produces: `_VALUE_BEFORE_KEY: dict[str, object]` in `training/run.py`, holding, for each config key added after checkpoints already existed, the value the code used before the key: `{"horizontal_flip": True, "background_frame": False}`. Tasks 2 and 3 add their keys to it.

- [ ] **Step 1: Write failing tests.**
  - Resume guard (`tests/test_pipeline.py`, with `_fast_training_config`): train run A for 2 steps and run B (another `experiment_name`, same directories) for 2 steps; `run_training(config_B, resume=<A's latest.pt>)` raises `FileExistsError` whose message names B's `latest.pt` and the resume path, and B's `latest.pt` is byte-for-byte unchanged. `run_training(config_A | more steps, resume=<A's latest.pt>)` still works. `run_training(config_C, resume=<A's latest.pt>)`, where C's experiment directory has no checkpoint, still works and writes C's `latest.pt`.
  - Warnings (`tests/test_training.py` or wherever `_warn_config_changes` is tested): a stored config without `horizontal_flip` and `background_frame`, against a current config with `horizontal_flip: True, background_frame: False`, logs nothing; against `horizontal_flip: False` it logs `horizontal_flip: True -> False`.
  - Loss format: a logged step line and epoch line for a loss of `0.0000123` contain `1.230e-05`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.** The guard compares resolved paths (a resume path given relative or through a symlink to the experiment's own `latest.pt` is the same file) and runs before datasets or the model are built. In `_warn_config_changes`, a key absent from the stored config is read as `_VALUE_BEFORE_KEY[key]` when it is listed there, for both the list of changes and the "have no effect" list. The three log lines use `%.3e` for the loss; nothing else in them changes.
- [ ] **Step 4: Verify and commit** (`fix(training): refuse a resume that would overwrite another run; quieter resume warnings; loss in scientific notation`).

---

### Task 2: Colour-jitter strength and probability from the config

**Files:**
- Modify: `data/transforms.py` (`FrameColorJitter`), `training/run.py` (`_build_datasets`, `_VALUE_BEFORE_KEY`), `configs/default.yaml`
- Test: `tests/test_data.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `_VALUE_BEFORE_KEY` from Task 1.
- Produces: `FrameColorJitter(brightness: float = 0.3, contrast: float = 0.3, saturation: float = 0.3, p: float = 1.0)`. With probability `p` the window is jittered as today, otherwise returned unchanged. When `p` is 1.0 no random number is drawn for the decision, so the stream is the same as before. `p` outside [0, 1] raises `ValueError`.
- Config, in `configs/default.yaml` after `horizontal_flip`:

```yaml
color_jitter:           # one draw per training window, applied to every frame in it
  brightness: 0.3
  contrast: 0.3
  saturation: 0.3
  p: 1.0                # share of windows that are jittered
```

- `_build_datasets` builds the training transform's jitter from those four values. `_VALUE_BEFORE_KEY["color_jitter"]` is that default mapping.

- [ ] **Step 1: Write failing tests.** `FrameColorJitter(p=0.0)` returns frames equal to its input over 20 seeded calls; `p=1.0` gives, under the same seed, exactly what `FrameColorJitter()` gave before (compare against `T.ColorJitter` applied directly under the same seed); `p=0.5` over 200 seeded calls changes between 70 and 130 of them; `p=1.5` and `p=-0.1` raise `ValueError`. `_build_datasets` with `color_jitter` strengths of 0.1 and `p` 0.5 builds a train transform whose jitter holds those values, and with the defaults holds 0.3 and 1.0 (follow `test_train_split_flips_only_when_the_config_says_so`). The val split has no transform in either case.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify and commit** (`feat(data): colour-jitter strength and probability from the config`).

---

### Task 3: Mixup on training batches

**Files:**
- Modify: `data/transforms.py` (replace the unused `Mixup` class with `mixup_batch`), `data/__init__.py`, `training/trainer.py` (`__init__` validation, `_train_one_epoch`), `training/run.py` (`_VALUE_BEFORE_KEY`), `configs/default.yaml`
- Test: `tests/test_data.py` (replace `TestMixup`), `tests/test_training.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `_VALUE_BEFORE_KEY` from Task 1.
- Produces: `mixup_batch(frames: torch.Tensor, heatmaps: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]`, exported from `data`. TrackNetV3's sample mixup: for a batch of B samples draw `lam` from Beta(alpha, alpha) per sample and replace it by `max(lam, 1 - lam)`, so each sample keeps the larger share; draw one random permutation of the batch; return `frames * lam + frames[perm] * (1 - lam)` and the same for the heatmaps, with `lam` broadcast over channels, height and width. Random numbers come from torch's global generator (drawn on the CPU), so the Trainer's seed covers them. Outputs have the inputs' device, dtype and shape.
- Config: `mixup_alpha: 0.0` in `configs/default.yaml`, in the Training block, with the comment `# Beta(alpha, alpha) sample mixup on each training batch; 0 turns it off (TrackNetV3 uses 0.5)`. `_VALUE_BEFORE_KEY["mixup_alpha"] = 0.0`.
- Trainer: reads `config["mixup_alpha"]`; a negative value raises `ValueError` in `__init__`, before any file is written. In `_train_one_epoch`, when it is above 0, each batch is replaced by `mixup_batch(frames, heatmaps, alpha)` after it is moved to the device and before the forward pass. When it is 0, `mixup_batch` is not called. Validation is untouched.

- [ ] **Step 1: Write failing tests.**
  - `mixup_batch`, seeded: shapes, dtype and device preserved for `(4, 9, 8, 16)` frames and `(4, 3, 8, 16)` heatmaps, and for 12-channel frames; every output sample equals `lam_i * x_i + (1 - lam_i) * x_j` for one `j` and one `lam_i` in [0.5, 1], the same `lam_i` and `j` for its frames and its heatmaps (make them recoverable with constant inputs, sample `i` filled with `i` in its frames and `i * i` in its heatmaps, and find the `j` and `lam_i` that explain both); heatmaps built from values in [0, 1] stay in [0, 1]; a batch of one sample is returned unchanged; `alpha` of 0 or below raises `ValueError`.
  - Trainer (stub model, as the existing Trainer tests): with `mixup_alpha: 0.5` the model receives training batches that differ from the dataset's samples while the validation batches are exactly the dataset's; with `mixup_alpha: 0.0` a monkeypatched `mixup_batch` that raises is never called and the first training batches equal those of a Trainer built before this task under the same seed (compare two Trainers, one with the patched function); `mixup_alpha: -0.1` raises `ValueError` and leaves no file under the checkpoint or log directory.
  - Pipeline: a few steps with `mixup_alpha: 0.5` train and validate without error, with `background_frame` false and true.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.** Remove the `Mixup` class, its export and its tests; nothing else uses it.
- [ ] **Step 4: Verify and commit** (`feat(training): TrackNetV3-style sample mixup on training batches`).

---

### Task 4: Experiment configs and documentation

**Files:**
- Create: `configs/tennis_v2_mixup.yaml`, `configs/tennis_v2_jitter.yaml`
- Modify: `README.md`, `CLAUDE.md`, `docs/2026-11-15-pre-training-review.md`, `docs/2026-11-16-training-results.md`
- Test: `tests/test_training.py`

- [ ] **Step 1:** `configs/tennis_v2_mixup.yaml`: `base: tennis_v2_noflip.yaml`, `experiment_name: tennis_v2_mixup`, `mixup_alpha: 0.5`. `configs/tennis_v2_jitter.yaml`: `base: tennis_v2_noflip.yaml`, `experiment_name: tennis_v2_jitter`, `color_jitter: {brightness: 0.1, contrast: 0.1, saturation: 0.1, p: 0.5}` (WASB's strength). Tests assert each loads to the no-flip config plus exactly its overrides.
- [ ] **Step 2:** Docs. CLAUDE.md and README: the two keys, what mixup does and where (training batches only; with batch size 2 the random permutation leaves half the batches unmixed), the resume guard, the loss format, test counts measured. Review doc §5 items 6 and 8: dated one-line updates. Results doc: planned rows in §1 for both runs; remove the two "small things noticed" that Task 1 fixed, saying they are fixed.
- [ ] **Step 3: Verify and commit** (`docs: mixup and colour-jitter experiment configs and documentation`).
