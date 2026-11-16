# Median Background Frame Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a training run feed the model each clip's median background frame as a fourth input image (12 channels), and make `evaluate` and `infer` build whatever architecture a checkpoint was trained with.

**Architecture:** A config key `background_frame` (default false) switches the dataset from 9 to 12 input channels and the backbone's first convolution to match. The checkpoint already stores its config, so loading a model reads the architecture from there. Experiment configs become small files that name a base config and list only their differences.

**Tech Stack:** Python 3.14, PyTorch, OpenCV, NumPy, pytest; `uv run` for everything.

**Spec:** `docs/2026-11-15-pre-training-review.md` §3.1 ("Median background frame is the best-supported addition") and §5 item 7; `docs/2026-11-16-training-results.md` for the baseline it is compared against.

## Global Constraints

- **Defaults must not change behaviour.** With `background_frame: false` every existing code path gives the same tensors, the same model and the same checkpoints as before. Existing tests pass without edits other than those a task names.
- Project conventions in `CLAUDE.md`: ruff format and check, `ty` (the count of diagnostics must stay at 12), GroupNorm never BatchNorm, tests on synthetic data only, a test that trains uses `_fast_training_config` so it writes under `tmp_path`.
- Verification before each commit: `uv run ruff format .`, `uv run ruff check --fix .`, `uv run pytest tests/ -q -n 4`, `uv run ty check`.
- A real training run is using the GPU and writing under `checkpoints/` and `runs/`. Do not start training on real data, do not touch `checkpoints/`, `runs/`, `datasets/` or `/home/kevinlee/workspace/playground/playground-track-net-runs`.
- Change files only with the Write and Edit tools. No `sed -i`, no shell redirection into repo files, no `git stash`, no `git commit --amend`, no `git add -A`: stage by path.
- One commit per task, message in the repo's style (`feat(scope): ...`), ending with the session's `Co-Authored-By` trailer.

## Review Focus

- A checkpoint trained with a background frame, evaluated with a config that says `background_frame: false`: the checkpoint wins and a warning names both values. No shape error.
- A bare state dict (no stored config) still loads as the 9-channel V2 model.
- A clip of one or two frames with `background=True` still yields its single padded window.
- `infer` on a model trained with a background frame builds the background from the frames it was given, and still returns one result per frame.
- An experiment config whose `base` chain loops fails with a clear error instead of recursing.

---

### Task 1: Experiment configs inherit from a base config

**Files:**
- Modify: `training/run.py` (`load_config`)
- Modify: `configs/tennis_v2_noflip.yaml`
- Test: `tests/test_training.py`

**Interfaces:**
- Produces: `load_config(path: str | Path) -> dict`. Unchanged for a file without a `base` key. With a top-level `base: <path>` (relative to the directory of the file that names it), the base is loaded first, by the same rules, and the file's own keys are merged over it: a dict value merges key by key into the base's dict, any other value replaces the base's. The returned dict has no `base` key.

- [ ] **Step 1: Write failing tests** in `tests/test_training.py`, using `tmp_path` YAML files:

```python
def test_load_config_merges_over_its_base(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "epochs: 30\nlr_schedule: {milestones: [20, 25], gamma: 0.1}\n"
    )
    (tmp_path / "exp.yaml").write_text(
        "base: base.yaml\nepochs: 5\nlr_schedule: {gamma: 0.5}\n"
    )
    assert load_config(tmp_path / "exp.yaml") == {
        "epochs": 5,
        "lr_schedule": {"milestones": [20, 25], "gamma": 0.5},
    }


def test_load_config_follows_a_chain_of_bases(tmp_path): ...
    # c.yaml -> b.yaml -> a.yaml; the nearest file wins; no "base" key in the result


def test_load_config_rejects_a_base_cycle(tmp_path): ...
    # a.yaml names b.yaml and b.yaml names a.yaml:
    # pytest.raises(ValueError, match="base.*cycle")


def test_noflip_config_is_the_default_with_two_overrides():
    default = load_config(CONFIG_DIR / "default.yaml")
    noflip = load_config(CONFIG_DIR / "tennis_v2_noflip.yaml")
    assert noflip == default | {
        "horizontal_flip": False,
        "experiment_name": "tennis_v2_noflip",
    }
```

- [ ] **Step 2: Run them, confirm they fail** (`uv run pytest tests/test_training.py -q -k load_config`; the last one passes already, since the file is a full copy today).
- [ ] **Step 3: Implement the merge in `load_config`.**
- [ ] **Step 4: Rewrite `configs/tennis_v2_noflip.yaml`** as `base: default.yaml` plus its two overrides and its comment lines. `test_noflip_config_is_the_default_with_two_overrides` must still pass.
- [ ] **Step 5: Verify and commit** (`feat(training): let a config inherit from a base config`).

---

### Task 2: The dataset can append the clip's median background

**Files:**
- Create: `data/background.py`
- Modify: `data/dataset.py`, `data/sources.py` (`build_dataset`), `data/transforms.py` (`FrameColorJitter`), `data/__init__.py`, `training/run.py` (`_build_datasets`), `configs/default.yaml`
- Test: `tests/test_data.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces, in `data/background.py` and exported from `data`:
  - `background_indices(num_frames: int, max_frames: int = 50) -> list[int]`: all indices when `num_frames <= max_frames`, otherwise `max_frames` indices evenly spaced from the first frame to the last, increasing, without repeats.
  - `median_frame(frames: Iterable[np.ndarray]) -> np.ndarray`: the per-element median of equally shaped arrays, in the inputs' dtype and shape; for an integer dtype the median is rounded to the nearest integer before the cast.
- `TrackNetDataset(..., background: bool = False)`. When true, `__init__` computes `self.background`: the median, over `background_indices(self.num_frames)`, of the clip's frames as RGB `uint8` arrays of shape `(height, width, 3)` at model resolution; `__getitem__` returns frames of shape `(12, H, W)`: the three frames, then the background as float32 in `[0, 1]`. When false, `self.background` is `None` and nothing else changes.
- `build_dataset(..., background: bool = False)` passes it to every clip.
- `FrameColorJitter` accepts any multiple of three channels and applies one colour draw to all of the images.
- `_build_datasets` passes `background=config["background_frame"]` for every split. `configs/default.yaml` gains `background_frame: false`, with a one-line comment, next to `input_size`.

- [ ] **Step 1: Write failing tests.** In `tests/test_data.py`, with synthetic clips written to `tmp_path` (follow the helpers already in that file):
  - `background_indices`: `(5)` → `[0, 1, 2, 3, 4]`; `(101, 50)` → 50 increasing unique indices starting at 0 and ending at 100; `(1)` → `[0]`.
  - `median_frame`: three `uint8` arrays holding 10, 20 and 250 → all 20, dtype `uint8`; two arrays holding 1 and 2 → 2 (rounded half to even gives 2 for 1.5); float32 inputs keep float32.
  - A clip of 9 frames whose scene is a fixed random image with a bright 6x6 square at a different place in every frame: `background=True` gives windows of shape `(12, H, W)`; channels 9–11 equal the fixed scene (within 1/255) and are identical in every window; channels 0–8 equal what `background=False` returns for the same window; heatmaps are identical in both modes.
  - A two-frame clip with `background=True` yields one window of shape `(12, H, W)`.
  - `FrameColorJitter` on a `(12, H, W)` tensor built from four copies of one image returns four identical images, and still works on `(9, H, W)`.
  - `HorizontalFlip(p=1.0)` on a `(12, H, W)` tensor flips all twelve channels.
  In `tests/test_pipeline.py`: `_build_datasets` with `background_frame: true` yields 12-channel train and val samples, and 9-channel ones with `false`.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.** Store `self.background` as `uint8` (the dataset is pickled to every DataLoader worker) and convert when a window is built.
- [ ] **Step 4: Verify and commit** (`feat(data): optional median background frame as a fourth input image`).

---

### Task 3: Models are built from config; `evaluate` and `infer` follow the checkpoint

**Files:**
- Modify: `models/checkpoint.py`, `models/__init__.py`, `training/run.py` (`run_training`, `run_evaluation`), `main.py` (`run_inference`), `inference/video_preprocess.py` (`create_sliding_windows`), `scripts/analysis/val_errors.py`, `scripts/analysis/prediction_offset.py`
- Test: `tests/test_models.py`, `tests/test_inference.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes from Task 2: `background_indices`, `median_frame`, `TrackNetDataset(background=...)`, the config key `background_frame`.
- Produces, in `models/checkpoint.py` and exported from `models`:
  - `build_model(config: dict | None = None) -> TrackNet`: the V2 `TrackNet` whose backbone takes 12 input channels when `config` has a true `background_frame`, and 9 otherwise (including `config=None` and configs written before the key existed).
  - `load_model(path: str | Path, device: torch.device) -> tuple[TrackNet, dict | None]`: reads the file once with `weights_only=True`; for a trainer checkpoint builds the model from its stored config, for a bare state dict builds the default; loads the weights, moves the model to `device`, and returns it with the stored config (or `None`).
  - `load_model_weights` stays as it is for existing callers.
- `run_training` builds its model with `build_model(config)`.
- `run_evaluation` loads with `load_model`. When the checkpoint stores a config, its `background_frame` value (false if the key is absent) decides the dataset's input, and a warning is logged if the config passed in says otherwise: `"%s was trained with background_frame=%s; the config says %s. Using the checkpoint's value."`
- `create_sliding_windows(frames, window_size=3, stride=3, background: np.ndarray | None = None)`: with a background of shape `(C, H, W)`, every window has it appended after the frames.
- `run_inference` loads with `load_model`; when the stored config has a true `background_frame`, the background is `median_frame` of the preprocessed frames at `background_indices(len(frames))` and is passed to `create_sliding_windows`.
- The two analysis scripts load with `load_model` and, like `run_evaluation`, take `background_frame` from the checkpoint.

- [ ] **Step 1: Write failing tests.**
  - `tests/test_models.py`: `build_model()` and `build_model({})` accept `(1, 9, 72, 128)` and return `(1, 3, 72, 128)`; `build_model({"background_frame": True})` accepts `(1, 12, 72, 128)`. `load_model` on a saved `{"model_state_dict": ..., "config": {"background_frame": True}}` returns a 12-channel model with those weights and that config; on a bare `state_dict()` of the default model returns a 9-channel model and `None`.
  - `tests/test_inference.py`: `create_sliding_windows` with a background appends it to every window (`(12, H, W)`, last three channels equal the background) and without one is unchanged.
  - `tests/test_pipeline.py`: parametrise `test_train_evaluate_infer_roundtrip` over `background_frame` false and true (train a few steps at 128x72, evaluate on val, infer on a frames directory; one result per frame). Add: evaluating a background checkpoint with a config whose `background_frame` is false succeeds and logs the warning; evaluating a 9-channel checkpoint with a config that says true does the same in the other direction.
- [ ] **Step 2: Run them, confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Update the two analysis scripts**, then check each still prints the same numbers for the 9-channel baseline: `PYTHONPATH=. uv run python scripts/analysis/prediction_offset.py configs/default.yaml checkpoints/tennis_v2_baseline/best.pt val 2` must print `dx -1.073` and `dx +0.075`. This reads the baseline checkpoint and the validation frames; it trains nothing.
- [ ] **Step 5: Verify and commit** (`feat: build the model a checkpoint was trained with in evaluate and infer`).

---

### Task 4: Experiment config and documentation

**Files:**
- Create: `configs/tennis_v2_bg.yaml`
- Modify: `README.md`, `CLAUDE.md`, `docs/2026-11-15-pre-training-review.md`, `docs/2026-11-16-training-results.md`
- Test: `tests/test_training.py`

- [ ] **Step 1:** `configs/tennis_v2_bg.yaml`: `base: default.yaml`, `experiment_name: tennis_v2_bg`, `background_frame: true`, and `horizontal_flip` as the no-flip run's result decides. A test asserts it loads to the default plus exactly those overrides.
- [ ] **Step 2:** Docs: the limitation "`infer` and `evaluate` assume the default V2 architecture" is gone; describe `background_frame`, `base:` configs, `build_model` / `load_model`; update test counts; add the run to the results doc's table as planned.
- [ ] **Step 3: Verify and commit** (`docs: background-frame experiment config and documentation`).
