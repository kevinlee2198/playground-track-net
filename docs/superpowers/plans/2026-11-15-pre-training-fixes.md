# Pre-Training Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the V2 pipeline train, evaluate and infer correctly on the real TrackNet tennis dataset, and report metrics under the two evaluation protocols the tennis literature uses.

**Architecture:** Fix the defects in place and keep existing module boundaries. The model keeps returning probabilities by default and gains a `return_logits` flag that only the training loop uses; the loss is computed from logits in float32. Datasets are read in their native layout (no on-disk conversion): a label loader normalises CSV formats, the dataset rescales coordinates to 512x288, and a small source module discovers clips and splits them by game. `main.py` gains working `train`, `evaluate` and `check-data` commands that call into a new `training/run.py`.

**Tech Stack:** Python 3.14, PyTorch 2.10, torchvision 0.25 (`transforms.v2`), OpenCV, SciPy, pytest, ruff, ty, uv.

**Spec:** [docs/2026-11-15-pre-training-review.md](../../2026-11-15-pre-training-review.md) §4 (issues 1–11 and 8a), with the "Revised 2026-11-15" notes in [the design spec](../specs/2026-03-21-tracknet-ball-tracking-design.md).

**Review:** revised 2026-11-15 after an independent plan review; its findings are folded in below.

## Global Constraints

- Run everything through `uv run` (project Python is 3.14; system Python is 3.12).
- Before each commit: `uv run ruff format .`, `uv run ruff check --fix .`, `uv run pytest tests/ -q -n 4` all clean. `uv run ty check` must not exceed its current 18 diagnostics.
- No new dependencies.
- Tests use synthetic data only: no downloaded datasets, no real video files. They must pass on CPU. On a machine with CUDA the `Trainer` picks the GPU, so Trainer tests run there; both must work.
- GroupNorm(num_groups=8) only, never BatchNorm. `torchvision.transforms.v2`, not v1. `torch.amp.autocast(device_type, dtype=...)`, not `torch.cuda.amp`.
- Model input resolution is 512x288 (width x height). Heatmap radius is **2.5** px.
- Hyperparameters stay as they are (AdamW, lr 1e-4, batch 2, 30 epochs, MultiStepLR at 20 and 25). Task 10 measures whether a larger batch fits; changing them is a separate decision.
- V5 (`models/mdd.py`, R-STR internals) is out of scope except for the `return_logits` pass-through in Task 1. Pickleball label parsing (normalised coordinates) is out of scope.
- Write and edit files with the Write/Edit tools, never shell redirection or heredocs.
- Work on branch `fix/pre-training-blockers`. One commit per task, ending with the `Co-Authored-By` trailer from the session's attribution reminder. Do not push or open a PR.
- Per the user's workflow: each task goes implementer → code simplifier → code reviewer → fix rounds before the next task starts. Subagents run on Opus.
- Line numbers in this plan refer to the code before Task 1. After earlier tasks land, find code by the method or test name given.

## Decisions Made in This Plan

These are choices the review left open. Flag any you disagree with before execution.

| Decision | Choice | Reason |
|---|---|---|
| Tennis split | Train game1–6, validate game7, test game8–10 | The test games match WASB's. WASB has no validation split; we need one for checkpoint selection and take it from WASB's training games, so we train on one game fewer than WASB does. Reports must say so |
| Evaluation protocols | `evaluate` reports both: 4 px at 512x288 (TrackNetV3/V4) and 4 px at source resolution (WASB), which is 1.6 px at 512x288 for 1280x720 video | The two differ by 2.5x in strictness. Reporting one would make the number comparable with only half the literature |
| Which frames are scored | Each frame once. Stride-3 windows are not padded at the end of a clip, so up to 2 tail frames per clip are dropped | Matches WASB's evaluation windows; padding would score the last frame up to three times |
| Which visibility values count as visible | 1 and 2. Tennis value 3 (occluded) is treated as not visible, for both the target and the metrics | WASB's tennis config (`visible_flags: [1,2]`) |
| Ground-truth position in metrics | The centroid of the target heatmap | TrackNetV3's evaluation does the same; measured error against the label is at most 0.29 px at radius 2.5 |
| Primary F1 | `f1` uses recall = TP / (TP + FN), the TrackNetV3/V4/WASB convention. `f1_strict` also counts far detections against recall (RacketVision). Checkpoint selection uses `f1` at 4 px, 512x288 | Both are logged; selection follows the tennis literature |
| Augmentation | Horizontal flip (p = 0.5) and colour jitter at the existing strength (0.3, always applied) | Unchanged from the current code. Stronger than WASB's (0.1 at p = 0.5); tuning it is an experiment, not a fix |
| torch.compile unavailable | Fail at startup with a clear message when `compile_model` is true | A silent fallback would hide a 1.5–2x slowdown |
| Not in this plan | Mixup, median background frame, batch-size or learning-rate changes | The first experiments after a baseline |

## Review Focus

Inputs the spec implies but that would otherwise go untested. Each has a test in the task named.

1. **A clip whose frame files and label rows don't match** (missing image, extra row): dataset construction raises `ValueError` naming the clip. Task 3.
2. **A label row marked visible but with empty coordinates, or a duplicated frame id**: `load_labels` raises `ValueError` naming the file. Task 2.
3. **A ball labelled at the image edge or outside the frame after rescaling**: the target is clipped or empty, never an error. Task 3.
4. **A checkpoint written while the model is compiled, a bare state dict, or the wrong architecture**: the first two load into a plain model; the third raises instead of loading partially. Task 6.
5. **Loss becoming NaN or infinite mid-run**: training stops with `FloatingPointError` naming the step, and no checkpoint is written from that state. Task 1.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `models/losses.py` | WBCE focal loss from logits, float32 | Modify |
| `models/backbone.py`, `models/rstr.py`, `models/tracknet.py` | `return_logits` flag | Modify |
| `models/checkpoint.py` | Load weights from a trainer checkpoint or bare state dict | Create |
| `data/labels.py` | `Label`, `load_labels`: parse the internal and tennis CSV layouts | Create |
| `data/heatmap.py` | Float coordinates, radius 2.5 | Modify |
| `data/dataset.py` | Join frames to labels, rescale coordinates, `stride`, no tail padding | Modify |
| `data/transforms.py` | Colour jitter sampled once per window | Modify |
| `data/sources.py` | `Clip`, clip discovery, split by group, `build_dataset` | Create |
| `training/evaluate.py` | Largest-blob position, five outcome classes, both recall conventions | Modify |
| `training/trainer.py` | Logits + float32 loss, non-finite guard, clean and atomic checkpoints, resume state, `max_steps`, step logging, compile probe | Modify |
| `training/run.py` | `load_config`, `run_training`, `run_evaluation`, `write_label_overlays` | Create |
| `inference/video_preprocess.py` | BGR → RGB | Modify |
| `main.py` | Wire `train`, `evaluate`, `check-data`; `infer` uses `load_model_weights` | Modify |
| `configs/default.yaml` | Radius 2.5, `train_stride`, `log_every`, `data` section | Modify |
| `tests/conftest.py` | `tennis_root` fixture | Modify |
| `tests/test_pipeline.py` | End-to-end train → evaluate → infer | Create |

---

### Task 0: Branch and baseline

- [ ] **Step 1:** `git checkout -b fix/pre-training-blockers`
- [ ] **Step 2:** Commit the pending documentation as one commit (`docs: pre-training review, data sources, spec revisions, fix plan`): `README.md`, `CLAUDE.md`, everything under `docs/`.
- [ ] **Step 3:** Commit `.python-version` separately (`chore: pin Python 3.14`).
- [ ] **Step 4:** Run `uv run pytest tests/ -q -n 4`. Expected: `166 passed`.

---

### Task 1: Loss from logits in float32

**Files:**
- Modify: `models/losses.py`, `models/backbone.py:100-109`, `models/rstr.py` (`RSTRHead.forward`), `models/tracknet.py:38-49`, `training/trainer.py` (`__init__`, `_train_one_epoch`)
- Test: `tests/test_models.py`, `tests/test_training.py`

**Interfaces:**
- Produces: `WBCEFocalLoss.forward(logits: Tensor, target: Tensor) -> Tensor` — takes **pre-sigmoid logits** of any float dtype, returns a float32 scalar.
- Produces: `UNetBackbone.forward(x, return_logits: bool = False)`, `RSTRHead.forward(logits, attention, return_logits: bool = False)`, `TrackNet.forward(x, return_logits: bool = False)`. With `return_logits=False` behaviour is unchanged (probabilities). With `True` the final sigmoid is skipped.
- Produces: the training-loop contract — any model given to `Trainer` must accept `model(frames, return_logits=True)`.
- Produces: `Trainer` rejects `amp_dtype: float16` with `ValueError` (there is no GradScaler). Accepted values: `bfloat16`, `float32`.

- [ ] **Step 1: Write the failing loss tests** in `TestWBCEFocalLoss` (`tests/test_models.py:157`)

```python
def test_confident_logits_are_finite_in_bf16(self):
    loss_fn = WBCEFocalLoss()
    target = torch.ones(1, 3, 8, 8)
    good = loss_fn(torch.full((1, 3, 8, 8), 12.0, dtype=torch.bfloat16), target)
    bad = loss_fn(torch.full((1, 3, 8, 8), -12.0, dtype=torch.bfloat16), target)
    assert good.dtype == torch.float32
    assert torch.isfinite(good) and good < 1e-6
    assert torch.isfinite(bad) and bad > 10.0

def test_matches_probability_formula(self):
    torch.manual_seed(0)
    z = torch.randn(2, 3, 16, 16, dtype=torch.float64) * 3
    y = (torch.rand(2, 3, 16, 16) > 0.9).double()
    p = torch.sigmoid(z)
    expected = -((1 - p) ** 2 * y * torch.log(p) + p**2 * (1 - y) * torch.log(1 - p)).mean()
    assert torch.allclose(WBCEFocalLoss()(z, y).double(), expected, atol=1e-5)
```

- [ ] **Step 2: Write the failing model test** in `tests/test_models.py`

```python
@pytest.mark.parametrize("build", [TrackNet, tracknet_v5])
def test_return_logits_matches_probabilities(build):
    model = build().eval()
    x = torch.rand(1, 9, 288, 512)
    with torch.no_grad():
        assert torch.allclose(torch.sigmoid(model(x, return_logits=True)), model(x), atol=1e-6)
```

- [ ] **Step 3: Write the failing trainer tests** in `tests/test_training.py`

Add imports for `math`, `pytest` and `WBCEFocalLoss`. Change the stubs first: `SimpleModel.forward(self, x, return_logits=False)` returns the conv output when `return_logits` is true; `SimpleLoss` uses `binary_cross_entropy_with_logits`.

```python
def test_training_loss_finite_under_bf16_autocast():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        nn.init.constant_(model.conv.bias, 20.0)  # confident everywhere
        config = _make_test_config(tmpdir) | {"amp_dtype": "bfloat16"}
        ds = _make_synthetic_dataset()
        trainer = Trainer(model, WBCEFocalLoss(), ds, ds, config)
        assert math.isfinite(trainer._train_one_epoch())

def test_non_finite_loss_stops_training():
    class NanLoss(nn.Module):
        def forward(self, logits, target):
            return logits.sum() * float("nan")

    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_synthetic_dataset()
        trainer = Trainer(SimpleModel(), NanLoss(), ds, ds, _make_test_config(tmpdir))
        with pytest.raises(FloatingPointError, match="step"):
            trainer._train_one_epoch()

def test_float16_amp_is_rejected():
    # _make_test_config(tmpdir) | {"amp_dtype": "float16"} -> Trainer(...) raises ValueError, match="float16"
```

- [ ] **Step 4: Run and confirm failure**

Run: `uv run pytest tests/test_models.py tests/test_training.py -q`
Expected: the new tests FAIL (`nan`, `unexpected keyword argument 'return_logits'`, no `ValueError`).

- [ ] **Step 5: Implement**

- `WBCEFocalLoss.forward`: cast `logits` and `target` to float32. Use `F.logsigmoid(z)` for log p, `F.logsigmoid(-z)` for log(1 − p), `torch.sigmoid(z)` for p and `torch.sigmoid(-z)` for 1 − p. Drop the `eps` argument and the clamp.
- Backbone, R-STR head, wrapper: thread `return_logits` through as described under Interfaces. A backbone built with `apply_sigmoid=False` returns logits either way.
- `Trainer._train_one_epoch`: forward pass inside autocast with `return_logits=True`; compute the loss **outside** the autocast block on `logits.float()`. Before `backward()`, raise `FloatingPointError(f"non-finite loss at step {n}")` if the loss is not finite.
- Update what the new interfaces break:
  - The existing `TestWBCEFocalLoss` tests and the loss uses near `tests/test_models.py:393` and `:712` pass probabilities; make them pass logits.
  - `CapturingRSTR` (`tests/test_models.py:311`) and `MockRSTR` (`:637`) define `forward(self, logits, attention)`; add `return_logits=False` to both.

- [ ] **Step 6: Run the full suite** — `uv run pytest tests/ -q -n 4` → all pass.

- [ ] **Step 7: Commit** — `fix(loss): compute WBCE from logits in float32`

---

### Task 2: Label loading for real dataset layouts

**Files:**
- Create: `data/labels.py`
- Modify: `data/__init__.py` (export `Label`, `load_labels`), `tests/conftest.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class Label:
    frame: int      # frame id; for the tennis layout, int(stem of "file name")
    visible: bool
    x: float        # source-video pixels; 0.0 when not visible
    y: float

def load_labels(path: str | Path, visible_flags: tuple[int, ...] = (1, 2)) -> list[Label]:
    """Sorted by frame. Layout is detected from the header row."""
```

- Recognised headers (exact names, extra columns ignored):
  - Internal: `Frame, Visibility, X, Y`
  - Tennis: `file name, visibility, x-coordinate, y-coordinate`
- Produces the `tennis_root` fixture for later tasks.

- [ ] **Step 1: Add the fixture** to `tests/conftest.py`

```python
TENNIS_LABEL_ROWS = [  # file name, visibility, x, y, status — 1280x720 coordinates
    ("0000.jpg", 1, 640, 360, 0),
    ("0001.jpg", 1, 650, 355, 0),
    ("0002.jpg", 2, 660, 350, 0),
    ("0003.jpg", 0, "", "", ""),
    ("0004.jpg", 3, 680, 340, 0),
    ("0005.jpg", 1, 690, 335, 0),
]
```

`tennis_root(tmp_path) -> Path`: creates `game1/Clip1`, `game1/Clip2`, `game2/Clip1`, `game3/Clip1`. Each holds six 1280x720 JPEGs named as above, frame `i` filled with the uniform grey level `20 * i` so frames are distinguishable, and a `Label.csv` with header `file name,visibility,x-coordinate,y-coordinate,status` and those rows.

- [ ] **Step 2: Write the failing tests** (`class TestLoadLabels`)

```python
def test_tennis_layout(self, tennis_root):
    labels = load_labels(tennis_root / "game1" / "Clip1" / "Label.csv")
    assert [l.frame for l in labels] == [0, 1, 2, 3, 4, 5]
    assert [l.visible for l in labels] == [True, True, True, False, False, True]
    assert (labels[0].x, labels[0].y) == (640.0, 360.0)
    assert (labels[3].x, labels[3].y) == (0.0, 0.0)

def test_internal_layout(self, sample_frames_dir):
    _, csv_path = sample_frames_dir
    labels = load_labels(csv_path)
    assert [l.visible for l in labels] == [True, True, True, False, True, True, True, True, True]

def test_visible_row_without_coordinates_raises(self, tmp_path):   # match="coordinates"
def test_duplicate_frame_raises(self, tmp_path):                   # match="duplicate"
def test_unknown_header_raises(self, tmp_path):                    # match="header"
```

Each of the three error tests writes a small CSV to `tmp_path` and asserts `pytest.raises(ValueError, match=...)`; the message must include the file path.

- [ ] **Step 3: Run and confirm failure** — `uv run pytest tests/test_data.py -q` → FAIL (import error).

- [ ] **Step 4: Implement `data/labels.py`.** Empty or NaN coordinates are allowed only on rows whose visibility is not in `visible_flags`.

- [ ] **Step 5: Run** — `uv run pytest tests/test_data.py -q` → PASS.

- [ ] **Step 6: Commit** — `feat(data): label loader for internal and tennis CSV layouts`

---

### Task 3: Targets at the right place and scale

**Files:**
- Modify: `data/heatmap.py`, `data/dataset.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `Label`, `load_labels` (Task 2).
- Produces: `generate_heatmap(x: float, y: float, visibility: int, height: int = 288, width: int = 512, radius: float = 2.5) -> Tensor`
- Produces:

```python
TrackNetDataset(
    frames_dir, label_path,
    height: int = 288, width: int = 512,
    radius: float = 2.5,
    stride: int = 3,                       # 3 or 1; anything else raises ValueError
    transform: Callable | None = None,
)
dataset.source_size: tuple[int, int]       # (width, height) of the frames on disk
```

- Behaviour the tests pin:
  - Frame files are joined to labels by integer file stem (`0004.jpg` ↔ frame 4). The two sets must be equal.
  - Label coordinates are multiplied by `width / source_width` and `height / source_height`, with the source size read from the first frame.
  - Windows start at `range(0, N - 2, stride)` and are **never padded at the end of a clip**: stride 3 on 6 frames gives 2 windows, on 4 frames gives 1 (the fourth frame is dropped); stride 1 on 6 frames gives 4. A clip shorter than 3 frames gives one window padded by repeating its last frame.

- [ ] **Step 1: Write the failing tests**

```python
# TestGenerateHeatmap
def test_default_radius_is_2_5(self):
    assert generate_heatmap(x=256, y=144, visibility=1).sum() == 21

def test_float_centre_at_edge_is_clipped(self):
    hm = generate_heatmap(x=511.84, y=0.2, visibility=1)
    assert hm[0, 511] == 1.0
    assert hm[:, :500].sum() == 0

def test_label_outside_frame_gives_empty_target(self):
    assert generate_heatmap(x=-50.0, y=10.0, visibility=1).sum() == 0

# TestTrackNetDataset
def test_labels_rescaled_from_source_resolution(self, tennis_root):
    clip = tennis_root / "game1" / "Clip1"
    ds = TrackNetDataset(clip, clip / "Label.csv")
    assert ds.source_size == (1280, 720)
    _, heatmaps = ds[0]
    ys, xs = torch.where(heatmaps[0] > 0)
    assert abs(xs.float().mean().item() - 256.0) <= 0.5   # 640 * 512 / 1280
    assert abs(ys.float().mean().item() - 144.0) <= 0.5   # 360 * 288 / 720
    assert heatmaps[0].sum() == 21

def test_occluded_and_invisible_frames_have_empty_targets(self, tennis_root):
    clip = tennis_root / "game1" / "Clip1"
    _, heatmaps = TrackNetDataset(clip, clip / "Label.csv")[1]   # frames 3, 4, 5
    assert heatmaps[0].sum() == 0 and heatmaps[1].sum() == 0 and heatmaps[2].sum() == 21

def test_stride_1_window_count_and_overlap(self, tennis_root):
    clip = tennis_root / "game1" / "Clip1"
    ds = TrackNetDataset(clip, clip / "Label.csv", stride=1)
    assert len(ds) == 4
    assert torch.equal(ds[1][0][:3], ds[0][0][3:6])
    assert not torch.equal(ds[1][0][:3], ds[0][0][:3])

def test_short_clip_gives_one_padded_window(self, tmp_path):     # 2 frames -> len == 1, for stride 1 and 3
def test_invalid_stride_raises(self, tennis_root):               # stride=2 -> ValueError
def test_frame_label_mismatch_raises(self, tennis_root):
    clip = tennis_root / "game1" / "Clip1"
    (clip / "0005.jpg").unlink()
    with pytest.raises(ValueError, match="Clip1"):
        TrackNetDataset(clip, clip / "Label.csv")
```

Rewrite `TestDatasetBoundaryPadding.test_four_frames_produces_two_samples` as `test_four_frames_gives_one_window_without_tail_padding` (`len(ds) == 1`).

- [ ] **Step 2: Run and confirm failure** — `uv run pytest tests/test_data.py -q` → the new tests FAIL.

- [ ] **Step 3: Implement.** Remove `_load_labels` from `data/dataset.py` in favour of `load_labels`.

- [ ] **Step 4: Run** — `uv run pytest tests/test_data.py -q` → PASS.

- [ ] **Step 5: Commit** — `fix(data): rescale labels to model resolution, radius 2.5, stride option`

---

### Task 4: Colour jitter once per window

**Files:**
- Modify: `data/transforms.py:26-47`
- Test: `tests/test_data.py` (`TestFrameColorJitter`)

- [ ] **Step 1: Write the failing test**

```python
def test_same_jitter_applied_to_all_three_frames(self):
    torch.manual_seed(0)
    frame = torch.rand(3, 32, 64)
    out, _ = FrameColorJitter()(frame.repeat(3, 1, 1), torch.zeros(3, 32, 64))
    a, b, c = out.chunk(3, dim=0)
    assert torch.allclose(a, b) and torch.allclose(b, c)
    assert not torch.allclose(a, frame)
```

- [ ] **Step 2: Run and confirm failure** — `uv run pytest tests/test_data.py -k same_jitter -q` → FAIL.

- [ ] **Step 3: Implement.** Make a single `self.jitter` call on the frames viewed as a `(3, 3, H, W)` batch; `transforms.v2` samples its parameters once per call. Update the class docstring.

- [ ] **Step 4: Run** — `uv run pytest tests/test_data.py -q` → PASS.

- [ ] **Step 5: Commit** — `fix(data): sample colour jitter once per frame window`

---

### Task 5: Detection metrics

**Files:**
- Modify: `training/evaluate.py`, `training/__init__.py`, `training/trainer.py` (`train`, logging keys only)
- Test: `tests/test_training.py`

**Interfaces:**
- Produces: `heatmap_to_position(heatmap: Tensor, threshold: float = 0.5) -> tuple[float, float, bool]` — same signature, now the centroid of the **largest connected component**. Delegate to `inference.postprocess.heatmap_to_coordinates` with `orig_width`/`orig_height` equal to the heatmap's own size.
- Produces: `classify_frame(pred, gt, distance_threshold: float = 4.0) -> str`, one of `"tp" | "tn" | "fp1" | "fp2" | "fn"`. `fp1`: both present, distance above threshold. `fp2`: prediction with no ground truth. Replaces `compute_detection_metrics`.
- Produces: `aggregate_metrics(counts: dict[str, int]) -> dict[str, float]` with keys `precision`, `recall`, `f1`, `recall_strict`, `f1_strict`, `accuracy`.
  - `precision = tp / (tp + fp1 + fp2)`
  - `recall = tp / (tp + fn)`; `recall_strict = tp / (tp + fn + fp1)`
  - `accuracy = (tp + tn) / total`
  - Any zero denominator gives 0.0.
- Produces: `evaluate_epoch(model, dataloader, device, detection_threshold=0.5, distance_threshold=4.0)` returns those six keys plus the five counts `tp`, `tn`, `fp1`, `fp2`, `fn`. Signature unchanged; `distance_threshold` is in heatmap pixels.

- [ ] **Step 1: Write the failing tests**

```python
def test_classify_frame():
    assert classify_frame((10, 10, True), (12, 10, True), 4.0) == "tp"
    assert classify_frame((10, 10, True), (20, 10, True), 4.0) == "fp1"
    assert classify_frame((10, 10, True), (0, 0, False), 4.0) == "fp2"
    assert classify_frame((0, 0, False), (10, 10, True), 4.0) == "fn"
    assert classify_frame((0, 0, False), (0, 0, False), 4.0) == "tn"

def test_aggregate_metrics_both_recall_conventions():
    m = aggregate_metrics({"tp": 8, "tn": 2, "fp1": 1, "fp2": 1, "fn": 2})
    assert m["precision"] == pytest.approx(0.8)
    assert m["recall"] == pytest.approx(0.8)
    assert m["f1"] == pytest.approx(0.8)
    assert m["recall_strict"] == pytest.approx(8 / 11)
    assert m["f1_strict"] == pytest.approx(16 / 21)
    assert m["accuracy"] == pytest.approx(10 / 14)

def test_heatmap_to_position_uses_largest_blob():
    hm = torch.zeros(288, 512)
    hm[100:110, 200:210] = 1.0   # 100 px
    hm[20:22, 30:32] = 1.0       # 4 px of noise
    x, y, detected = heatmap_to_position(hm)
    assert detected and x == pytest.approx(204.5) and y == pytest.approx(104.5)
```

- [ ] **Step 2: Run and confirm failure** — `uv run pytest tests/test_training.py -q` → FAIL.

- [ ] **Step 3: Implement.** Rewrite the existing tests that use the old shapes: the `test_compute_detection_metrics_*` tests, `test_aggregate_metrics`, `test_aggregate_metrics_zero_division` (keep the zero-division case), and `test_evaluate_epoch_all_tp`, which asserts `metrics["fp"]` (use `fp1` and `fp2`). In `Trainer.train`, also log `Metrics/f1_strict` and `Metrics/accuracy`; best-checkpoint selection stays on `f1`.

- [ ] **Step 4: Run** — `uv run pytest tests/test_training.py -q` → PASS.

- [ ] **Step 5: Commit** — `fix(eval): largest-blob position and both recall conventions`

---

### Task 6: Inference matches training

**Files:**
- Create: `models/checkpoint.py`
- Modify: `models/__init__.py` (export), `training/trainer.py` (`__init__`, `_save_checkpoint`, `load_checkpoint`), `inference/video_preprocess.py:10-14`, `main.py:76-89`
- Test: `tests/test_inference.py`, `tests/test_training.py`

**Interfaces:**
- Produces: `load_model_weights(model: nn.Module, path: str | Path, device: torch.device) -> None` — accepts a trainer checkpoint (dict with `"model_state_dict"`) or a bare state dict; loads with `weights_only=True` and strict key matching.
- Produces: trainer checkpoints whose `model_state_dict` keys never carry the `_orig_mod.` prefix. The Trainer compiles in place with `nn.Module.compile()` (keys stay plain) and saves and loads through `getattr(self.model, "_orig_mod", self.model)` so a model compiled before being passed in is handled too.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_inference.py
def test_preprocess_frame_converts_bgr_to_rgb():
    bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
    bgr[..., 0] = 255                       # pure blue in OpenCV order
    out = preprocess_frame(bgr)
    assert out.shape == (3, 288, 512)
    assert out[2].min() == 1.0 and out[0].max() == 0.0

# tests/test_training.py
def test_checkpoint_from_compiled_model_loads_into_plain_model():
    # torch.compile wraps lazily, so this needs no compiler.
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_synthetic_dataset()
        trainer = Trainer(torch.compile(SimpleModel()), SimpleLoss(), ds, ds, _make_test_config(tmpdir))
        path = trainer._save_checkpoint("c.pt")
        ckpt = torch.load(path, weights_only=True)
        assert set(ckpt["model_state_dict"]) == set(SimpleModel().state_dict())
        load_model_weights(SimpleModel(), path, torch.device("cpu"))   # must not raise

def test_load_model_weights_accepts_checkpoint_and_bare_state_dict(tmp_path):
    # Save SimpleModel weights both ways; both load into a fresh SimpleModel with equal parameters.

def test_load_model_weights_wrong_architecture_raises(tmp_path):
    # A SimpleModel checkpoint loaded into nn.Conv2d(9, 3, 3) raises RuntimeError.
```

- [ ] **Step 2: Run and confirm failure** — `uv run pytest tests/test_inference.py tests/test_training.py -q` → the new tests FAIL.

- [ ] **Step 3: Implement.** `preprocess_frame` converts BGR → RGB before resizing. `main.run_inference` builds `TrackNet()` and calls `load_model_weights`; delete the two stale comments above the import.

- [ ] **Step 4: Run** — `uv run pytest tests/ -q -n 4` → PASS.

- [ ] **Step 5: Commit** — `fix(inference): RGB input and trainer-checkpoint loading`

---

### Task 7: Clip discovery and splits

**Files:**
- Create: `data/sources.py`
- Modify: `data/__init__.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `TrackNetDataset` (Task 3), `tennis_root` fixture (Task 2).
- Produces:

```python
@dataclass(frozen=True)
class Clip:
    frames_dir: Path
    label_path: Path
    group: str            # the unit splits are made on; for tennis, the game directory name

def discover_tennis_clips(root: str | Path) -> list[Clip]:
    """root/<game>/<clip>/Label.csv, in natural order (game2 before game10, Clip2 before Clip10)."""

def split_clips(
    clips: Sequence[Clip], val_groups: Sequence[str], test_groups: Sequence[str]
) -> dict[str, list[Clip]]:
    """Keys "train", "val", "test". Train is every group not named."""

def build_dataset(
    clips: Sequence[Clip], *, stride: int, radius: float,
    height: int = 288, width: int = 512, transform: Callable | None = None,
) -> ConcatDataset:      # of TrackNetDataset, one per clip
```

- [ ] **Step 1: Write the failing tests** (`class TestSources`)

```python
def test_discover_tennis_clips(self, tennis_root):
    clips = discover_tennis_clips(tennis_root)
    assert [(c.group, c.frames_dir.name) for c in clips] == [
        ("game1", "Clip1"), ("game1", "Clip2"), ("game2", "Clip1"), ("game3", "Clip1"),
    ]
    assert all(c.label_path.name == "Label.csv" for c in clips)

def test_natural_order(self, tmp_path):
    # game2/Clip1, game10/Clip1, game1/Clip10, game1/Clip2, each holding only a Label.csv
    # -> [("game1","Clip2"), ("game1","Clip10"), ("game2","Clip1"), ("game10","Clip1")]

def test_split_by_group_never_shares_a_game(self, tennis_root):
    s = split_clips(discover_tennis_clips(tennis_root), val_groups=["game2"], test_groups=["game3"])
    assert [len(s[k]) for k in ("train", "val", "test")] == [2, 1, 1]
    assert {c.group for c in s["train"]} == {"game1"}

def test_build_dataset_lengths(self, tennis_root):
    clips = discover_tennis_clips(tennis_root)
    assert len(build_dataset(clips, stride=1, radius=2.5)) == 16   # 4 clips x 4 windows
    assert len(build_dataset(clips, stride=3, radius=2.5)) == 8    # 4 clips x 2 windows

def test_unknown_group_raises(self, tennis_root):        # val_groups=["game9"] -> ValueError, match="game9"
def test_overlapping_groups_raises(self, tennis_root):   # game2 in both -> ValueError
def test_empty_train_raises(self, tennis_root):          # every game named -> ValueError
def test_missing_root_raises(self, tmp_path):            # FileNotFoundError
```

- [ ] **Step 2: Run and confirm failure** — `uv run pytest tests/test_data.py -k Sources -q` → FAIL.
- [ ] **Step 3: Implement `data/sources.py`.**
- [ ] **Step 4: Run** — `uv run pytest tests/test_data.py -q` → PASS.
- [ ] **Step 5: Commit** — `feat(data): tennis clip discovery and split by game`

---

### Task 8: Trainer ready for a long run

**Files:**
- Modify: `training/trainer.py`, `training/__init__.py`
- Test: `tests/test_training.py`

**Interfaces:**
- Produces: `trainer.global_step: int` — optimizer steps taken so far, saved in and restored from checkpoints.
- Produces: `config.get("max_steps")` — stop after that many optimizer steps in total; the epoch in progress still validates and checkpoints once.
- Produces: `config.get("log_every", 100)` — every that many steps, log the running mean loss (and, on CUDA, `torch.cuda.max_memory_allocated()` in MiB) and write `Loss/train_step` to TensorBoard. Log elapsed seconds per epoch.
- Produces: `ensure_compile_supported(device: torch.device) -> None` — compiles and calls a trivial function on `device`; on any failure raises `RuntimeError` whose message contains `build-essential` and `compile_model: false`, chained from the original error. Called by the Trainer before compiling the model.
- Produces: checkpoints also hold `"global_step"` and `"config"` (the config dict), and are written atomically (save to `<path>.tmp`, then `os.replace`).
- Produces: `Trainer.load_checkpoint(path)` restores `global_step` as well, so `train()` continues from the next epoch.
- Changes: the training `DataLoader` uses `drop_last=True` (a short final batch would trigger a recompile under `torch.compile`).

- [ ] **Step 1: Write the failing tests**

```python
def test_max_steps_stops_early():
    # 4 samples at batch 2 = 2 steps per epoch; epochs=5, max_steps=3 -> trainer.global_step == 3

def test_resume_continues_from_next_epoch():
    # Train epochs=1, keep latest.pt. New Trainer with epochs=2, load_checkpoint(latest.pt), train().
    # -> current_epoch == 2 and global_step == 4 (2 restored + 2 new)

def test_checkpoint_holds_config_and_step():
    ckpt = torch.load(path, weights_only=True)
    assert ckpt["global_step"] == 2 and ckpt["config"]["batch_size"] == 2

def test_checkpoint_write_leaves_no_temp_file():
    # After _save_checkpoint("latest.pt"), the checkpoint dir lists exactly ["latest.pt"]

def test_train_loader_drops_short_last_batch():
    # 5 samples at batch 2 -> len(trainer.train_loader) == 2

def test_ensure_compile_supported_reports_failure(monkeypatch):
    def broken(_fn):
        raise RuntimeError("Failed to find C compiler")
    monkeypatch.setattr("training.trainer.torch.compile", broken)
    with pytest.raises(RuntimeError, match="build-essential"):
        ensure_compile_supported(torch.device("cpu"))

def test_ensure_compile_supported_passes_when_compile_works(monkeypatch):
    monkeypatch.setattr("training.trainer.torch.compile", lambda fn: fn)
    ensure_compile_supported(torch.device("cpu"))   # no error
```

- [ ] **Step 2: Run and confirm failure** — `uv run pytest tests/test_training.py -q` → the new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — `uv run pytest tests/ -q -n 4` → PASS.
- [ ] **Step 5: Commit** — `feat(training): resumable checkpoints, step logging, compile probe`

---

### Task 9: Working `train` and `evaluate` commands

**Files:**
- Create: `training/run.py`, `tests/test_pipeline.py`
- Modify: `training/__init__.py`, `main.py`, `configs/default.yaml`, `.gitignore`, `tests/test_training.py` (`test_config_loads_from_yaml`)

**Interfaces:**
- Consumes: `discover_tennis_clips`, `split_clips`, `build_dataset`, `TrackNetDataset.source_size` (Tasks 3, 7); `load_model_weights` (Task 6); `evaluate_epoch` (Task 5); `Trainer`, `Trainer.load_checkpoint` (Task 8); `WBCEFocalLoss`, `TrackNet` (Task 1).
- Produces:

```python
def load_config(path: str | Path) -> dict
def run_training(config: dict, resume: str | Path | None = None) -> None
def run_evaluation(config: dict, weights: str | Path, split: str) -> dict   # split: "val" | "test"
```

- `run_evaluation` returns:

```python
{
    "split": "test",
    "frames": 1234,                 # frames scored
    "model_res_4px": {...},         # evaluate_epoch output at distance_threshold = 4.0
    "source_res_4px": {...},        # evaluate_epoch output at distance_threshold = 4.0 * width / source_width
}
```

  It runs `evaluate_epoch` twice (the split is small; a second pass is simpler than threading several thresholds through). Every clip in the split must share one `source_size`, and `width / source_width` must equal `height / source_height`; otherwise raise `ValueError`.
- CLI: `main.py train --config PATH [--max-steps N] [--resume CHECKPOINT]`; `main.py evaluate --config PATH --weights PATH [--split {val,test}]` (default `test`), printing the result as JSON. Both call `logging.basicConfig(level=logging.INFO)`.
- Config changes in `configs/default.yaml`:

```yaml
heatmap_radius: 2.5
train_stride: 1
log_every: 100          # steps between running-loss log lines
experiment_name: tennis_v2_baseline

data:
  format: tennis
  root: datasets/tennis
  val_groups: [game7]
  test_groups: [game8, game9, game10]
```

- Dataset settings: width and height come from `config["input_size"]`. Train: `train_stride`, `Compose([HorizontalFlip(), FrameColorJitter()])`. Validation and test: stride 3, no transform. Model `TrackNet()`, loss `WBCEFocalLoss()`. An unknown `data.format` raises `ValueError`.
- `.gitignore`: add `datasets/`, `checkpoints/`, `runs/`.

- [ ] **Step 1: Write the failing test** in `tests/test_pipeline.py`

```python
CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"

def test_train_evaluate_infer_roundtrip(tennis_root, tmp_path):
    config = load_config(CONFIG)
    config |= {
        "input_size": [128, 72],          # small for speed; the U-Net is fully convolutional
        "epochs": 1, "max_steps": 2, "batch_size": 2,
        "num_workers": 0, "pin_memory": False, "persistent_workers": False,
        "amp_dtype": "float32", "compile_model": False,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "log_dir": str(tmp_path / "runs"),
        "experiment_name": "e2e",
        "data": {"format": "tennis", "root": str(tennis_root),
                 "val_groups": ["game2"], "test_groups": ["game3"]},
    }
    run_training(config)
    ckpt = tmp_path / "checkpoints" / "e2e" / "latest.pt"
    assert ckpt.exists()

    result = run_evaluation(config, ckpt, split="test")
    assert result["frames"] == 6
    for protocol in ("model_res_4px", "source_res_4px"):
        m = result[protocol]
        assert {"precision", "recall", "f1", "f1_strict", "accuracy"} <= m.keys()
        assert m["tp"] + m["tn"] + m["fp1"] + m["fp2"] + m["fn"] == 6

    out_csv = tmp_path / "pred.csv"
    args = build_parser().parse_args(
        ["infer", "--video", str(tennis_root / "game3" / "Clip1"),
         "--model", str(ckpt), "--output", str(out_csv)]
    )
    run_inference(args)
    with open(out_csv) as f:
        assert len(list(csv.DictReader(f))) == 6
```

Update `test_config_loads_from_yaml` for `heatmap_radius == 2.5` and the new keys.

- [ ] **Step 2: Run and confirm failure** — `uv run pytest tests/test_pipeline.py tests/test_training.py -q` → FAIL.
- [ ] **Step 3: Implement** `training/run.py` and the `main.py` wiring. Remove the "not yet implemented" branches.
- [ ] **Step 4: Run** — `uv run pytest tests/ -q -n 4` → PASS. The round-trip test should take well under a minute; if it does not, report the time instead of marking it slow.
- [ ] **Step 5: Commit** — `feat(cli): working train and evaluate commands`

---

### Task 10: Real-data preflight

This task has one small code deliverable (Steps 1–2, for an implementer) and a verification run on the real dataset and GPU (Steps 3–8). **Steps 3–8 are run by the controller session, not an implementer subagent**: they need the user's go-ahead for a 2.56 GB download and possibly a `sudo` command, and they run for several minutes. Record every number they produce; the numbers go into the docs in Task 11.

**Files:**
- Modify: `training/run.py` (`write_label_overlays`), `main.py` (`check-data` command)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `write_label_overlays(config: dict, out_dir: str | Path, count: int = 12) -> list[Path]`
  - Picks `count` visible-ball frames spread evenly across **all three splits**.
  - Each PNG shows the frame resized to model resolution with a **hollow** circle (radius 6 px, 1 px line) at the rescaled label position, beside a 4x-enlarged 64x64 crop centred on the label, so a ball a few pixels wide stays visible inside the marker.
  - Logs, per split: clip count, frame count, and a count of each raw visibility value found in the label files.
- CLI: `main.py check-data --config PATH --out DIR [--count N]`.

- [ ] **Step 1: Write the failing test**

```python
def test_write_label_overlays(tennis_root, tmp_path):
    config = load_config(CONFIG)
    config["data"] = {"format": "tennis", "root": str(tennis_root),
                      "val_groups": ["game2"], "test_groups": ["game3"]}
    paths = write_label_overlays(config, tmp_path / "overlays", count=3)
    assert len(paths) == 3 and all(p.suffix == ".png" and p.exists() for p in paths)
    img = cv2.imread(str(paths[0]))
    assert img.shape[0] == 288 and img.shape[1] > 512   # frame plus the enlarged crop
    assert img.min() != img.max()   # fixture frames are uniform, so any variation is the marker
```

- [ ] **Step 2:** Run it (FAIL), implement, run again (PASS). Commit — `feat(cli): check-data command for label overlays`.

- [ ] **Step 3 (controller): Download the dataset** (2.56 GB) and unzip so that `datasets/tennis/game1/Clip1/0000.jpg` exists. Keep the cookie jar out of the repo.

```bash
curl -L -c <scratchpad>/cj -b <scratchpad>/cj -o datasets/Dataset.zip \
  "https://nycu1-my.sharepoint.com/:u:/g/personal/tik_m365_nycu_edu_tw/ETCr6-M0e1VDhGCdMbvljcsBu31AJTO5xa_1cW8pHa7niA?download=1"
```

- [ ] **Step 4 (controller): Check the labels on real frames.** Run `uv run python main.py check-data --config configs/default.yaml --out <scratchpad>/overlays --count 24`. Open at least eight overlays covering all three splits and confirm the ball is inside the circle in each enlarged crop. Record clips, frames and the visibility-value counts per split.

  **If dataset construction raises** (header names differ, frames and rows don't match, visible rows without coordinates), or the visibility values are not a subset of {0, 1, 2, 3}: stop and report exactly what the files contain. Do not loosen the validation to get past it.

- [ ] **Step 5 (controller): Check compilation.** Run a 5-step training with the default config. If it stops with the `build-essential` message, ask the user to run `sudo apt install build-essential`. If they decline, use a scratchpad copy of the config with `compile_model: false` for the rest of this task and say so in the report.

- [ ] **Step 6 (controller): Short GPU run.** `uv run python main.py train --config configs/default.yaml --max-steps 300`.
  Expected: loss finite at every log line and lower at step 300 than at step 100; `checkpoints/tennis_v2_baseline/latest.pt` written. Record steps per second, the logged peak VRAM, and how long validation on game7 took. Repeat with a scratchpad config at `batch_size: 8` and record the same, or the out-of-memory error. From steps per second, estimate the wall-clock time of the full 30-epoch run.

- [ ] **Step 7 (controller): Evaluate, infer and resume on the result.**
  - `uv run python main.py evaluate --config configs/default.yaml --weights checkpoints/tennis_v2_baseline/latest.pt --split val` prints both protocols. Values will be poor after 300 steps; passing means it runs and each protocol's five counts sum to `frames`.
  - `uv run python main.py infer --video datasets/tennis/game8/Clip1 --model checkpoints/tennis_v2_baseline/latest.pt --output <scratchpad>/pred.csv` writes one row per frame.
  - `uv run python main.py train --config configs/default.yaml --resume checkpoints/tennis_v2_baseline/latest.pt --max-steps 400` continues from step 300, not from 0.

- [ ] **Step 8 (controller):** Delete the preflight checkpoints and TensorBoard runs so the real run starts clean.

---

### Task 11: Documentation

**Files:**
- Modify: `README.md` (Status, Implementation Status, Known Issues, Quick Start, Roadmap), `CLAUDE.md` (Status section, Commands, Testing, test count), `docs/2026-11-15-pre-training-review.md` (§4 statuses, §5, §6), `docs/data-sources.md` (tennis format line), `docs/superpowers/specs/2026-03-21-tracknet-ball-tracking-design.md` (revision summary, §11 tolerance row, §13 note)

- [ ] **Step 1:** In the review doc §4, mark each issue fixed with its commit, and add the Task 10 numbers: clips, frames and visibility counts per split, steps per second and peak VRAM at batch 2 and 8, validation time, estimated run time. Replace "inferred from WASB's loader" with what the downloaded files actually contain.
- [ ] **Step 2:** In `README.md` and `CLAUDE.md`, remove the fixed items from Known Issues and Status, document the `train` (including `--resume`), `evaluate` and `check-data` commands, update the test count from the suite's own output, and correct the claim that tests always run on CPU.
- [ ] **Step 3:** In the design spec, change "The code has not caught up yet" to say which revisions are now implemented, and state the evaluation protocols as implemented: both tolerances reported, tail frames dropped, trained on game1–6.
- [ ] **Step 4:** State what remains open: batch size and learning rate (with the Task 10 numbers), mixup, the median background frame, jitter strength, and the limitations below.
- [ ] **Step 5: Commit** — `docs: record pre-training fixes and preflight results`

---

## Known Limitations Left In Place

- **Up to two frames at the end of each clip are neither trained on with full context nor scored**, because windows are not padded. This matches WASB's evaluation.
- **The 4 px comparison uses `<=`; WASB uses `<`.** The difference only matters for a distance of exactly 4.0.
- **At the source-resolution tolerance (1.6 px at 512x288), ground-truth quantisation of up to 0.29 px is a meaningful share of the budget.** If results sit near the boundary, switch that protocol to label coordinates.
- **Mixed resolutions inside one clip are not supported.** The source size is read from the first frame.
- **`infer` assumes the default V2 architecture.** Checkpoints now record the config, but `infer` does not read it yet; it must before the 12-channel background-frame experiment.
- **Resume is per epoch.** An interrupted epoch restarts from its beginning.
