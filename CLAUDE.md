# CLAUDE.md

## Project
TrackNet V2 ball tracking for tennis, badminton, pickleball. PyTorch U-Net heatmap detector.
Design spec: docs/superpowers/specs/2026-03-21-tracknet-ball-tracking-design.md
Preflight numbers and open decisions: docs/2026-11-15-pre-training-review.md
Training runs, results and error analysis: docs/2026-11-16-training-results.md
Datasets (links, licences, verification log): docs/data-sources.md

## Commands
- `uv sync` — install deps
- `uv run pytest tests/ -q -n 4` — run all tests (273 tests, 65–80 s; serial `-v` takes about as long)
- `uv run ruff format .` — format code
- `uv run ruff check --fix .` — lint and auto-fix
- `uv run ty check` — type check (12 pre-existing diagnostics as of 2026-11-15)
- `uv run pytest tests/test_models.py -v` — model tests only
- `uv run pytest tests/test_data.py -v` — data tests only
- `uv run pytest tests/test_training.py -v` — training tests only
- `uv run pytest tests/test_inference.py -v` — inference tests only
- `uv run pytest tests/test_mdd.py -v` — MDD module tests only
- `uv run pytest tests/test_pipeline.py -v` — end-to-end train / evaluate / check-data / infer tests only
- `uv run python main.py check-data --config configs/default.yaml --out <dir> [--count 12]` — log per-split clip, frame and visibility counts; write label-overlay PNGs to `<dir>` (keep it outside the repo)
- `uv run python main.py train --config configs/default.yaml [--max-steps N] [--resume checkpoints/<experiment_name>/latest.pt]` — train on the train split, validate on val each epoch; writes `checkpoints/<experiment_name>/latest.pt` every epoch, `best.pt` only when validation `f1` beats the best so far (starts at 0, so a run stuck at F1 0 has none), and TensorBoard logs in `runs/<experiment_name>/`. `--max-steps` is a total that counts steps restored by `--resume`, so N must exceed the checkpoint's step
- `uv run python main.py evaluate --config configs/default.yaml --weights <checkpoint.pt> --split val` — print JSON metrics under both protocols. The default split is test: always pass `--split val` until the model is final
- `uv run python main.py infer --video <video or frame dir> --model <checkpoint.pt> --output <csv> [--output-video <mp4>] [--threshold 0.5]`

## Code Style
- Formatter/linter: ruff (replaces black + flake8). Run before committing.
- Type checker: ty (Astral). Not mypy.
- GroupNorm(num_groups=8) everywhere, never BatchNorm (batch size 2 is too small)
- Use `torchvision.transforms.v2` imports, not v1
- Use `torch.amp.autocast(device_type, dtype=torch.bfloat16)`, not deprecated `torch.cuda.amp`
- No filterpy — custom NumPy Kalman filter instead (filterpy dead since 2018)
- scipy for connected components (ndimage.label) and spline interpolation

## Architecture
- models/ — U-Net backbone, TrackNet wrapper, WBCE loss, MDD module, R-STR head, `load_model_weights` (checkpoint.py)
- data/ — label loader (labels.py: our CSV layout and the tennis `Label.csv`), clip discovery and split by game (sources.py), dataset, heatmap generation and coordinate mapping, transforms (flip, jitter, mixup)
- training/ — trainer loop, evaluation metrics, run.py (builds the splits; entry points for train, evaluate, check-data)
- inference/ — postprocess, trajectory rectification, Kalman tracker, video I/O
- utils/ — visualization (ball overlay drawing for annotated video output)
- main.py — CLI: train | evaluate | check-data | infer
- TrackNet(backbone, mdd=None, rstr=None) — V2 uses backbone only; V5 adds MDD + R-STR
- tracknet_v5() factory in models/tracknet.py builds the full V5 stack

## Testing
- Tests use synthetic data only: no downloaded datasets, no real video files. A test that trains must put `checkpoint_dir` and `log_dir` under `tmp_path` (`_fast_training_config`): the default ones hold real runs. The `tennis_root` fixture in tests/conftest.py is a miniature tennis dataset (1280x720 frames, real `Label.csv` header)
- Tests must pass on CPU, but the Trainer picks CUDA when it is available, so on this machine Trainer and pipeline tests run on the GPU. Keep both working
- Model tests use small tensors (batch=1-2, 288x512) to keep them fast
- Training unit tests use stub models; tests/test_pipeline.py runs the real TrackNet at 128x72

## Status (2026-11-16)
- First full run done: `tennis_v2_baseline` (default recipe, 30 epochs, 5.6 h on this laptop). Validation (game7): F1 0.949 at 4 px, 512x288; 0.771 at 4 px of the source. Nothing has been evaluated on the test split.
- Splits (configs/default.yaml): train game1–6, val game7, test game8–10. WASB trains on game1–7, so say so beside any comparison. Leave the test split alone until a model is final.
- The baseline's predictions sit 1.1 px left of the labels on the validation game (results doc §3–§4). Being tested: `horizontal_flip: false` (configs/tennis_v2_noflip.yaml).
- Still open (review doc §5): batch size and learning rate, mixup, median background frame, jitter strength, `torch.compile` speed.
- V5 is deferred.

## Gotchas
- Model resolution is 512x288: the default `input_size` and the only resolution trained or measured. `input_size` in the config changes it for train, evaluate and check-data (pipeline tests use 128x72); `infer` always resizes to 512x288. Keep it static for torch.compile
- Windows are never padded: up to two frames at the end of each clip are neither trained on with full context nor scored (9 frames in val, 35 in test). `infer` pads its last window so every frame gets a prediction
- Map label coordinates between source and model resolution only with `source_to_model` / `model_to_source` (data/heatmap.py): `(x + 0.5) * scale - 0.5`, through pixel centres as `cv2.resize` does. This deliberately departs from TrackNetV3, whose plain `x * scale` puts the target 0.3 px off the ball at 1280→512
- Tennis visibility 3 rows carry coordinates but count as ball-absent (`visible_flags=(1, 2)`)
- Tennis labels are about 1.5 source px right of and 1.3 px below the ball in every game. With `horizontal_flip: true` the horizontal part becomes orientation-dependent and shows up as a constant 1.1 px error on unseen games (results doc §4)
- Models return sigmoid heatmaps by default; training calls `model(frames, return_logits=True)` and the loss runs in float32 on the logits. Backbone, R-STR head and TrackNet all take `return_logits`
- Two evaluation protocols: `model_res_4px` (4 px at 512x288) and `source_res_4px` (4 px in source pixels = 1.6 px at 512x288 for 1280-wide video). `evaluate` reports both; per-epoch validation and best.pt selection use `f1` at 4 px, 512x288 only
- Checkpoints load with `weights_only=True`, so the config must hold only plain YAML types (dict, list, str, int, float, bool, None); the Trainer rejects anything else
- `train` refuses to overwrite an existing run's `latest.pt` unless `--resume` is given. Resume is per epoch: an interrupted epoch restarts from its beginning, and a checkpoint whose epoch was cut short by `--max-steps` warns and skips the rest of that epoch
- `infer` and `evaluate` always build the default V2 `TrackNet()`; they do not read the config stored in the checkpoint
- Skip connections tapped BEFORE MaxPool in encoder
- Python 3.14 since 2026-11-15 (pinned in .python-version, interpreter managed by uv; system python is still 3.12). pyproject keeps requires-python >=3.12
- No C compiler (gcc) on this machine, so configs/default.yaml has `compile_model: false`. With it true on a CUDA machine, `train` stops before writing anything (on CPU the Trainer never compiles): "torch.compile does not work on this machine (...). Install a C compiler (on Debian/Ubuntu: sudo apt install build-essential) or set compile_model: false in the config."
