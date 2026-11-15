# CLAUDE.md

## Project
TrackNet V2 ball tracking for tennis, badminton, pickleball. PyTorch U-Net heatmap detector.
Design spec: docs/superpowers/specs/2026-03-21-tracknet-ball-tracking-design.md
Current state and open decisions: docs/2026-11-15-pre-training-review.md
Datasets (links, licences, verification log): docs/data-sources.md

## Commands
- `uv sync` — install deps
- `uv run pytest tests/ -v` — run all tests (166 tests, ~50s)
- `uv run ruff format .` — format code
- `uv run ruff check --fix .` — lint and auto-fix
- `uv run ty check` — type check (18 pre-existing diagnostics as of 2026-11-15)
- `uv run pytest tests/test_models.py -v` — model tests only
- `uv run pytest tests/test_data.py -v` — data tests only
- `uv run pytest tests/test_training.py -v` — training tests only
- `uv run pytest tests/test_inference.py -v` — inference tests only
- `uv run pytest tests/test_mdd.py -v` — MDD module tests only

## Code Style
- Formatter/linter: ruff (replaces black + flake8). Run before committing.
- Type checker: ty (Astral). Not mypy.
- GroupNorm(num_groups=8) everywhere, never BatchNorm (batch size 2 is too small)
- Use `torchvision.transforms.v2` imports, not v1
- Use `torch.amp.autocast("cuda", dtype=torch.bfloat16)`, not deprecated `torch.cuda.amp`
- No filterpy — custom NumPy Kalman filter instead (filterpy dead since 2018)
- scipy for connected components (ndimage.label) and spline interpolation

## Architecture
- models/ — U-Net backbone, TrackNet wrapper, WBCE loss, MDD module, R-STR head
- data/ — dataset, heatmap generation, transforms (flip, jitter, mixup)
- training/ — trainer loop, evaluation metrics, YAML config
- inference/ — postprocess, trajectory rectification, Kalman tracker, video I/O
- utils/ — visualization (ball overlay drawing for annotated video output)
- main.py — CLI (train | evaluate | infer); only `infer` is wired today
- TrackNet(backbone, mdd=None, rstr=None) — V2 uses backbone only; V5 adds MDD + R-STR
- tracknet_v5() factory in models/tracknet.py builds the full V5 stack

## Testing
- All tests use synthetic data — no GPU, no real video files required
- tests/conftest.py has shared fixtures for synthetic frames + CSV labels
- Model tests use small tensors (batch=1-2, 288x512) to keep them fast
- Training tests use stub models (not real TrackNet) for isolation

## Status (2026-11-15)
- Nothing has been trained on real data. The first run is V2 on tennis; V5 is deferred.
- The code is NOT ready to train. Blocking defects are listed in docs/2026-11-15-pre-training-review.md §4:
  loss NaN under bf16, labels not rescaled to 512x288, label loader only parses our own CSV layout,
  BGR/RGB mismatch at inference, `infer` can't load trainer checkpoints, `train`/`evaluate` CLI stubbed.
- Agreed recipe changes, not yet in code: heatmap radius 30 -> 2.5, training stride 3 -> 1,
  colour jitter once per window, evaluation aligned with published conventions.
- Passing tests do not mean the pipeline works on real data: fixtures are 512x288 and training tests use stub models.

## Gotchas
- CUDA driver warning is expected in WSL2 (tests run on CPU fine)
- Input resolution is always 512x288 (static, required for torch.compile)
- Skip connections tapped BEFORE MaxPool in encoder
- Sigmoid applied in UNetBackbone output head — V5 will need this moved
- Python 3.14 since 2026-11-15 (pinned in .python-version, interpreter managed by uv; system python is still 3.12). pyproject keeps requires-python >=3.12
- No C compiler (gcc) on this machine, so torch.compile fails at the first forward pass. configs/default.yaml has compile_model: true — install build-essential or set it false before training
