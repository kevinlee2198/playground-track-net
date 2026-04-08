# CLAUDE.md

## Project
TrackNet V2 ball tracking for tennis, badminton, pickleball. PyTorch U-Net heatmap detector.
Design spec: docs/superpowers/specs/2026-03-21-tracknet-ball-tracking-design.md

## Commands
- `uv sync` — install deps
- `uv run pytest tests/ -v` — run all tests (166 tests, ~50s)
- `uv run ruff format .` — format code
- `uv run ruff check --fix .` — lint and auto-fix
- `uv run ty` — type check
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

## Gotchas
- CUDA driver warning is expected in WSL2 (tests run on CPU fine)
- Input resolution is always 512x288 (static, required for torch.compile)
- Skip connections tapped BEFORE MaxPool in encoder
- Sigmoid applied in UNetBackbone output head — V5 will need this moved
- Python 3.14 upgrade coming ~April 2026, all deps must be forward-compatible
