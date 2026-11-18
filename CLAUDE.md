# CLAUDE.md

## Project
TrackNet V2 ball tracking for tennis, badminton, pickleball. PyTorch U-Net heatmap detector.
Design spec: docs/superpowers/specs/2026-03-21-tracknet-ball-tracking-design.md
Preflight numbers and open decisions: docs/2026-11-15-pre-training-review.md
Training runs, results and error analysis: docs/2026-11-16-training-results.md (tennis), docs/2026-11-18-badminton-results.md (badminton)
Datasets (links, licences, verification log): docs/data-sources.md

## Commands
- `uv sync` — install deps
- `uv run pytest tests/ -q -n 4` — run all tests (400 tests, 60–80 s; serial `-v` takes about as long)
- `uv run ruff format .` — format code
- `uv run ruff check --fix .` — lint and auto-fix
- `uv run ty check` — type check (12 pre-existing diagnostics as of 2026-11-15)
- `uv run pytest tests/test_models.py -v` — model tests only
- `uv run pytest tests/test_data.py -v` — data tests only
- `uv run pytest tests/test_training.py -v` — training tests only
- `uv run pytest tests/test_inference.py -v` — inference tests only
- `uv run pytest tests/test_mdd.py -v` — MDD module tests only
- `uv run pytest tests/test_pipeline.py -v` — end-to-end train / evaluate / check-data / infer tests only
- `uv run python main.py extract-frames --root datasets/badminton [--jpeg-quality 95]` — decode every badminton rally video into `<part>/<match>/frames/<rally>/NNNN.jpg`, one per label row; finished rallies are skipped on a re-run
- `uv run python main.py check-data --config configs/default.yaml --out <dir> [--count 12]` — log per-split clip, frame and visibility counts; write label-overlay PNGs to `<dir>` (keep it outside the repo)
- `uv run python main.py train --config configs/default.yaml [--max-steps N] [--resume checkpoints/<experiment_name>/latest.pt]` — train `build_model(config)` (12 input channels with `background_frame: true`, else 9) on the train split, validate on val each epoch; writes `checkpoints/<experiment_name>/latest.pt` every epoch, `best.pt` only when validation `f1` beats the best so far (starts at 0, so a run stuck at F1 0 has none), and TensorBoard logs in `runs/<experiment_name>/`. `--max-steps` is a total that counts steps restored by `--resume`, so N must exceed the checkpoint's step. `--resume` refuses, before writing anything, when the config's experiment directory already holds a `latest.pt` that is not the resume file (so another run's config cannot overwrite it, and resuming from the run's own `best.pt` needs a new `experiment_name`); a resume file that does not exist is reported as such
- `uv run python main.py evaluate --config configs/default.yaml --weights <checkpoint.pt> --split val` — print JSON metrics under both protocols for the model the checkpoint was trained as. The default split is test: always pass `--split val` until the model is final
- `uv run python main.py infer --video <video or frame dir> --model <checkpoint.pt> --output <csv> [--output-video <mp4>] [--threshold 0.5]` — builds the model the checkpoint was trained as, adding the video's median background frame if it was trained with one

## Code Style
- Formatter/linter: ruff (replaces black + flake8). Run before committing.
- Type checker: ty (Astral). Not mypy.
- GroupNorm(num_groups=8) everywhere, never BatchNorm (batch size 2 is too small)
- Use `torchvision.transforms.v2` imports, not v1
- Use `torch.amp.autocast(device_type, dtype=torch.bfloat16)`, not deprecated `torch.cuda.amp`
- No filterpy — custom NumPy Kalman filter instead (filterpy dead since 2018)
- scipy for connected components (ndimage.label) and spline interpolation

## Architecture
- models/ — U-Net backbone, TrackNet wrapper, WBCE loss, MDD module, R-STR head; checkpoint.py: `build_model` (the V2 model a config trains, 9 or 12 input channels), `load_model` (the model a checkpoint was trained as, plus its stored config), `load_model_weights`
- data/ — label loader (labels.py: our CSV layout and the tennis `Label.csv`), clip discovery for tennis and badminton and split by group (sources.py; `discover_clips(config["data"])` picks the discovery function by `data.format`, and everything that finds clips goes through it), badminton frame extraction (extract.py, `extract-frames`), dataset, heatmap generation and coordinate mapping, transforms (`HorizontalFlip`, `FrameColorJitter`, `mixup_batch`, which the Trainer applies to training batches), median background frame (background.py)
- training/ — trainer loop, evaluation metrics, run.py (`load_config`, which follows `base:`; builds the splits; entry points for train, evaluate, check-data)
- configs/ — default.yaml has every key (flip off, experiment name `tennis_v2`); experiment configs (tennis_v2_baseline.yaml, default.yaml with the flip on, which reproduces the first run; tennis_v2_noflip.yaml, default.yaml under its own name with the flip off set explicitly; tennis_v2_bg.yaml, tennis_v2_mixup.yaml and tennis_v2_jitter.yaml, each tennis_v2_noflip plus one setting; badminton_v2.yaml, default.yaml on the badminton dataset with 10 epochs and the flip off, set explicitly; badminton_v2_flip.yaml, badminton_v2 with the flip on and its first 5 epochs only) name a base config with `base:` and set only what differs
- inference/ — postprocess, trajectory rectification, Kalman tracker, video I/O
- utils/ — visualization (ball overlay drawing for annotated video output)
- main.py — CLI: train | evaluate | check-data | extract-frames | infer
- TrackNet(backbone, mdd=None, rstr=None) — V2 uses backbone only; V5 adds MDD + R-STR
- tracknet_v5() factory in models/tracknet.py builds the full V5 stack

## Testing
- Tests use synthetic data only: no downloaded datasets, no real video files. A test that trains must put `checkpoint_dir` and `log_dir` under `tmp_path` (`_fast_training_config`): the default ones hold real runs. The `tennis_root` fixture in tests/conftest.py is a miniature tennis dataset (1280x720 frames, real `Label.csv` header); `badminton_root` is a miniature badminton dataset before extraction (320x180 mp4v videos the fixture writes, `Frame,Visibility,X,Y` CSVs, one video with 3 surplus frames). mp4v is lossy, so compare decoded frames with a tolerance
- Tests must pass on CPU, but the Trainer picks CUDA when it is available, so on this machine Trainer and pipeline tests run on the GPU. Keep both working
- Model tests use small tensors (batch=1-2, 288x512) to keep them fast
- Training unit tests use stub models; tests/test_pipeline.py runs the real TrackNet at 128x72

## Status (2026-11-18)
- First full run done: `tennis_v2_baseline` (the default recipe of the time, with the flip on; 30 epochs, 5.7 h on this laptop). Validation (game7): F1 0.949 at 4 px, 512x288; 0.771 at 4 px of the source. Nothing has been evaluated on the test split.
- Splits (configs/default.yaml): train game1–6, val game7, test game8–10. WASB trains on game1–7, so say so beside any comparison. Leave the test split alone until a model is final.
- The baseline's predictions sit 1.1 px left of the labels on the validation game, caused by the flip augmentation (results doc §3–§4). Run `tennis_v2_noflip` (configs/tennis_v2_noflip.yaml, done) has no such offset (−0.01 px): validation F1 at its best epoch (15) is 0.947 at 4 px, 512x288 and 0.903 at 4 px of the source. Later experiments are compared with it. Since 2026-11-18 `configs/default.yaml` has the flip off and `experiment_name: tennis_v2` (renamed so that resuming the baseline's `latest.pt` under it writes a new run instead of continuing the baseline's with the flip off); configs/tennis_v2_baseline.yaml reproduces the baseline. `infer` has no label-offset correction, by decision (results doc §4, update of 2026-11-18).
- Median background frame (`background_frame: true`, 12 input channels): run `tennis_v2_bg` (configs/tennis_v2_bg.yaml: the no-flip recipe plus the background frame, done 2026-11-17) scored level with `tennis_v2_noflip`: validation F1 at its best epoch (15) is 0.949 at 4 px, 512x288 and 0.909 at 4 px of the source, a difference one run cannot tell from noise (results doc §5). It stays off by default, and later experiments are still compared with `tennis_v2_noflip`.
- Mixup (`mixup_alpha: 0.5`): run `tennis_v2_mixup` (configs/tennis_v2_mixup.yaml: the no-flip recipe plus mixup, done 2026-11-17) scored level with `tennis_v2_noflip` and slightly below it: validation F1 at its best epoch (15) is 0.945 at 4 px, 512x288 and 0.898 at 4 px of the source (results doc §6). It stays off by default (`mixup_alpha: 0`). Its heatmap peaks look lower than the no-flip model's (recall at threshold 0.9: 0.40 against 0.58 at the best epoch), as the soft targets would predict.
- Colour-jitter strength (`color_jitter`): run `tennis_v2_jitter` (configs/tennis_v2_jitter.yaml: the no-flip recipe at WASB's 0.1 and p 0.5, done 2026-11-18) scored a little below `tennis_v2_noflip`: validation F1 at its best epoch (18) is 0.941 at 4 px, 512x288 and 0.896 at 4 px of the source, lower in most epochs but by an amount one run cannot tell from noise (results doc §6). The default stays at 0.3 with `p: 1.0`.
- Still open (review doc §5): batch size and learning rate, `torch.compile` speed.
- Badminton data loading is implemented (`data.format: badminton`, configs/badminton_v2.yaml: 10 epochs, flip off, val Professional/match22, Professional/match23 and Amateur/match3, test the dataset's own Test part). The real frames are extracted (91,214, 21 GB). First run done 2026-11-18: `badminton_v2` (10 epochs, 10 h 45 min, about 64 min per epoch). Validation F1 at its best epoch (7) is 0.951 at 4 px, 512x288 and 0.914 at 4 px of the source; F1 reached 0.950 at epoch 6 and stayed within 0.002 of it. One seed; not comparable with the tennis numbers or with published badminton results, which are on `Test`. No badminton `Test` frame has been looked at and no model or analysis has been run on it (only its row and frame counts are recorded). Its predictions shift by only 0.22 px when the input is mirrored (tennis: 1.14), which suggests the badminton labels have little horizontal offset, so the flip would probably cost little there; not tried (badminton results doc §5).
- V5 is deferred.

## Gotchas
- Model resolution is 512x288: the default `input_size` and the only resolution trained or measured. `input_size` in the config changes it for train, evaluate and check-data (pipeline tests use 128x72); `infer` always resizes to 512x288. Keep it static for torch.compile
- Windows are never padded: up to two frames at the end of each clip are neither trained on with full context nor scored (9 frames in val, 35 in test). `infer` pads its last window so every frame gets a prediction
- Map label coordinates between source and model resolution only with `source_to_model` / `model_to_source` (data/heatmap.py): `(x + 0.5) * scale - 0.5`, through pixel centres as `cv2.resize` does. This deliberately departs from TrackNetV3, whose plain `x * scale` puts the target 0.3 px off the ball at 1280→512
- Tennis visibility 3 rows carry coordinates but count as ball-absent (`visible_flags=(1, 2)`)
- Badminton (`data.format: badminton`, configs/badminton_v2.yaml) needs `main.py extract-frames --root datasets/badminton` once before check-data, train or evaluate: the dataset ships mp4 videos. Extraction covers every part, Test included. Frames are required only for the splits built (`build_dataset` calls `require_frames`): `train` and `evaluate --split val` need only Professional and Amateur, `check-data` builds all three splits and needs Test too. Never run two `extract-frames` on the same root at once: the second removes the first's `frames/<rally>.partial` as a leftover (there is no lock). Groups are `<part>/<match>` (match names repeat across parts); overlay file names write them with `-`. Video frame i is label row `Frame == i`; surplus trailing video frames are dropped and counted, and fewer video frames than label rows is an error. The badminton `Test` part is held out exactly like tennis game8–10
- Tennis labels are about 1.5 source px right of and 1.4 px below the ball in game1–7 (test games not measured). With `horizontal_flip: true` the horizontal part becomes orientation-dependent, and the baseline was 1.1 px off on the validation game because of it (results doc §4). A model trained without the flip reports the label convention, not the ball's visual centre
- Analysis scripts in scripts/analysis/ (per-frame validation errors, prediction offset with mirrored input, label versus ball). Never point them at game8–10 or badminton `Test`. prediction_offset.py finds clips through `discover_clips`, so it takes either format; val_errors.py and label_vs_ball.py read the tennis `Label.csv` columns and are tennis-only (val_errors.py refuses a config whose `data.format` is not tennis; label_vs_ball.py takes a dataset root and game names, not a config)
- Models return sigmoid heatmaps by default; training calls `model(frames, return_logits=True)` and the loss runs in float32 on the logits. Backbone, R-STR head and TrackNet all take `return_logits`
- Two evaluation protocols: `model_res_4px` (4 px at 512x288) and `source_res_4px` (4 px in source pixels = 1.6 px at 512x288 for 1280-wide video). `evaluate` reports both; per-epoch validation and best.pt selection use `f1` at 4 px, 512x288 only
- Checkpoints load with `weights_only=True`, so the config must hold only plain YAML types (dict, list, str, int, float, bool, None); the Trainer rejects anything else
- `train` refuses to overwrite an existing run's `latest.pt` unless `--resume` names that same file. Resume is per epoch: an interrupted epoch restarts from its beginning, and a checkpoint whose epoch was cut short by `--max-steps` warns and skips the rest of that epoch
- A resume warns about each config key that differs from the checkpoint's (apart from `epochs` and `max_steps`). A key the checkpoint predates is compared using the value the code used before the key existed (`_VALUE_BEFORE_KEY` in training/run.py), so old checkpoints do not warn about `horizontal_flip`, `background_frame`, `color_jitter` or `mixup_alpha` when nothing changed. Add a key there when adding a config key that changes training
- The training loss is logged in scientific notation (`%.3e`): the baseline's epoch mean was 4.96e-5 at epoch 5 and 2.36e-5 at epoch 30
- `mixup_alpha` (default 0, off) is TrackNetV3's sample mixup: each training batch, on the device before the forward pass, is blended with a random permutation of itself, frames and heatmaps alike, one weight `max(lam, 1 - lam)` per sample with `lam` from Beta(alpha, alpha). Validation is never mixed. With batch size 2 the permutation pairs each sample with itself half the time, so about half the batches are not mixed. Its draws come from torch's global generator, so a mixup run does not share the off run's sample order after the first epoch. The training loss of a mixup run has a floor above zero from the soft targets, about 1.4e-5 by calculation (`tennis_v2_mixup` ended 1.6e-5 above `tennis_v2_noflip`), so compare runs on validation F1, not training loss
- `color_jitter` has four keys: `brightness`, `contrast`, `saturation` and `p`, the share of training windows jittered. One draw covers every image in a window (and the background, if any). With `p: 1.0` (the default) no number is drawn for the decision, so it draws exactly what the hard-coded jitter drew
- `infer` and `evaluate` build the model from the config stored in the checkpoint (`load_model`). The checkpoint's `background_frame` wins over the config passed to `evaluate`, with a warning when they differ; a checkpoint saved before the key existed counts as false. A bare state dict is the 9-channel V2 model. `infer` still always works at 512x288, and builds the background from the median of up to 50 evenly spaced frames of the video it is given, so for such a model give it one static-camera clip: a video with cuts or camera motion has no meaningful median
- A config file must have every key or inherit them from `default.yaml` via `base: <path>` (paths relative to the file naming them, chains allowed). Nested dicts merge key by key, so an experiment can override one key of `data:`
- With `background_frame: true`, each clip decodes up to 50 frames when a dataset is built, to compute its median background: 28 s of start-up for the train and val splits (65 clips), against 0.3 s without. The background is computed once per clip at model resolution, and the training transforms (jitter, flip) apply to it as to the frames
- Skip connections tapped BEFORE MaxPool in encoder
- Python 3.14 since 2026-11-15 (pinned in .python-version, interpreter managed by uv; system python is still 3.12). pyproject keeps requires-python >=3.12
- No C compiler (gcc) on this machine, so configs/default.yaml has `compile_model: false`. With it true on a CUDA machine, `train` stops before writing anything (on CPU the Trainer never compiles): "torch.compile does not work on this machine (...). Install a C compiler (on Debian/Ubuntu: sudo apt install build-essential) or set compile_model: false in the config."
