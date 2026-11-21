"""End-to-end: train, evaluate and infer on the miniature tennis dataset."""

import argparse
import csv
import logging
import re
import shutil
import subprocess
import sys
import weakref
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from data import (
    Compose,
    FrameColorJitter,
    HorizontalFlip,
    TrackNetDataset,
    background_indices,
    build_dataset,
    discover_tennis_clips,
    extract_badminton_frames,
    generate_heatmap,
    median_frame,
    mixup_batch,
    source_to_model,
)
from inference import video_preprocess
from inference.postprocess import heatmap_to_coordinates
from inference.video_preprocess import (
    SourceInfo,
    create_sliding_windows,
    extract_frames,
)
from main import _track_and_write, build_parser, run_inference
from models import TrackNet, build_model
from tests.conftest import CLIP_SIZE, spy_on_video_capture, write_moving_square_clip
from training.run import (
    _build_datasets,
    _source_res_distance,
    _warn_config_changes,
    load_config,
    run_evaluation,
    run_training,
    write_label_overlays,
)

CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def _config_for(tennis_root, **data_overrides):
    """The default config pointed at the miniature dataset."""
    config = load_config(CONFIG)
    config["data"] = {
        "format": "tennis",
        "root": str(tennis_root),
        "val_groups": ["game2"],
        "test_groups": ["game3"],
    } | data_overrides
    return config


def _fast_settings(tmp_path) -> dict:
    """Config keys that train for a few steps at low resolution on CPU or GPU,
    writing under ``tmp_path``."""
    return {
        # Small for speed; the U-Net is fully convolutional.
        "input_size": [128, 72],
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "amp_dtype": "float32",
        "compile_model": False,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "log_dir": str(tmp_path / "runs"),
        "experiment_name": "e2e",
    }


def _fast_training_config(tennis_root, tmp_path, **overrides):
    """The tennis config with ``_fast_settings``, then ``overrides``."""
    return _config_for(tennis_root) | _fast_settings(tmp_path) | overrides


@pytest.mark.parametrize("background", [False, True])
def test_train_evaluate_infer_roundtrip(tennis_root, tmp_path, background):
    # The default has the flip off; this test keeps it in a real training step.
    config = _fast_training_config(
        tennis_root,
        tmp_path,
        epochs=1,
        max_steps=2,
        background_frame=background,
        horizontal_flip=True,
    )
    run_training(config)
    ckpt = tmp_path / "checkpoints" / "e2e" / "latest.pt"
    assert ckpt.exists()

    result = run_evaluation(config, ckpt, split="test", final=True)
    assert result["frames"] == 6
    for protocol in ("model_res_4px", "source_res_4px"):
        m = result[protocol]
        assert {"precision", "recall", "f1", "f1_strict", "accuracy"} <= m.keys()
        assert m["tp"] + m["tn"] + m["fp1"] + m["fp2"] + m["fn"] == 6

    out_csv = tmp_path / "pred.csv"
    args = build_parser().parse_args(
        [
            "infer",
            "--video",
            str(tennis_root / "game3" / "Clip1"),
            "--model",
            str(ckpt),
            "--output",
            str(out_csv),
        ]
    )
    run_inference(args)
    with open(out_csv) as f:
        assert len(list(csv.DictReader(f))) == 6


@pytest.mark.parametrize("background", [False, True])
def test_training_with_mixup_trains_and_validates(
    tennis_root, tmp_path, monkeypatch, background
):
    calls = []

    def counting_mixup(frames, heatmaps, alpha):
        calls.append(alpha)
        return mixup_batch(frames, heatmaps, alpha)

    monkeypatch.setattr("training.trainer.mixup_batch", counting_mixup)
    config = _fast_training_config(
        tennis_root,
        tmp_path,
        epochs=1,
        max_steps=2,
        background_frame=background,
        mixup_alpha=0.5,
    )
    run_training(config)
    assert calls == [0.5, 0.5]  # once per training step
    checkpoint = torch.load(
        tmp_path / "checkpoints" / "e2e" / "latest.pt", weights_only=True
    )
    assert (checkpoint["epoch"], checkpoint["global_step"]) == (1, 2)


class _RecordingModel(nn.Module):
    """Records every input it is given and finds no ball."""

    def __init__(self):
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, x):
        self.inputs.append(x.cpu())
        b, _, h, w = x.shape
        return torch.zeros(b, 3, h, w, device=x.device)


def test_inference_appends_the_videos_median_background(tmp_path, monkeypatch):
    # A fixed scene with a bright square that moves every frame: its median is
    # the scene, which zeros, the first frame or the mean would not match.
    rng = np.random.default_rng(0)
    scene = rng.integers(0, 200, (72, 128, 3), dtype=np.uint8)
    video = tmp_path / "frames"
    video.mkdir()
    for i in range(7):
        img = scene.copy()
        img[20:28, 8 + 16 * i : 16 + 16 * i] = 255
        cv2.imwrite(str(video / f"{i:04d}.png"), img)
    model = _RecordingModel()
    # run_inference imports load_model from models when it runs.
    monkeypatch.setattr(
        "models.load_model", lambda path, device: (model, {"background_frame": True})
    )
    args = build_parser().parse_args(
        [
            "infer",
            "--video",
            str(video),
            "--model",
            str(tmp_path / "unused.pt"),
            "--output",
            str(tmp_path / "pred.csv"),
        ]
    )
    run_inference(args)

    frames, _ = extract_frames(str(video))
    background = median_frame(frames[i] for i in background_indices(len(frames)))
    assert not np.allclose(background, frames[0])
    assert not np.allclose(background, np.mean(frames, axis=0))
    assert len(model.inputs) == 3  # frames 0-2, 3-5 and 6 padded with itself
    for k, x in enumerate(model.inputs):
        assert x.shape == (1, 12, 288, 512)
        window = [frames[min(3 * k + j, len(frames) - 1)] for j in range(3)]
        assert torch.equal(x[0, :9], torch.from_numpy(np.concatenate(window))), k
        assert torch.equal(x[0, 9:], torch.from_numpy(background)), k


class _MixingModel(nn.Module):
    """A stand-in for TrackNet whose heatmaps depend on every image of a window.

    Heatmap j is 0.8 of frame j's grey level (less the background's, when the
    window has one) plus 0.2 of frame j - 1's, cyclically, so the padding of a
    clip's last window changes its heatmaps. A white square gives a peak.
    """

    def forward(self, x):
        b, c, h, w = x.shape
        grey = x.view(b, c // 3, 3, h, w).mean(dim=2)
        frames = grey[:, :3]
        if c == 12:
            frames = (frames - grey[:, 3:]).abs()
        return 0.8 * frames + 0.2 * frames.roll(1, dims=1)


def _infer_args(source: Path, out_csv: Path, *extra: str) -> argparse.Namespace:
    return build_parser().parse_args(
        [
            "infer",
            "--video",
            str(source),
            "--model",
            str(out_csv.parent / "unused.pt"),
            "--output",
            str(out_csv),
            *extra,
        ]
    )


def _use_model(monkeypatch, model: nn.Module, background: bool) -> None:
    # run_inference imports load_model from models when it runs.
    monkeypatch.setattr(
        "models.load_model",
        lambda path, device: (model, {"background_frame": background}),
    )


def _list_based_inference(source: Path, model, background: bool, out_csv: Path):
    """A reference for run_inference that holds every frame of the source
    (extract_frames), cuts them into windows (create_sliding_windows), runs the
    model on each, trims the detections to the frame count, then tracks and writes
    the CSV as run_inference does."""
    frames, metadata = extract_frames(str(source))
    orig_w, orig_h = metadata["original_width"], metadata["original_height"]
    bg = None
    if frames and background:
        bg = median_frame(frames[i] for i in background_indices(len(frames)))
    windows = create_sliding_windows(frames, window_size=3, stride=3, background=bg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detections = []
    with torch.no_grad():
        for window in windows:
            x = torch.from_numpy(window).unsqueeze(0).to(device)
            for hm in model(x).squeeze(0).cpu().numpy():
                detections.append(
                    heatmap_to_coordinates(
                        hm, orig_width=orig_w, orig_height=orig_h, threshold=0.5
                    )
                )
    detections = detections[: len(frames)]
    info = SourceInfo(orig_w, orig_h, metadata.get("fps", 30.0))
    _track_and_write(detections, info, _infer_args(source, out_csv))


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize(
    ("name", "num_frames"),
    [
        ("clip.mp4", 1),
        ("clip.mp4", 8),
        ("clip.mp4", 9),
        ("clip.mp4", 61),
        ("frames", 8),
    ],
)
def test_streaming_inference_matches_the_list_based_path(
    tmp_path, monkeypatch, name, num_frames, background
):
    # 61 frames take 50 of them for the background; 8 pad the last window.
    source = write_moving_square_clip(tmp_path / name, num_frames)
    model = _MixingModel()
    _use_model(monkeypatch, model, background)
    run_inference(_infer_args(source, tmp_path / "streamed.csv"))
    _list_based_inference(source, model, background, tmp_path / "listed.csv")

    streamed = (tmp_path / "streamed.csv").read_text()
    assert streamed == (tmp_path / "listed.csv").read_text()
    rows = list(csv.DictReader(streamed.splitlines()))
    assert len(rows) == num_frames
    # The square is found, except against the background of a one-frame clip,
    # which is that frame.
    found = any(row["Visibility"] == "1" for row in rows)
    assert found == (num_frames > 1 or not background)


def test_infer_fills_a_short_gap_in_a_track_but_does_not_join_far_clusters(
    tmp_path,
):
    """On a 320-wide frame: a cluster on the left, three missing frames, then a
    track on the right, 260 px (65 px per frame) from the cluster, which is
    faster than the fill allows. The track then misses two frames while moving
    5 px per frame, which is filled."""
    left = [(20.0, 90.0, 0.9), (25.0, 90.0, 0.9), (30.0, 90.0, 0.9)]
    right = [(290.0, 90.0, 0.9), (285.0, 90.0, 0.9), (280.0, 90.0, 0.9)]
    right_on = [(265.0, 90.0, 0.9), (260.0, 90.0, 0.9)]
    detections = [*left, *[None] * 3, *right, None, None, *right_on]
    out_csv = tmp_path / "out.csv"
    _track_and_write(
        detections,
        SourceInfo(320, 180, 30.0),
        _infer_args(tmp_path / "unused.mp4", out_csv),
    )
    rows = list(csv.DictReader(out_csv.read_text().splitlines()))
    visible = [int(row["Frame"]) for row in rows if row["Visibility"] == "1"]
    assert visible == [0, 1, 2, *range(6, 13)]


class _LiveCounter:
    """How many of the objects given to ``track`` are alive, now and at most at
    once. Each is counted until it is freed (weakref.finalize; CPython frees an
    object as soon as nothing refers to it any more); one tracked twice counts
    once."""

    def __init__(self):
        self.alive: set[int] = set()
        self.max = 0

    def track(self, obj):
        if id(obj) not in self.alive:
            self.alive.add(id(obj))
            self.max = max(self.max, len(self.alive))
            weakref.finalize(obj, self.alive.discard, id(obj))
        return obj


class _CountingModel(_MixingModel):
    """_MixingModel, counting the inputs it is given and the outputs it returns
    while they are alive."""

    def __init__(self):
        super().__init__()
        self.inputs = _LiveCounter()
        self.outputs = _LiveCounter()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        self.inputs.track(x)
        return self.outputs.track(super().forward(x))


@pytest.mark.parametrize("background", [False, True])
def test_inference_holds_only_a_few_frames_at_once(tmp_path, monkeypatch, background):
    # Counts what infer holds of the video while it runs: the preprocessed
    # frames, the windows built from them (copies), the model's inputs and
    # outputs, and what the heatmap rows the detections are taken from are views
    # of (the output tensor on CPU, its copy on CUDA).
    frames, windows, heatmaps = _LiveCounter(), _LiveCounter(), _LiveCounter()
    frames_alive_per_window: list[int] = []
    preprocess = video_preprocess.preprocess_frame
    concatenate = video_preprocess._concatenate

    def to_coordinates(hm, **kwargs):
        assert hm.base is not None  # a row of the window's heatmaps
        heatmaps.track(hm.base)
        return heatmap_to_coordinates(hm, **kwargs)

    def concatenate_counting(*args):
        frames_alive_per_window.append(len(frames.alive))
        return windows.track(concatenate(*args))

    monkeypatch.setattr(
        video_preprocess, "preprocess_frame", lambda f: frames.track(preprocess(f))
    )
    monkeypatch.setattr(video_preprocess, "_concatenate", concatenate_counting)
    monkeypatch.setattr("main.heatmap_to_coordinates", to_coordinates)
    video = write_moving_square_clip(tmp_path / "clip.mp4", 200)
    model = _CountingModel()
    _use_model(monkeypatch, model, background)
    run_inference(_infer_args(video, tmp_path / "pred.csv"))

    with open(tmp_path / "pred.csv") as f:
        assert len(list(csv.DictReader(f))) == 200
    assert model.calls == 67
    # A window holds three frames. A background model first holds the 50 frames
    # its median is taken over, and has freed them when the first window is built.
    assert 3 <= frames.max <= 6 + (50 if background else 0)
    assert max(frames_alive_per_window) <= 6
    # The next window, input, output and heatmaps are made while the loop still
    # refers to the last ones.
    for counter in (windows, model.inputs, model.outputs, heatmaps):
        assert 1 <= counter.max <= 2


@pytest.mark.parametrize(("background", "passes"), [(False, 1), (True, 3)])
def test_inference_decodes_the_video_once_or_three_times_with_a_background(
    tmp_path, monkeypatch, background, passes
):
    # The container claims 810,570 frames for 61: the frame count a background
    # model's frames are chosen by must come from decoding.
    video = write_moving_square_clip(tmp_path / "clip.mp4", 61)
    model = _MixingModel()
    _use_model(monkeypatch, model, background)
    stats = spy_on_video_capture(monkeypatch, frame_count=810_570)
    run_inference(_infer_args(video, tmp_path / "streamed.csv"))
    assert stats["passes"] == passes

    _list_based_inference(video, model, background, tmp_path / "listed.csv")
    streamed = (tmp_path / "streamed.csv").read_text()
    assert streamed == (tmp_path / "listed.csv").read_text()
    assert len(streamed.splitlines()) == 1 + 61


@pytest.mark.parametrize(
    ("short_pass", "then"),
    [
        (2, "ended after 8 frames when collecting the background"),
        (3, "to 8 frames when run through the model"),
    ],
)
def test_background_inference_refuses_a_video_that_decodes_differently(
    tmp_path, monkeypatch, short_pass, then
):
    # Pass 1 counts 9 frames, pass 2 collects the background's frames, pass 3
    # runs the model; either ending a frame early must stop infer before it
    # writes anything.
    video = write_moving_square_clip(tmp_path / "clip.mp4", 9)
    _use_model(monkeypatch, _MixingModel(), True)
    spy_on_video_capture(monkeypatch, short_pass=(short_pass, 8))
    with pytest.raises(RuntimeError) as raised:
        run_inference(_infer_args(video, tmp_path / "pred.csv"))
    assert str(raised.value) == (
        f"{video} decoded to 9 frames when counted for the background frame, "
        f"but {then}: it does not decode the same way twice"
    )
    assert not (tmp_path / "pred.csv").exists()


@pytest.mark.parametrize("counted", [7, 0])
def test_background_inference_stops_as_soon_as_the_model_pass_has_more_frames(
    tmp_path, monkeypatch, counted
):
    # Pass 1 counts fewer frames than the 9 the model pass yields: infer stops at
    # the window that goes past the count (the third for 7), before the model
    # sees it. Counting 0 builds no background, and a 12-channel model would fail
    # on the 9-channel windows.
    video = write_moving_square_clip(tmp_path / "clip.mp4", 9)
    model = _CountingModel()
    _use_model(monkeypatch, model, True)
    stats = spy_on_video_capture(monkeypatch, short_pass=(1, counted))
    with pytest.raises(RuntimeError) as raised:
        run_inference(_infer_args(video, tmp_path / "pred.csv"))
    stopped_at = 3 * (counted // 3 + 1)  # the end of the window past the count
    assert str(raised.value) == (
        f"{video} decoded to {counted} frames when counted for the background "
        f"frame, but to at least {stopped_at} frames when run through the model: "
        "it does not decode the same way twice"
    )
    assert model.calls == counted // 3
    assert stats["passes"] == (3 if counted else 2)
    assert not (tmp_path / "pred.csv").exists()


def test_inference_releases_the_video_when_the_model_fails(tmp_path, monkeypatch):
    # The traceback refers to run_inference's frame and so to the suspended
    # frame reader; the capture must not wait for it to be dropped.
    class FailingModel(_MixingModel):
        def forward(self, x):
            raise RuntimeError("model failed")

    video = write_moving_square_clip(tmp_path / "clip.mp4", 9)
    _use_model(monkeypatch, FailingModel(), False)
    stats = spy_on_video_capture(monkeypatch)
    with pytest.raises(RuntimeError, match="model failed") as raised:
        run_inference(_infer_args(video, tmp_path / "pred.csv"))
    assert raised.traceback  # still held
    assert stats == {"passes": 1, "open": 0}


@pytest.mark.parametrize("background", [False, True])
def test_inference_on_a_video_that_decodes_no_frames(tmp_path, monkeypatch, background):
    # The file opens, but no frame decodes in any pass: a CSV with no rows.
    video = write_moving_square_clip(tmp_path / "clip.mp4", 9)
    model = _CountingModel()
    _use_model(monkeypatch, model, background)
    spy_on_video_capture(monkeypatch, short_pass=(None, 0))
    run_inference(_infer_args(video, tmp_path / "pred.csv"))
    assert (tmp_path / "pred.csv").read_text().splitlines() == [
        "Frame,X,Y,Confidence,Visibility"
    ]
    assert model.calls == 0


def test_inference_writes_the_annotated_video(tmp_path, monkeypatch):
    # Its frame size and rate are the source's, from read_source_info.
    video = write_moving_square_clip(tmp_path / "clip.mp4", 9)
    out_video = tmp_path / "annotated.mp4"
    _use_model(monkeypatch, _MixingModel(), False)
    run_inference(
        _infer_args(video, tmp_path / "pred.csv", "--output-video", str(out_video))
    )

    cap = cv2.VideoCapture(str(out_video))
    assert cap.isOpened()
    size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    assert decoded == 9
    assert size == CLIP_SIZE
    assert fps == 25.0


def _background_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "training.run"
        and r.levelname == "WARNING"
        and "background_frame" in r.getMessage()
    ]


def _trainer_checkpoint(tmp_path, config: dict) -> Path:
    path = tmp_path / "latest.pt"
    torch.save(
        {"model_state_dict": build_model(config).state_dict(), "config": config}, path
    )
    return path


@pytest.mark.parametrize("trained_with", [True, None])  # None: saved before the key
def test_evaluation_follows_the_checkpoints_background_frame(
    tennis_root, tmp_path, caplog, trained_with
):
    config = _fast_training_config(tennis_root, tmp_path)
    stored = {k: v for k, v in config.items() if k != "background_frame"}
    if trained_with is not None:
        stored["background_frame"] = trained_with
    checkpoint = _trainer_checkpoint(tmp_path, stored)
    trained = bool(trained_with)
    config["background_frame"] = not trained

    with caplog.at_level(logging.INFO, logger="training.run"):
        result = run_evaluation(config, checkpoint, split="val")
    assert result["frames"] == 6
    assert config["background_frame"] is not trained  # the caller's dict is untouched
    assert _background_warnings(caplog) == [
        f"{checkpoint} was trained with background_frame={trained}; the config "
        f"says {not trained}. Using the checkpoint's value."
    ]


def test_evaluation_of_bare_state_dict_has_no_background_frame(
    tennis_root, tmp_path, caplog
):
    config = _fast_training_config(tennis_root, tmp_path, background_frame=True)
    weights = _default_weights(tmp_path)

    with caplog.at_level(logging.INFO, logger="training.run"):
        result = run_evaluation(config, weights, split="val")
    assert result["frames"] == 6
    assert _background_warnings(caplog) == [
        f"{weights} is a bare state dict, loaded as the 9-channel model; the "
        "config says background_frame=True. Using no background frame."
    ]
    assert "was trained with" not in caplog.text  # its training config is unknown


def test_evaluation_of_matching_checkpoint_does_not_warn(tennis_root, tmp_path, caplog):
    config = _fast_training_config(tennis_root, tmp_path, background_frame=True)
    checkpoint = _trainer_checkpoint(tmp_path, config)
    with caplog.at_level(logging.INFO, logger="training.run"):
        assert run_evaluation(config, checkpoint, split="val")["frames"] == 6
    assert _background_warnings(caplog) == []


def test_fresh_run_refuses_to_overwrite_existing_checkpoint(tennis_root, tmp_path):
    config = _fast_training_config(tennis_root, tmp_path)
    latest = tmp_path / "checkpoints" / "e2e" / "latest.pt"
    latest.parent.mkdir(parents=True)
    latest.touch()
    with pytest.raises(FileExistsError, match="--resume") as err:
        run_training(config)
    assert str(latest) in str(err.value)
    assert "experiment_name" in str(err.value)
    assert latest.stat().st_size == 0  # untouched
    assert not (tmp_path / "runs").exists()  # refused before anything was created


def _config_change_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "training.run"
        and r.levelname == "WARNING"
        and "config" in r.getMessage()
    ]


def test_resume_into_existing_run_is_allowed(tennis_root, tmp_path, caplog):
    run_training(_fast_training_config(tennis_root, tmp_path, epochs=1, max_steps=2))
    latest = tmp_path / "checkpoints" / "e2e" / "latest.pt"

    config = _fast_training_config(tennis_root, tmp_path, epochs=2, max_steps=4)
    with caplog.at_level(logging.INFO, logger="training.run"):
        run_training(config, resume=latest)
    checkpoint = torch.load(latest, weights_only=True)
    assert (checkpoint["epoch"], checkpoint["global_step"]) == (2, 4)
    # Only epochs and max_steps changed, and those are meant to change on resume.
    assert _config_change_warnings(caplog) == []


def test_resume_of_background_run(tennis_root, tmp_path):
    run_training(
        _fast_training_config(
            tennis_root, tmp_path, epochs=1, max_steps=2, background_frame=True
        )
    )
    latest = tmp_path / "checkpoints" / "e2e" / "latest.pt"

    config = _fast_training_config(
        tennis_root, tmp_path, epochs=2, max_steps=4, background_frame=True
    )
    run_training(config, resume=latest)
    checkpoint = torch.load(latest, weights_only=True)
    assert (checkpoint["epoch"], checkpoint["global_step"]) == (2, 4)
    first_conv = checkpoint["model_state_dict"]["backbone.down1.conv1.conv.weight"]
    assert first_conv.shape[1] == 12


def test_resume_with_unchanged_config_does_not_warn(tennis_root, tmp_path, caplog):
    config = _fast_training_config(tennis_root, tmp_path, epochs=1, max_steps=2)
    run_training(config)
    with caplog.at_level(logging.INFO, logger="training.run"):
        run_training(config, resume=tmp_path / "checkpoints" / "e2e" / "latest.pt")
    assert "Resumed from" in caplog.text  # the resume did happen
    assert _config_change_warnings(caplog) == []


def test_resume_warns_about_changed_config(tennis_root, tmp_path, caplog):
    config = _fast_training_config(tennis_root, tmp_path, epochs=1, max_steps=2)
    run_training(config)
    changed = config | {"heatmap_radius": 3.0, "epochs": 2, "max_steps": 4}
    with caplog.at_level(logging.INFO, logger="training.run"):
        run_training(changed, resume=tmp_path / "checkpoints" / "e2e" / "latest.pt")
    warnings = _config_change_warnings(caplog)
    assert len(warnings) == 1, warnings
    assert "heatmap_radius: 2.5 -> 3.0" in warnings[0]
    assert "epochs" not in warnings[0] and "max_steps" not in warnings[0]
    assert "have no effect" not in warnings[0]


def test_resume_says_changed_optimizer_settings_have_no_effect(
    tennis_root, tmp_path, caplog
):
    config = _fast_training_config(tennis_root, tmp_path, epochs=1, max_steps=2)
    run_training(config)
    changed = config | {"weight_decay": 0.05, "epochs": 2, "max_steps": 4}
    with caplog.at_level(logging.INFO, logger="training.run"):
        run_training(changed, resume=tmp_path / "checkpoints" / "e2e" / "latest.pt")
    warnings = _config_change_warnings(caplog)
    assert len(warnings) == 1, warnings
    assert "weight_decay: 0.01 -> 0.05" in warnings[0]
    assert warnings[0].endswith(
        "restored from the checkpoint, so these changes have no effect: weight_decay"
    )


@pytest.mark.parametrize("existing_run", [False, True])
def test_resume_from_missing_file_says_so(tennis_root, tmp_path, existing_run):
    config = _fast_training_config(tennis_root, tmp_path)
    checkpoints = tmp_path / "checkpoints"
    if existing_run:
        latest = checkpoints / "e2e" / "latest.pt"
        latest.parent.mkdir(parents=True)
        latest.write_bytes(b"checkpoint")
    before = sorted(checkpoints.rglob("*"))
    typo = checkpoints / "e2e" / "latest.p"
    with pytest.raises(FileNotFoundError) as err:
        run_training(config, resume=typo)
    assert str(err.value) == f"resume checkpoint {typo} does not exist"
    assert sorted(checkpoints.rglob("*")) == before  # nothing created
    assert not (tmp_path / "runs").exists()


def test_resume_from_directory_says_so(tennis_root, tmp_path):
    config = _fast_training_config(tennis_root, tmp_path)
    checkpoints = tmp_path / "checkpoints"
    run_dir = checkpoints / "e2e"
    run_dir.mkdir(parents=True)
    with pytest.raises(IsADirectoryError) as err:
        run_training(config, resume=run_dir)
    assert str(err.value) == (
        f"resume checkpoint {run_dir} is a directory, not a checkpoint file"
    )
    assert list(checkpoints.rglob("*")) == [run_dir]  # nothing created
    assert not (tmp_path / "runs").exists()


def test_resume_from_own_best_refuses_to_replace_latest(tennis_root, tmp_path):
    config = _fast_training_config(tennis_root, tmp_path)
    latest = tmp_path / "checkpoints" / "e2e" / "latest.pt"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b"latest")
    best = latest.with_name("best.pt")
    best.write_bytes(b"best")
    with pytest.raises(FileExistsError) as err:
        run_training(config, resume=best)
    assert str(err.value) == (
        f"continuing from {best} would replace this run's own {latest}: set a "
        "new experiment_name to branch a new run from it"
    )
    assert latest.read_bytes() == b"latest"  # untouched
    assert not (tmp_path / "runs").exists()


def test_resume_refuses_to_overwrite_another_experiments_checkpoint(
    tennis_root, tmp_path, monkeypatch
):
    def config(name, **overrides):
        return _fast_training_config(
            tennis_root, tmp_path, experiment_name=name, **overrides
        )

    run_training(config("a", epochs=1, max_steps=2))
    run_training(config("b", epochs=1, max_steps=2))
    checkpoints = tmp_path / "checkpoints"
    latest_a = checkpoints / "a" / "latest.pt"
    latest_b = checkpoints / "b" / "latest.pt"
    before = latest_b.read_bytes()

    with pytest.raises(FileExistsError) as err:
        run_training(config("b", epochs=2, max_steps=4), resume=latest_a)
    assert str(latest_b) in str(err.value)
    assert str(latest_a) in str(err.value)
    assert "experiment_name" in str(err.value)
    assert latest_b.read_bytes() == before  # untouched

    # An experiment with no checkpoint yet may start from another run's.
    run_training(config("c", epochs=2, max_steps=4), resume=latest_a)
    checkpoint = torch.load(checkpoints / "c" / "latest.pt", weights_only=True)
    assert (checkpoint["epoch"], checkpoint["global_step"]) == (2, 4)

    # Its own latest.pt, named through a symlink, is the same file.
    link = tmp_path / "link.pt"
    link.symlink_to(latest_a)
    run_training(config("a", epochs=2, max_steps=4), resume=link)
    checkpoint = torch.load(latest_a, weights_only=True)
    assert (checkpoint["epoch"], checkpoint["global_step"]) == (2, 4)

    # And so is a path relative to the working directory.
    monkeypatch.chdir(tmp_path)
    run_training(
        config("a", epochs=3, max_steps=6), resume=Path("checkpoints/a/latest.pt")
    )
    checkpoint = torch.load(latest_a, weights_only=True)
    assert (checkpoint["epoch"], checkpoint["global_step"]) == (3, 6)


class _PassedTheGuard(Exception):
    """Raised by a stubbed dataset build: run_training got past its resume guard."""


def test_resume_through_symlinked_checkpoint_directory(
    tennis_root, tmp_path, monkeypatch
):
    # The layout real runs use: a git worktree whose checkpoints directory is a
    # symlink to the main checkout's.
    real = tmp_path / "main" / "checkpoints"
    real.mkdir(parents=True)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / "checkpoints").symlink_to(real, target_is_directory=True)
    monkeypatch.chdir(worktree)
    config = _fast_training_config(
        tennis_root, tmp_path, checkpoint_dir="checkpoints", epochs=1, max_steps=2
    )
    run_training(config)
    latest = real / "e2e" / "latest.pt"
    assert latest.is_file()

    def passed_the_guard(config, splits):
        raise _PassedTheGuard

    monkeypatch.setattr("training.run._build_datasets", passed_the_guard)
    for resume in (Path("checkpoints/e2e/latest.pt"), latest):
        with pytest.raises(_PassedTheGuard):
            run_training(config, resume=resume)
    with pytest.raises(FileExistsError, match="--resume"):
        run_training(config)


def _warnings_for(caplog, stored: dict, current: dict) -> list[str]:
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="training.run"):
        _warn_config_changes(stored, current)
    return _config_change_warnings(caplog)


# What the code did before each of these config keys existed.
_OLD_BEHAVIOUR = {
    "horizontal_flip": True,
    "background_frame": False,
    "color_jitter": {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "p": 1.0},
    "mixup_alpha": 0.0,
}


@pytest.mark.parametrize(
    ("key", "new"),
    [
        ("horizontal_flip", False),
        ("background_frame", True),
        ("color_jitter", _OLD_BEHAVIOUR["color_jitter"] | {"p": 0.5}),
        ("mixup_alpha", 0.5),
    ],
)
def test_resume_reads_a_key_a_checkpoint_predates_as_its_old_behaviour(
    caplog, key, new
):
    stored = {"seed": 42}  # saved before any of these keys
    current = {"seed": 42} | _OLD_BEHAVIOUR
    assert _warnings_for(caplog, stored, current) == []

    warnings = _warnings_for(caplog, stored, current | {key: new})
    assert len(warnings) == 1, warnings
    assert f"{key}: {_OLD_BEHAVIOUR[key]!r} -> {new!r}" in warnings[0]
    for other in _OLD_BEHAVIOUR.keys() - {key}:
        assert other not in warnings[0]


def test_resume_shows_a_key_the_current_config_lacks_as_none(caplog):
    warnings = _warnings_for(caplog, {"horizontal_flip": True}, {})
    assert len(warnings) == 1, warnings
    assert "horizontal_flip: True -> None" in warnings[0]


def test_cli_log_lines_carry_timestamp_level_and_logger(tennis_root, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config_for(tennis_root)))
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "check-data",
            "--config",
            str(config_path),
            "--out",
            str(tmp_path / "overlays"),
            "--count",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )
    assert result.returncode == 0, result.stderr
    line = (
        r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} INFO training\.run: train: 2 clips, .*"
    )
    assert re.search(line, result.stderr), result.stderr


@pytest.mark.parametrize("flip", [True, False])
def test_train_split_flips_only_when_the_config_says_so(tennis_root, flip):
    config = _config_for(tennis_root) | {"horizontal_flip": flip}
    datasets = _build_datasets(config, ("train", "val"))
    train_clip = cast(TrackNetDataset, datasets["train"].datasets[0])
    val_clip = cast(TrackNetDataset, datasets["val"].datasets[0])
    kinds = [type(t) for t in cast(Compose, train_clip.transform).transforms]
    assert (HorizontalFlip in kinds) is flip
    assert FrameColorJitter in kinds
    assert val_clip.transform is None


@pytest.mark.parametrize(
    ("color_jitter", "strength", "p"),
    [
        (None, 0.3, 1.0),  # the default config's values
        ({"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "p": 0.5}, 0.1, 0.5),
    ],
)
def test_train_split_jitters_with_the_configured_strength_and_p(
    tennis_root, color_jitter, strength, p
):
    config = _config_for(tennis_root)
    if color_jitter is not None:
        config["color_jitter"] = color_jitter
    datasets = _build_datasets(config, ("train", "val"))
    train_clip = cast(TrackNetDataset, datasets["train"].datasets[0])
    val_clip = cast(TrackNetDataset, datasets["val"].datasets[0])
    (jitter,) = [
        t
        for t in cast(Compose, train_clip.transform).transforms
        if isinstance(t, FrameColorJitter)
    ]
    expected = pytest.approx((1 - strength, 1 + strength))
    assert jitter.jitter.brightness == expected
    assert jitter.jitter.contrast == expected
    assert jitter.jitter.saturation == expected
    assert jitter.p == p
    assert val_clip.transform is None


@pytest.mark.parametrize("background", [False, True])
def test_background_frame_adds_channels_to_every_split(tennis_root, background):
    config = _config_for(tennis_root) | {"background_frame": background}
    datasets = _build_datasets(config, ("train", "val"))
    channels = 12 if background else 9
    for split in ("train", "val"):
        frames, heatmaps = datasets[split][0]
        assert frames.shape == (channels, 288, 512), split
        assert heatmaps.shape == (3, 288, 512), split


def test_unknown_data_format_raises(tennis_root, tmp_path):
    # Output paths under tmp_path: the default ones may hold a real run.
    config = _fast_training_config(tennis_root, tmp_path) | {
        "data": _config_for(tennis_root, format="pickleball")["data"]
    }
    with pytest.raises(ValueError, match="pickleball"):
        run_training(config)


def _default_weights(tmp_path) -> Path:
    """A bare state dict of the default model: evaluation reads its weights
    before it builds the split."""
    path = tmp_path / "weights.pt"
    torch.save(TrackNet().state_dict(), path)
    return path


def test_evaluation_rejects_mixed_source_sizes(tennis_root, tmp_path):
    for frame in (tennis_root / "game3" / "Clip1").glob("*.jpg"):
        cv2.imwrite(str(frame), np.zeros((360, 640, 3), dtype=np.uint8))
    config = _config_for(tennis_root, val_groups=[], test_groups=["game2", "game3"])
    with pytest.raises(ValueError, match="source size"):
        run_evaluation(config, _default_weights(tmp_path), split="test", final=True)


def test_evaluation_rejects_aspect_ratio_change(tennis_root, tmp_path):
    config = _config_for(tennis_root) | {"input_size": [128, 128]}
    with pytest.raises(ValueError, match="aspect ratio"):
        run_evaluation(config, _default_weights(tmp_path), split="test", final=True)


def test_source_resolution_threshold_is_4px_of_the_source(tennis_root):
    dataset = build_dataset(discover_tennis_clips(tennis_root), stride=3, radius=2.5)
    assert _source_res_distance(dataset, 512, 288) == pytest.approx(1.6)
    assert _source_res_distance(dataset, 128, 72) == pytest.approx(0.4)


class _BallOffBy3pxModel(nn.Module):
    """Predicts the ball 3 px (at model resolution) right of source pixel
    (640, 360) in every frame."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))  # something for weights to hold

    def forward(self, x):
        b, _, h, w = x.shape
        heatmap = generate_heatmap(
            source_to_model(640, 1280, w) + 3,
            source_to_model(360, 720, h),
            visibility=1,
            height=h,
            width=w,
        )
        return heatmap.expand(b, 3, h, w).to(x.device)


def test_evaluation_protocols_disagree_between_their_thresholds(
    tennis_root, tmp_path, monkeypatch
):
    # Every test frame shows the ball at (640, 360); the model is 3 px off,
    # inside 4.0 model pixels but outside 4 source pixels (1.6 model pixels).
    label_path = tennis_root / "game3" / "Clip1" / "Label.csv"
    with open(label_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file name", "visibility", "x-coordinate", "y-coordinate"])
        writer.writerows((f"{i:04d}.jpg", 1, 640, 360) for i in range(6))
    monkeypatch.setattr(
        "models.checkpoint.build_model", lambda config=None: _BallOffBy3pxModel()
    )
    weights = tmp_path / "weights.pt"
    torch.save(_BallOffBy3pxModel().state_dict(), weights)
    config = _config_for(tennis_root) | {"num_workers": 0, "pin_memory": False}

    result = run_evaluation(config, weights, split="test", final=True)
    assert (result["model_res_4px"]["tp"], result["model_res_4px"]["fp1"]) == (6, 0)
    assert (result["source_res_4px"]["tp"], result["source_res_4px"]["fp1"]) == (0, 6)


def test_write_label_overlays(tennis_root, tmp_path):
    paths = write_label_overlays(
        _config_for(tennis_root), tmp_path / "overlays", count=3
    )
    assert len(paths) == 3 and all(p.suffix == ".png" and p.exists() for p in paths)
    img = cv2.imread(str(paths[0]))
    assert img is not None
    assert img.shape[0] == 288 and img.shape[1] > 512  # frame plus the enlarged crop
    # Fixture frames are uniform, so any variation is the marker.
    assert img.min() != img.max()


VISIBLE_STEMS = {"0000", "0001", "0002", "0005"}  # visibility 1 or 2 in the fixture


def test_label_overlays_cover_every_split(tennis_root, tmp_path):
    paths = write_label_overlays(_config_for(tennis_root), tmp_path, 3, final=True)
    assert [p.name.split("_")[0] for p in paths] == ["train", "val", "test"]


def test_label_overlays_show_only_visible_frames(tennis_root, tmp_path):
    paths = write_label_overlays(_config_for(tennis_root), tmp_path, 100, final=True)
    assert len(paths) == 16  # every visible frame of the four clips, no more
    assert {p.stem.rsplit("_", 1)[1] for p in paths} == VISIBLE_STEMS


def test_label_overlay_marker_at_rescaled_position(tennis_root, tmp_path):
    write_label_overlays(_config_for(tennis_root), tmp_path, 100, final=True)
    # Frame 0000 is labelled at (640, 360) on a 1280x720 frame and is uniform black.
    # That maps to (255.7, 143.7) at 512x288, so the frame marker is centred on
    # the nearest pixel (256, 144).
    img = cv2.imread(str(tmp_path / "test_game3_Clip1_0000.png"))
    assert img is not None
    frame = img[:, :512]
    cx, cy = 256, 144
    assert frame[cy, cx].max() == 0  # hollow: the ball itself is not covered
    for x, y in ((cx + 6, cy), (cx - 6, cy), (cx, cy + 6), (cx, cy - 6)):
        assert frame[y, x].max() > 0, (x, y)
    assert frame[cy, cx + 12].max() == 0  # nothing drawn outside the circle
    # The enlarged crop keeps the sub-pixel offset: the crop starts at model
    # pixel 256 - 32, so the label sits at (31.7 * 4 + 1.5) = 128.3 -> 128.
    enlarged = img[:256, 512:]
    ex = ey = 128
    assert enlarged[ey, ex].max() == 0
    for x, y in ((ex + 24, ey), (ex - 24, ey), (ex, ey + 24), (ex, ey - 24)):
        assert enlarged[y, x].max() > 0, (x, y)


def _move_test_label(tennis_root, x, y):
    """Move the ball of game3/Clip1 frame 0001 (visible, uniform grey 20) to (x, y)."""
    label_path = tennis_root / "game3" / "Clip1" / "Label.csv"
    with open(label_path, newline="") as f:
        rows = list(csv.reader(f))
    rows[2][2:4] = [str(x), str(y)]
    with open(label_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return label_path


def test_label_overlay_crop_stays_centred_at_frame_edge(tennis_root, tmp_path):
    _move_test_label(tennis_root, 0, 0)
    write_label_overlays(_config_for(tennis_root), tmp_path, 100, final=True)
    img = cv2.imread(str(tmp_path / "test_game3_Clip1_0001.png"))
    assert img is not None
    enlarged = img[:256, 512:]
    # The label is at the crop's centre: frame below-right of it, padding above-left.
    assert enlarged[140, 140].tolist() == [20, 20, 20]
    assert enlarged[100, 100].tolist() == [0, 0, 0]


def test_label_overlays_log_split_counts(tennis_root, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="training.run"):
        write_label_overlays(_config_for(tennis_root), tmp_path, 3, final=True)
    log = caplog.text
    assert "train: 2 clips, 12 frames, visibility {0: 2, 1: 6, 2: 2, 3: 2}" in log
    assert "test: 1 clips, 6 frames, visibility {0: 1, 1: 3, 2: 1, 3: 1}" in log


def test_label_overlay_drawn_for_label_outside_frame(tennis_root, tmp_path, caplog):
    label_path = _move_test_label(tennis_root, 1400, 355)  # frame is 1280 wide
    with caplog.at_level(logging.INFO, logger="training.run"):
        paths = write_label_overlays(
            _config_for(tennis_root), tmp_path, 100, final=True
        )
    path = tmp_path / "test_game3_Clip1_0001.png"
    assert path in paths
    img = cv2.imread(str(path))
    assert img is not None
    assert img.shape == (288, 768, 3)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        str(label_path) in m and "frame 1" in m and "(1400.0, 355.0)" in m
        for m in warnings
    ), warnings


def test_label_overlays_count_visible_labels_outside_frame(
    tennis_root, tmp_path, caplog
):
    _move_test_label(tennis_root, 1400, 355)
    # count=3 draws frame 0002 of the test clip, not the moved frame 0001: the
    # count covers every label row, not only those drawn.
    with caplog.at_level(logging.INFO, logger="training.run"):
        write_label_overlays(_config_for(tennis_root), tmp_path, 3, final=True)
    for split, outside in (("train", 0), ("val", 0), ("test", 1)):
        line = rf"{split}: \d+ clips, .*, {outside} visible labels outside the frame"
        assert re.search(line, caplog.text), (split, caplog.text)


def _remove_frame(clip_dir: Path, stem: str) -> None:
    """Delete frame ``stem`` from both the clip's images and its Label.csv."""
    (clip_dir / f"{stem}.jpg").unlink()
    label_path = clip_dir / "Label.csv"
    with open(label_path, newline="") as f:
        rows = [row for row in csv.reader(f) if row[0] != f"{stem}.jpg"]
    with open(label_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def _gap_line(split: str, gaps: int) -> str:
    return (
        rf"{split}: \d+ clips, .* visible labels outside the frame, "
        rf"{gaps} clips with non-consecutive frame ids, source sizes 1280x720\n"
    )


def test_check_data_reports_no_frame_gaps_in_consecutive_clips(
    tennis_root, tmp_path, caplog
):
    with caplog.at_level(logging.INFO, logger="training.run"):
        write_label_overlays(_config_for(tennis_root), tmp_path, 3, final=True)
    for split in ("train", "val", "test"):
        assert re.search(_gap_line(split, 0), caplog.text), (split, caplog.text)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_check_data_reports_clip_with_frame_gap(tennis_root, tmp_path, caplog):
    clip_dir = tennis_root / "game3" / "Clip1"
    _remove_frame(clip_dir, "0003")  # frames 0, 1, 2, 4, 5
    with caplog.at_level(logging.INFO, logger="training.run"):
        write_label_overlays(_config_for(tennis_root), tmp_path, 3, final=True)
    assert re.search(_gap_line("test", 1), caplog.text), caplog.text
    assert re.search(_gap_line("train", 0), caplog.text), caplog.text
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, warnings
    assert str(clip_dir) in warnings[0] and "2 to 4" in warnings[0], warnings


def test_check_data_computes_no_background(tennis_root, tmp_path, monkeypatch):
    def no_median(frames):
        raise AssertionError("check-data computed a background")

    monkeypatch.setattr("data.dataset.median_frame", no_median)
    config = _config_for(tennis_root) | {"background_frame": True}
    assert len(write_label_overlays(config, tmp_path, count=3)) == 3
    assert config["background_frame"] is True  # the caller's dict is untouched


def test_training_rejects_empty_val_split(tennis_root, tmp_path):
    config = _fast_training_config(tennis_root, tmp_path) | {
        "data": _config_for(tennis_root, val_groups=[])["data"]
    }
    with pytest.raises(ValueError, match=r"val split.*data\.val_groups"):
        run_training(config)
    assert not (tmp_path / "runs").exists()


def test_evaluation_rejects_empty_split(tennis_root, tmp_path):
    config = _config_for(tennis_root, test_groups=[])
    with pytest.raises(ValueError, match=r"test split.*data\.test_groups"):
        run_evaluation(config, _default_weights(tmp_path), split="test", final=True)


def test_check_data_skips_empty_split(tennis_root, tmp_path, caplog):
    config = _config_for(tennis_root, test_groups=[])
    with caplog.at_level(logging.INFO, logger="training.run"):
        paths = write_label_overlays(config, tmp_path, count=3, final=True)
    assert "test: no clips, skipped" in caplog.text
    assert len(paths) == 3
    assert {p.name.split("_")[0] for p in paths} == {"train", "val"}


BADMINTON_CONFIG = CONFIG.with_name("badminton_v2.yaml")


def _badminton_config(badminton_root, tmp_path) -> dict:
    """The shipped badminton config, pointed at the miniature dataset, with
    ``_fast_settings``: val Amateur/match2, test Test/match1."""
    config = load_config(BADMINTON_CONFIG)
    config["data"] |= {
        "root": str(badminton_root),
        "val_groups": ["Amateur/match2"],
        "test_groups": ["Test/match1"],
    }
    return config | _fast_settings(tmp_path)


def test_badminton_extract_check_data_train_evaluate(badminton_root, tmp_path, caplog):
    # extract-frames through the CLI, as a user runs it
    result = subprocess.run(
        [sys.executable, "main.py", "extract-frames", "--root", str(badminton_root)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )
    assert result.returncode == 0, result.stderr
    assert (
        "INFO data.extract: extract-frames: 5 rallies extracted, 0 skipped "
        "(already extracted), 30 frames written, 3 surplus video frames dropped"
    ) in result.stderr

    config = _badminton_config(badminton_root, tmp_path) | {
        "epochs": 1,
        "max_steps": 2,
    }

    with caplog.at_level(logging.INFO, logger="training.run"):
        paths = write_label_overlays(config, tmp_path / "overlays", 3, final=True)
    # The middle visible frame of each split; train's is in its second rally.
    assert [p.name for p in paths] == [
        "train_Professional-match2_1_01_10_0003.png",
        "val_Amateur-match2_1_01_01_0003.png",
        "test_Test-match1_1_01_01_0003.png",
    ]
    assert "train: 3 clips, 18 frames, visibility {0: 3, 1: 15}" in caplog.text
    assert "val: 1 clips, 6 frames, visibility {0: 1, 1: 5}" in caplog.text
    assert "source sizes 320x180" in caplog.text

    run_training(config)
    ckpt = tmp_path / "checkpoints" / "e2e" / "latest.pt"
    checkpoint = torch.load(ckpt, weights_only=True)
    assert (checkpoint["epoch"], checkpoint["global_step"]) == (1, 2)
    assert checkpoint["config"]["data"]["format"] == "badminton"

    result = run_evaluation(config, ckpt, split="val")
    assert result["frames"] == 6
    for protocol in ("model_res_4px", "source_res_4px"):
        m = result[protocol]
        assert m["tp"] + m["tn"] + m["fp1"] + m["fp2"] + m["fn"] == 6


def test_badminton_splits_need_frames_only_when_built(badminton_root, tmp_path):
    # Every rally extracted but the test split's.
    extract_badminton_frames(badminton_root)
    test_frames = badminton_root / "Test" / "match1" / "frames" / "1_01_01"
    shutil.rmtree(test_frames)
    config = _badminton_config(badminton_root, tmp_path)

    datasets = _build_datasets(config, ("train", "val"))
    assert (len(datasets["train"].datasets), len(datasets["val"].datasets)) == (3, 1)

    with pytest.raises(FileNotFoundError) as err:
        _build_datasets(config, ("test",))
    message = str(err.value)
    assert message.startswith(f"Test/match1/1_01_01: no frames directory {test_frames}")
    assert message.endswith(
        f"run `main.py extract-frames --root {badminton_root}` first"
    )
