"""End-to-end: train, evaluate and infer on the miniature tennis dataset."""

import csv
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from main import build_parser, run_inference
from training.run import (
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


def test_train_evaluate_infer_roundtrip(tennis_root, tmp_path):
    config = _config_for(tennis_root)
    config |= {
        "input_size": [128, 72],  # small for speed; the U-Net is fully convolutional
        "epochs": 1,
        "max_steps": 2,
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


def _fast_training_config(tennis_root, tmp_path, **overrides):
    """A config that trains for a few steps at low resolution on CPU or GPU."""
    return (
        _config_for(tennis_root)
        | {
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
        | overrides
    )


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


def test_resume_into_existing_run_is_allowed(tennis_root, tmp_path):
    run_training(_fast_training_config(tennis_root, tmp_path, epochs=1, max_steps=2))
    latest = tmp_path / "checkpoints" / "e2e" / "latest.pt"

    config = _fast_training_config(tennis_root, tmp_path, epochs=2, max_steps=4)
    run_training(config, resume=latest)
    checkpoint = torch.load(latest, weights_only=True)
    assert (checkpoint["epoch"], checkpoint["global_step"]) == (2, 4)


def test_unknown_data_format_raises(tennis_root):
    config = _config_for(tennis_root, format="pickleball")
    with pytest.raises(ValueError, match="pickleball"):
        run_training(config)


def test_evaluation_rejects_mixed_source_sizes(tennis_root, tmp_path):
    for frame in (tennis_root / "game3" / "Clip1").glob("*.jpg"):
        cv2.imwrite(str(frame), np.zeros((360, 640, 3), dtype=np.uint8))
    config = _config_for(tennis_root, val_groups=[], test_groups=["game2", "game3"])
    with pytest.raises(ValueError, match="source size"):
        run_evaluation(config, tmp_path / "unused.pt", split="test")


def test_evaluation_rejects_aspect_ratio_change(tennis_root, tmp_path):
    config = _config_for(tennis_root) | {"input_size": [128, 128]}
    with pytest.raises(ValueError, match="aspect ratio"):
        run_evaluation(config, tmp_path / "unused.pt", split="test")


def test_write_label_overlays(tennis_root, tmp_path):
    config = load_config(CONFIG)
    config["data"] = {
        "format": "tennis",
        "root": str(tennis_root),
        "val_groups": ["game2"],
        "test_groups": ["game3"],
    }
    paths = write_label_overlays(config, tmp_path / "overlays", count=3)
    assert len(paths) == 3 and all(p.suffix == ".png" and p.exists() for p in paths)
    img = cv2.imread(str(paths[0]))
    assert img is not None
    assert img.shape[0] == 288 and img.shape[1] > 512  # frame plus the enlarged crop
    # Fixture frames are uniform, so any variation is the marker.
    assert img.min() != img.max()


VISIBLE_STEMS = {"0000", "0001", "0002", "0005"}  # visibility 1 or 2 in the fixture


def test_label_overlays_cover_every_split(tennis_root, tmp_path):
    paths = write_label_overlays(_config_for(tennis_root), tmp_path, count=3)
    assert [p.name.split("_")[0] for p in paths] == ["train", "val", "test"]


def test_label_overlays_show_only_visible_frames(tennis_root, tmp_path):
    paths = write_label_overlays(_config_for(tennis_root), tmp_path, count=100)
    assert len(paths) == 16  # every visible frame of the four clips, no more
    assert {p.stem.rsplit("_", 1)[1] for p in paths} == VISIBLE_STEMS


def test_label_overlay_marker_at_rescaled_position(tennis_root, tmp_path):
    write_label_overlays(_config_for(tennis_root), tmp_path, count=100)
    # Frame 0000 is labelled at (640, 360) on a 1280x720 frame and is uniform black.
    img = cv2.imread(str(tmp_path / "test_game3_Clip1_0000.png"))
    assert img is not None
    frame = img[:, :512]
    cx, cy = 256, 144
    assert frame[cy, cx].max() == 0  # hollow: the ball itself is not covered
    for x, y in ((cx + 6, cy), (cx - 6, cy), (cx, cy + 6), (cx, cy - 6)):
        assert frame[y, x].max() > 0, (x, y)
    assert frame[cy, cx + 12].max() == 0  # nothing drawn outside the circle


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
    write_label_overlays(_config_for(tennis_root), tmp_path, count=100)
    img = cv2.imread(str(tmp_path / "test_game3_Clip1_0001.png"))
    assert img is not None
    enlarged = img[:256, 512:]
    # The label is at the crop's centre: frame below-right of it, padding above-left.
    assert enlarged[140, 140].tolist() == [20, 20, 20]
    assert enlarged[100, 100].tolist() == [0, 0, 0]


def test_label_overlays_log_split_counts(tennis_root, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="training.run"):
        write_label_overlays(_config_for(tennis_root), tmp_path, count=3)
    log = caplog.text
    assert "train: 2 clips, 12 frames, visibility {0: 2, 1: 6, 2: 2, 3: 2}" in log
    assert "test: 1 clips, 6 frames, visibility {0: 1, 1: 3, 2: 1, 3: 1}" in log


def test_label_overlay_drawn_for_label_outside_frame(tennis_root, tmp_path, caplog):
    label_path = _move_test_label(tennis_root, 1400, 355)  # frame is 1280 wide
    with caplog.at_level(logging.INFO, logger="training.run"):
        paths = write_label_overlays(_config_for(tennis_root), tmp_path, count=100)
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
        write_label_overlays(_config_for(tennis_root), tmp_path, count=3)
    for split, outside in (("train", 0), ("val", 0), ("test", 1)):
        line = rf"{split}: \d+ clips, .*, {outside} visible labels outside the frame"
        assert re.search(line, caplog.text), (split, caplog.text)
