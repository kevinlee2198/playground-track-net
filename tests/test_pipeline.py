"""End-to-end: train, evaluate and infer on the miniature tennis dataset."""

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from main import build_parser, run_inference
from training.run import load_config, run_evaluation, run_training

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
