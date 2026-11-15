"""End-to-end: train, evaluate and infer on the miniature tennis dataset."""

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from main import build_parser, run_inference
from training.run import load_config, run_evaluation, run_training

CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_train_evaluate_infer_roundtrip(tennis_root, tmp_path):
    config = load_config(CONFIG)
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
        "data": {
            "format": "tennis",
            "root": str(tennis_root),
            "val_groups": ["game2"],
            "test_groups": ["game3"],
        },
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


def _config_for(tennis_root, **data_overrides):
    """The default config pointed at the miniature dataset; nothing is trained."""
    config = load_config(CONFIG)
    config["data"] = {
        "format": "tennis",
        "root": str(tennis_root),
        "val_groups": ["game2"],
        "test_groups": ["game3"],
    } | data_overrides
    return config


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
