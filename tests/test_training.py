import glob as glob_module
import os
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from training.evaluate import (
    aggregate_metrics,
    compute_detection_metrics,
    evaluate_epoch,
    heatmap_to_position,
)
from training.trainer import Trainer


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class SimpleModel(nn.Module):
    """Tiny model for testing: conv 9ch -> 3ch."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(9, 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))


class SimpleLoss(nn.Module):
    """Simple BCE loss for testing."""

    def forward(self, pred, target):
        return nn.functional.binary_cross_entropy(pred, target)


class PerfectModel(nn.Module):
    """Returns pre-stored outputs for testing."""

    def __init__(self, outputs):
        super().__init__()
        self._outputs = outputs
        self._call_idx = 0
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self._outputs[self._call_idx]
        self._call_idx += 1
        return out


def _make_synthetic_heatmap(x, y, h=288, w=512, radius=5):
    """Create a synthetic heatmap with a ball at (x, y)."""
    heatmap = torch.zeros(h, w)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                py, px = y + dy, x + dx
                if 0 <= py < h and 0 <= px < w:
                    heatmap[py, px] = 1.0
    return heatmap


def _make_synthetic_dataset(num_samples=4, h=32, w=64):
    """Create a tiny synthetic dataset for training tests."""
    frames = torch.randn(num_samples, 9, h, w)
    heatmaps = torch.zeros(num_samples, 3, h, w)
    for i in range(num_samples):
        cx, cy = w // 2, h // 2
        for f in range(3):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx * dx + dy * dy <= 4:
                        heatmaps[i, f, cy + dy, cx + dx] = 1.0
    return TensorDataset(frames, heatmaps)


def _make_test_config(
    tmpdir: str, epochs: int = 1, experiment_name: str = "test"
) -> dict:
    """Create a test training config (CPU, no AMP, no compile)."""
    return {
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "lr_schedule": {"name": "MultiStepLR", "milestones": [20, 25], "gamma": 0.1},
        "batch_size": 2,
        "epochs": epochs,
        "input_size": [64, 32],
        "seq_len": 3,
        "seed": 42,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "amp_dtype": "float32",
        "compile_model": False,
        "checkpoint_dir": os.path.join(tmpdir, "checkpoints"),
        "experiment_name": experiment_name,
        "log_dir": os.path.join(tmpdir, "runs"),
        "detection_threshold": 0.5,
        "distance_threshold": 4,
    }


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_config_loads_from_yaml():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "default.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert config["optimizer"] == "AdamW"
    assert config["learning_rate"] == 1e-4
    assert config["lr_schedule"]["name"] == "MultiStepLR"
    assert config["lr_schedule"]["milestones"] == [20, 25]
    assert config["lr_schedule"]["gamma"] == 0.1
    assert config["batch_size"] == 2
    assert config["epochs"] == 30
    assert config["input_size"] == [512, 288]
    assert config["seq_len"] == 3
    assert config["heatmap_radius"] == 30
    assert config["num_workers"] == 4
    assert config["pin_memory"] is True
    assert config["persistent_workers"] is True
    assert config["amp_dtype"] == "bfloat16"
    assert config["compile_model"] is True
    assert config["seed"] == 42
    assert config["checkpoint_dir"] == "checkpoints"
    assert config["log_dir"] == "runs"


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------


def test_heatmap_to_position_with_ball():
    heatmap = torch.zeros(288, 512)
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            if dx * dx + dy * dy <= 25:
                heatmap[150 + dy, 100 + dx] = 0.9
    x, y, detected = heatmap_to_position(heatmap, threshold=0.5)
    assert detected is True
    assert abs(x - 100) < 2
    assert abs(y - 150) < 2


def test_heatmap_to_position_no_ball():
    heatmap = torch.zeros(288, 512)
    x, y, detected = heatmap_to_position(heatmap, threshold=0.5)
    assert detected is False


def test_compute_detection_metrics_true_positive():
    pred = (100.0, 150.0, True)
    gt = (102.0, 151.0, True)
    tp, fp, fn = compute_detection_metrics(pred, gt, distance_threshold=4)
    assert tp == 1 and fp == 0 and fn == 0


def test_compute_detection_metrics_false_positive():
    pred = (100.0, 150.0, True)
    gt = (200.0, 200.0, True)
    tp, fp, fn = compute_detection_metrics(pred, gt, distance_threshold=4)
    assert tp == 0 and fp == 1 and fn == 1


def test_compute_detection_metrics_false_negative():
    pred = (0.0, 0.0, False)
    gt = (100.0, 150.0, True)
    tp, fp, fn = compute_detection_metrics(pred, gt, distance_threshold=4)
    assert tp == 0 and fp == 0 and fn == 1


def test_compute_detection_metrics_true_negative():
    pred = (0.0, 0.0, False)
    gt = (0.0, 0.0, False)
    tp, fp, fn = compute_detection_metrics(pred, gt, distance_threshold=4)
    assert tp == 0 and fp == 0 and fn == 0


def test_compute_detection_metrics_fp_no_gt():
    pred = (100.0, 150.0, True)
    gt = (0.0, 0.0, False)
    tp, fp, fn = compute_detection_metrics(pred, gt, distance_threshold=4)
    assert tp == 0 and fp == 1 and fn == 0


def test_aggregate_metrics():
    metrics = aggregate_metrics(tp=7, fp=2, fn=1)
    assert abs(metrics["precision"] - 7 / 9) < 1e-6
    assert abs(metrics["recall"] - 7 / 8) < 1e-6
    expected_f1 = 2 * (7 / 9) * (7 / 8) / ((7 / 9) + (7 / 8))
    assert abs(metrics["f1"] - expected_f1) < 1e-6


def test_aggregate_metrics_zero_division():
    metrics = aggregate_metrics(tp=0, fp=0, fn=0)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


# ---------------------------------------------------------------------------
# evaluate_epoch integration
# ---------------------------------------------------------------------------


def test_evaluate_epoch_all_tp():
    gt1 = torch.stack(
        [
            _make_synthetic_heatmap(100, 150),
            _make_synthetic_heatmap(200, 100),
            _make_synthetic_heatmap(300, 200),
        ]
    )
    gt2 = torch.stack(
        [
            _make_synthetic_heatmap(50, 50),
            _make_synthetic_heatmap(400, 250),
            _make_synthetic_heatmap(250, 144),
        ]
    )
    frames = torch.randn(2, 9, 288, 512)
    gt_heatmaps = torch.stack([gt1, gt2])
    dataset = TensorDataset(frames, gt_heatmaps)
    dataloader = DataLoader(dataset, batch_size=2)
    model = PerfectModel(outputs=[gt_heatmaps])
    metrics = evaluate_epoch(model, dataloader, torch.device("cpu"))
    assert metrics["tp"] == 6
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["f1"] == 1.0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def test_trainer_runs_one_epoch():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config(tmpdir, epochs=1, experiment_name="test_run")
        trainer = Trainer(
            model=SimpleModel(),
            loss_fn=SimpleLoss(),
            train_dataset=_make_synthetic_dataset(num_samples=4, h=32, w=64),
            val_dataset=_make_synthetic_dataset(num_samples=2, h=32, w=64),
            config=config,
        )
        trainer.train()
        assert trainer.current_epoch == 1


def test_checkpoint_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config(tmpdir, epochs=2, experiment_name="test_ckpt")
        trainer = Trainer(
            model=SimpleModel(),
            loss_fn=SimpleLoss(),
            train_dataset=_make_synthetic_dataset(num_samples=4, h=32, w=64),
            val_dataset=_make_synthetic_dataset(num_samples=2, h=32, w=64),
            config=config,
        )
        trainer.train()
        original_epoch = trainer.current_epoch
        original_best_f1 = trainer.best_f1
        original_state = {k: v.clone() for k, v in trainer.model.state_dict().items()}

        trainer2 = Trainer(
            model=SimpleModel(),
            loss_fn=SimpleLoss(),
            train_dataset=_make_synthetic_dataset(num_samples=4, h=32, w=64),
            val_dataset=_make_synthetic_dataset(num_samples=2, h=32, w=64),
            config=config,
        )
        ckpt_path = os.path.join(tmpdir, "checkpoints", "test_ckpt", "latest.pt")
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.current_epoch == original_epoch
        assert trainer2.best_f1 == original_best_f1
        for key in original_state:
            assert torch.equal(trainer2.model.state_dict()[key], original_state[key])


# ---------------------------------------------------------------------------
# Package exports + TensorBoard
# ---------------------------------------------------------------------------


def test_training_package_imports():
    from training import (
        Trainer,
        heatmap_to_position,
    )

    assert Trainer is not None
    assert heatmap_to_position is not None


def test_tensorboard_logs_created():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config(tmpdir, epochs=1, experiment_name="test_tb")
        trainer = Trainer(
            model=SimpleModel(),
            loss_fn=SimpleLoss(),
            train_dataset=_make_synthetic_dataset(num_samples=4, h=32, w=64),
            val_dataset=_make_synthetic_dataset(num_samples=2, h=32, w=64),
            config=config,
        )
        trainer.train()
        log_dir = os.path.join(tmpdir, "runs", "test_tb")
        event_files = glob_module.glob(os.path.join(log_dir, "events.out.tfevents.*"))
        assert len(event_files) > 0
