import datetime
import glob as glob_module
import logging
import math
import os
import pathlib
import re
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from models import load_model_weights
from models.losses import WBCEFocalLoss
from training.evaluate import (
    aggregate_metrics,
    classify_frame,
    evaluate_epoch,
    heatmap_to_position,
)
from training.run import load_config
from training.trainer import Trainer, ensure_compile_supported


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class SimpleModel(nn.Module):
    """Tiny model for testing: conv 9ch -> 3ch."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(9, 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_logits=False):
        logits = self.conv(x)
        return logits if return_logits else self.sigmoid(logits)


class SimpleLoss(nn.Module):
    """Simple BCE loss on logits for testing."""

    def forward(self, logits, target):
        return nn.functional.binary_cross_entropy_with_logits(logits, target)


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


class CentreBallModel(SimpleModel):
    """Trains like SimpleModel, but its probabilities always mark one pixel at
    the frame centre, where _make_synthetic_dataset puts the ball, so
    validation f1 is 1.0 regardless of the random seed."""

    def forward(self, x, return_logits=False):
        if return_logits:
            return super().forward(x, return_logits=True)
        b, _, h, w = x.shape
        probs = torch.zeros(b, 3, h, w, device=x.device)
        probs[..., h // 2, w // 2] = 1.0
        return probs


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
        "weight_decay": 0.01,
        "lr_schedule": {"milestones": [20, 25], "gamma": 0.1},
        "batch_size": 2,
        "epochs": epochs,
        "input_size": [64, 32],
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
    assert config["weight_decay"] == 0.01
    assert config["lr_schedule"] == {"milestones": [20, 25], "gamma": 0.1}
    assert config["batch_size"] == 2
    assert config["epochs"] == 30
    assert config["input_size"] == [512, 288]
    assert config["heatmap_radius"] == 2.5
    assert config["train_stride"] == 1
    assert config["log_every"] == 100
    assert config["experiment_name"] == "tennis_v2_baseline"
    assert config["data"] == {
        "format": "tennis",
        "root": "datasets/tennis",
        "val_groups": ["game7"],
        "test_groups": ["game8", "game9", "game10"],
    }
    assert config["num_workers"] == 4
    assert config["pin_memory"] is True
    assert config["persistent_workers"] is True
    assert config["amp_dtype"] == "bfloat16"
    assert config["compile_model"] is False
    assert config["seed"] == 42
    assert config["checkpoint_dir"] == "checkpoints"
    assert config["log_dir"] == "runs"


def test_load_config_merges_over_its_base(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "epochs: 30\nlr_schedule: {milestones: [20, 25], gamma: 0.1}\n"
    )
    (tmp_path / "exp.yaml").write_text(
        "base: base.yaml\nepochs: 5\nlr_schedule: {gamma: 0.5}\n"
    )
    assert load_config(tmp_path / "exp.yaml") == {
        "epochs": 5,
        "lr_schedule": {"milestones": [20, 25], "gamma": 0.5},
    }


def test_load_config_follows_a_chain_of_bases(tmp_path):
    (tmp_path / "a.yaml").write_text("x: a\ny: a\nz: a\n")
    (tmp_path / "sub").mkdir()
    # A base path is relative to the file that names it.
    (tmp_path / "sub" / "b.yaml").write_text("base: ../a.yaml\ny: b\nz: b\n")
    (tmp_path / "c.yaml").write_text("base: sub/b.yaml\nz: c\n")
    assert load_config(tmp_path / "c.yaml") == {"x": "a", "y": "b", "z": "c"}


def test_load_config_rejects_a_base_cycle(tmp_path):
    (tmp_path / "a.yaml").write_text("base: b.yaml\nx: 1\n")
    (tmp_path / "b.yaml").write_text("base: a.yaml\ny: 2\n")
    with pytest.raises(ValueError, match="base.*cycle"):
        load_config(tmp_path / "a.yaml")


def test_noflip_config_is_the_default_with_two_overrides():
    configs = pathlib.Path(__file__).parent.parent / "configs"
    default = load_config(configs / "default.yaml")
    noflip = load_config(configs / "tennis_v2_noflip.yaml")
    assert noflip == default | {
        "horizontal_flip": False,
        "experiment_name": "tennis_v2_noflip",
    }


def test_bg_config_is_the_noflip_config_with_two_overrides():
    configs = pathlib.Path(__file__).parent.parent / "configs"
    noflip = load_config(configs / "tennis_v2_noflip.yaml")
    bg = load_config(configs / "tennis_v2_bg.yaml")
    assert bg == noflip | {"background_frame": True, "experiment_name": "tennis_v2_bg"}
    assert bg["horizontal_flip"] is False


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


def test_heatmap_to_position_uses_largest_blob():
    hm = torch.zeros(288, 512)
    hm[100:110, 200:210] = 1.0  # 100 px
    hm[20:22, 30:32] = 1.0  # 4 px of noise
    x, y, detected = heatmap_to_position(hm)
    assert detected and x == pytest.approx(204.5) and y == pytest.approx(104.5)


def test_classify_frame():
    assert classify_frame((10, 10, True), (12, 10, True), 4.0) == "tp"
    assert classify_frame((10, 10, True), (20, 10, True), 4.0) == "fp1"
    assert classify_frame((10, 10, True), (0, 0, False), 4.0) == "fp2"
    assert classify_frame((0, 0, False), (10, 10, True), 4.0) == "fn"
    assert classify_frame((0, 0, False), (0, 0, False), 4.0) == "tn"


def test_classify_frame_distance_equal_to_threshold_is_tp():
    assert classify_frame((10, 10, True), (14, 10, True), 4.0) == "tp"


def test_aggregate_metrics_both_recall_conventions():
    m = aggregate_metrics({"tp": 8, "tn": 2, "fp1": 1, "fp2": 1, "fn": 2})
    assert m["precision"] == pytest.approx(0.8)
    assert m["recall"] == pytest.approx(0.8)
    assert m["f1"] == pytest.approx(0.8)
    assert m["recall_strict"] == pytest.approx(8 / 11)
    assert m["f1_strict"] == pytest.approx(16 / 21)
    assert m["accuracy"] == pytest.approx(10 / 14)


def test_aggregate_metrics_zero_division():
    keys = {"precision", "recall", "f1", "recall_strict", "f1_strict", "accuracy"}
    empty = aggregate_metrics({"tp": 0, "tn": 0, "fp1": 0, "fp2": 0, "fn": 0})
    assert set(empty) == keys
    assert all(value == 0.0 for value in empty.values())

    # Only true negatives: every ball-based ratio has a zero denominator.
    no_ball = aggregate_metrics({"tp": 0, "tn": 5, "fp1": 0, "fp2": 0, "fn": 0})
    assert all(no_ball[key] == 0.0 for key in keys - {"accuracy"})
    assert no_ball["accuracy"] == 1.0


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
    assert metrics["tn"] == 0
    assert metrics["fp1"] == 0
    assert metrics["fp2"] == 0
    assert metrics["fn"] == 0
    assert metrics["f1"] == 1.0
    assert metrics["f1_strict"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_evaluate_epoch_counts_each_outcome():
    empty = torch.zeros(288, 512)
    gt_heatmaps = torch.stack(
        [
            torch.stack(
                [
                    _make_synthetic_heatmap(100, 150),
                    _make_synthetic_heatmap(200, 100),
                    empty,
                ]
            ),
            torch.stack([_make_synthetic_heatmap(300, 200), empty, empty]),
        ]
    )
    preds = torch.stack(
        [
            torch.stack(
                [
                    _make_synthetic_heatmap(101, 150),  # 1 px off -> tp
                    _make_synthetic_heatmap(250, 100),  # 50 px off -> fp1
                    _make_synthetic_heatmap(50, 50),  # no ball in gt -> fp2
                ]
            ),
            torch.stack([empty, empty, empty]),  # fn, tn, tn
        ]
    )
    dataset = TensorDataset(torch.randn(2, 9, 288, 512), gt_heatmaps)
    dataloader = DataLoader(dataset, batch_size=2)
    metrics = evaluate_epoch(
        PerfectModel(outputs=[preds]), dataloader, torch.device("cpu")
    )
    assert {k: metrics[k] for k in ("tp", "tn", "fp1", "fp2", "fn")} == {
        "tp": 1,
        "tn": 2,
        "fp1": 1,
        "fp2": 1,
        "fn": 1,
    }
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall"] == pytest.approx(1 / 2)
    assert metrics["f1"] == pytest.approx(0.4)
    assert metrics["recall_strict"] == pytest.approx(1 / 3)
    assert metrics["accuracy"] == pytest.approx(3 / 6)


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


def test_training_loss_finite_under_bf16_autocast():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        assert model.conv.bias is not None
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


def test_non_finite_loss_in_train_raises_and_writes_no_checkpoint():
    class NanFromStep3(nn.Module):
        """Finite for epoch 1 (steps 1-2), NaN from step 3 on."""

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, logits, target):
            self.calls += 1
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            return loss * float("nan") if self.calls > 2 else loss

    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_synthetic_dataset(num_samples=4)
        config = _make_test_config(tmpdir, epochs=3)
        trainer = Trainer(SimpleModel(), NanFromStep3(), ds, ds, config)
        with pytest.raises(FloatingPointError, match="step 3"):
            trainer.train()
        files = os.listdir(trainer.checkpoint_dir)
        assert "latest.pt" in files and set(files) <= {"latest.pt", "best.pt"}
        for name in files:  # all written by epoch 1, none from the NaN state
            ckpt = torch.load(
                os.path.join(trainer.checkpoint_dir, name), weights_only=True
            )
            assert (ckpt["epoch"], ckpt["global_step"]) == (1, 2), name


def test_resume_across_lr_milestone_restores_lr_and_schedule():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_synthetic_dataset(num_samples=4)
        config = _make_test_config(tmpdir, epochs=2) | {
            "lr_schedule": {"milestones": [1], "gamma": 0.1}
        }
        first = Trainer(SimpleModel(), SimpleLoss(), ds, ds, config)
        first.train()
        lr = first.optimizer.param_groups[0]["lr"]
        assert lr == pytest.approx(1e-4)  # 1e-3 decayed once, at the milestone

        second = Trainer(SimpleModel(), SimpleLoss(), ds, ds, config)
        assert second.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
        second.load_checkpoint(os.path.join(tmpdir, "checkpoints", "test", "latest.pt"))
        assert second.optimizer.param_groups[0]["lr"] == lr
        assert second.scheduler.last_epoch == first.scheduler.last_epoch == 2
        assert second.scheduler.get_last_lr() == first.scheduler.get_last_lr()


def test_float16_amp_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config(tmpdir) | {"amp_dtype": "float16"}
        ds = _make_synthetic_dataset()
        with pytest.raises(ValueError, match="float16"):
            Trainer(SimpleModel(), SimpleLoss(), ds, ds, config)


def test_latest_checkpoint_records_best_f1_of_its_epoch():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config(tmpdir, epochs=1, experiment_name="test_latest")
        trainer = Trainer(
            model=CentreBallModel(),
            loss_fn=SimpleLoss(),
            train_dataset=_make_synthetic_dataset(num_samples=4, h=32, w=64),
            val_dataset=_make_synthetic_dataset(num_samples=2, h=32, w=64),
            config=config,
        )
        trainer.train()
        latest = torch.load(
            os.path.join(tmpdir, "checkpoints", "test_latest", "latest.pt"),
            weights_only=True,
        )
        assert trainer.best_f1 == 1.0
        assert latest["best_f1"] == trainer.best_f1


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
        assert trainer2.load_checkpoint(ckpt_path) == config  # the stored config
        assert trainer2.current_epoch == original_epoch
        assert trainer2.best_f1 == original_best_f1
        for key in original_state:
            assert torch.equal(trainer2.model.state_dict()[key], original_state[key])


# torch.compile imports torch._dynamo, which warns about torch.jit on Python 3.14.
@pytest.mark.filterwarnings("ignore:`torch.jit.script_method`:DeprecationWarning")
def test_checkpoint_from_compiled_model_loads_into_plain_model():
    # torch.compile wraps lazily, so this needs no compiler.
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_synthetic_dataset()
        trainer = Trainer(
            torch.compile(SimpleModel()),
            SimpleLoss(),
            ds,
            ds,
            _make_test_config(tmpdir),
        )
        path = trainer._save_checkpoint("c.pt")
        ckpt = torch.load(path, weights_only=True)
        assert set(ckpt["model_state_dict"]) == set(SimpleModel().state_dict())
        load_model_weights(SimpleModel(), path, torch.device("cpu"))  # must not raise
        trainer.load_checkpoint(path)  # and resumes into the compiled wrapper


# Model.compile() imports torch._dynamo, which warns about torch.jit on Python 3.14.
@pytest.mark.filterwarnings("ignore:`torch.jit.script_method`:DeprecationWarning")
def test_checkpoint_from_model_compiled_in_place_has_plain_keys():
    # Module.compile() is lazy; no forward pass, so this needs no compiler.
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        model.compile()
        ds = _make_synthetic_dataset()
        trainer = Trainer(model, SimpleLoss(), ds, ds, _make_test_config(tmpdir))
        path = trainer._save_checkpoint("c.pt")
        ckpt = torch.load(path, weights_only=True)
        assert set(ckpt["model_state_dict"]) == set(SimpleModel().state_dict())


def test_max_steps_stops_early():
    # 4 samples at batch 2 = 2 steps per epoch; max_steps=3 stops mid-epoch 2.
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config(tmpdir, epochs=5) | {"max_steps": 3}
        ds = _make_synthetic_dataset(num_samples=4)
        trainer = Trainer(SimpleModel(), SimpleLoss(), ds, ds, config)
        trainer.train()
        assert trainer.global_step == 3
        # The epoch in progress still validated and checkpointed once.
        assert trainer.current_epoch == 2
        latest = torch.load(
            os.path.join(tmpdir, "checkpoints", "test", "latest.pt"),
            weights_only=True,
        )
        assert latest["epoch"] == 2
        assert latest["global_step"] == 3


# 4 samples at batch 2 = 2 steps per epoch: max_steps=3 stops in the middle of
# epoch 2, max_steps=4 at its last step.
@pytest.mark.parametrize(("max_steps", "complete"), [(3, False), (4, True)])
def test_checkpoint_records_whether_its_epoch_ran_to_the_end(max_steps, complete):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config(tmpdir, epochs=5) | {"max_steps": max_steps}
        ds = _make_synthetic_dataset(num_samples=4)
        Trainer(SimpleModel(), SimpleLoss(), ds, ds, config).train()
        latest = torch.load(
            os.path.join(tmpdir, "checkpoints", "test", "latest.pt"),
            weights_only=True,
        )
        assert latest["epoch"] == 2
        assert latest["epoch_complete"] is complete


def test_resume_from_cut_short_epoch_warns(caplog):
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_synthetic_dataset(num_samples=4)
        config = _make_test_config(tmpdir, epochs=5)
        Trainer(SimpleModel(), SimpleLoss(), ds, ds, config | {"max_steps": 3}).train()

        second = Trainer(SimpleModel(), SimpleLoss(), ds, ds, config)
        with caplog.at_level(logging.WARNING, logger="training.trainer"):
            second.load_checkpoint(
                os.path.join(tmpdir, "checkpoints", "test", "latest.pt")
            )
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1, warnings
        message = warnings[0]
        assert "epoch 2" in message and "step 3" in message, message
        assert "max_steps" in message and "epoch 3" in message, message
        # Resume itself is unchanged: it continues from the next epoch.
        assert (second.current_epoch, second.global_step) == (2, 3)


def test_resume_continues_from_next_epoch(caplog):
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_synthetic_dataset(num_samples=4)
        first = Trainer(
            SimpleModel(), SimpleLoss(), ds, ds, _make_test_config(tmpdir, epochs=1)
        )
        first.train()
        assert first.global_step == 2

        second = Trainer(
            SimpleModel(), SimpleLoss(), ds, ds, _make_test_config(tmpdir, epochs=2)
        )
        with caplog.at_level(logging.WARNING, logger="training.trainer"):
            second.load_checkpoint(
                os.path.join(tmpdir, "checkpoints", "test", "latest.pt")
            )
        assert not caplog.records  # the epoch ran to the end: nothing to warn about
        assert second.global_step == 2
        second.train()
        assert second.current_epoch == 2
        assert second.global_step == 4  # 2 restored + 2 new


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [({"epochs": 1}, "epochs"), ({"epochs": 5, "max_steps": 2}, "max_steps")],
)
def test_resume_with_nothing_left_to_do_says_why(caplog, overrides, reason):
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_synthetic_dataset(num_samples=4)
        config = _make_test_config(tmpdir, epochs=1)
        Trainer(SimpleModel(), SimpleLoss(), ds, ds, config).train()

        second = Trainer(SimpleModel(), SimpleLoss(), ds, ds, config | overrides)
        second.load_checkpoint(os.path.join(tmpdir, "checkpoints", "test", "latest.pt"))
        with caplog.at_level(logging.INFO, logger="training.trainer"):
            second.train()
        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 1 and "Nothing left to do" in messages[0], messages
        assert reason in messages[0], messages
        assert (second.current_epoch, second.global_step) == (1, 2)


@pytest.mark.parametrize(
    ("extra", "key_path", "type_name"),
    [
        ({"notes": datetime.date(2026, 11, 15)}, "notes", "date"),
        ({"data": {"roots": [pathlib.Path("datasets")]}}, "data.roots[0]", "Path"),
        ({"lr_by_epoch": {20: 1e-5}}, "lr_by_epoch", "int"),
    ],
)
def test_config_with_non_plain_value_is_rejected(extra, key_path, type_name):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config(tmpdir) | extra
        ds = _make_synthetic_dataset()
        with pytest.raises(ValueError, match=re.escape(key_path)) as err:
            Trainer(SimpleModel(), SimpleLoss(), ds, ds, config)
        assert type_name in str(err.value)
        # Rejected before any other work: no log or checkpoint directory.
        assert os.listdir(tmpdir) == []


def test_train_set_smaller_than_a_batch_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_synthetic_dataset(num_samples=1)
        with pytest.raises(ValueError, match="1 sample.*batch_size 2"):
            Trainer(SimpleModel(), SimpleLoss(), ds, ds, _make_test_config(tmpdir))


@pytest.mark.parametrize("log_every", [0, -5, 2.5, True])
def test_log_every_must_be_a_positive_int(log_every):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config(tmpdir) | {"log_every": log_every}
        ds = _make_synthetic_dataset()
        with pytest.raises(ValueError, match="log_every"):
            Trainer(SimpleModel(), SimpleLoss(), ds, ds, config)


@pytest.mark.parametrize(
    ("name", "optimizer_class"),
    [("AdamW", torch.optim.AdamW), ("Adam", torch.optim.Adam)],
)
def test_optimizer_is_built_from_config(name, optimizer_class):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config(tmpdir) | {"optimizer": name, "weight_decay": 0.05}
        ds = _make_synthetic_dataset()
        trainer = Trainer(SimpleModel(), SimpleLoss(), ds, ds, config)
        # Exact type: AdamW subclasses Adam.
        assert type(trainer.optimizer) is optimizer_class
        group = trainer.optimizer.param_groups[0]
        assert group["lr"] == config["learning_rate"]
        assert group["weight_decay"] == 0.05


def test_unknown_optimizer_is_rejected_before_any_file_is_written():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config(tmpdir) | {"optimizer": "SGD"}
        ds = _make_synthetic_dataset()
        with pytest.raises(ValueError, match="optimizer 'SGD'.*AdamW, Adam"):
            Trainer(SimpleModel(), SimpleLoss(), ds, ds, config)
        assert os.listdir(tmpdir) == []


def test_every_default_config_key_is_read_by_the_code():
    """A key nothing reads looks like a setting but changes nothing."""
    repo = pathlib.Path(__file__).parent.parent
    config = yaml.safe_load((repo / "configs" / "default.yaml").read_text())
    keys = set(config)
    for section in config.values():
        if isinstance(section, dict):
            keys |= set(section)
    source = (repo / "main.py").read_text() + "".join(
        path.read_text()
        for package in ("training", "data", "models", "inference", "utils")
        for path in (repo / package).rglob("*.py")
    )
    unread = sorted(key for key in keys if f'"{key}"' not in source)
    assert unread == []


def test_writer_closed_when_training_fails():
    class NanLoss(nn.Module):
        def forward(self, logits, target):
            return logits.sum() * float("nan")

    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_synthetic_dataset()
        trainer = Trainer(SimpleModel(), NanLoss(), ds, ds, _make_test_config(tmpdir))
        with pytest.raises(FloatingPointError):
            trainer.train()
        assert trainer.writer.all_writers is None  # SummaryWriter.close() ran


def test_checkpoint_holds_config_and_step():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config(tmpdir, epochs=1)
        ds = _make_synthetic_dataset(num_samples=4)
        Trainer(SimpleModel(), SimpleLoss(), ds, ds, config).train()
        path = os.path.join(tmpdir, "checkpoints", "test", "latest.pt")
        ckpt = torch.load(path, weights_only=True)
        assert ckpt["global_step"] == 2 and ckpt["config"]["batch_size"] == 2
        # The whole nested config round-trips under weights_only=True.
        assert ckpt["config"] == config


def test_checkpoint_write_leaves_no_temp_file(monkeypatch):
    saved_to = []
    real_save = torch.save

    def recording_save(obj, f, *args, **kwargs):
        saved_to.append(f)
        real_save(obj, f, *args, **kwargs)

    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_synthetic_dataset()
        trainer = Trainer(
            SimpleModel(), SimpleLoss(), ds, ds, _make_test_config(tmpdir)
        )
        monkeypatch.setattr("training.trainer.torch.save", recording_save)
        path = trainer._save_checkpoint("latest.pt")
        assert saved_to == [path + ".tmp"]  # written aside, then renamed
        assert os.listdir(trainer.checkpoint_dir) == ["latest.pt"]


def test_train_loader_drops_short_last_batch():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_synthetic_dataset(num_samples=5)
        trainer = Trainer(
            SimpleModel(), SimpleLoss(), ds, ds, _make_test_config(tmpdir)
        )
        assert len(trainer.train_loader) == 2


def test_step_loss_logged_every_log_every_steps(caplog):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config(tmpdir, epochs=2) | {"log_every": 1}
        ds = _make_synthetic_dataset(num_samples=4)
        trainer = Trainer(SimpleModel(), SimpleLoss(), ds, ds, config)
        with caplog.at_level("INFO", logger="training.trainer"):
            trainer.train()
        step_logs = [
            r.getMessage() for r in caplog.records if r.getMessage().startswith("Step ")
        ]
        assert len(step_logs) == 4  # 2 epochs x 2 steps, one line per step
        step_line = re.compile(
            r"Step (\d+) -- loss: \d\.\d{3}e[+-]\d\d, (\d+\.\d) steps/s"
            r"(, peak memory: \d+ MiB)?"
        )
        for step, message in enumerate(step_logs, start=1):
            match = step_line.fullmatch(message)
            assert match, message
            assert int(match[1]) == step and float(match[2]) > 0
            assert (match[3] is not None) == (trainer.device.type == "cuda")


def test_epoch_log_reports_train_and_validation_time(caplog):
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = _make_synthetic_dataset(num_samples=4)
        config = _make_test_config(tmpdir, epochs=2)
        with caplog.at_level("INFO", logger="training.trainer"):
            Trainer(SimpleModel(), SimpleLoss(), ds, ds, config).train()
        epoch_logs = [
            r.getMessage() for r in caplog.records if r.getMessage().startswith("Epoch")
        ]
        epoch_line = (
            r"Epoch {}/2 -- loss: \d\.\d{{3}}e[+-]\d\d, F1: \d\.\d{{4}}, "
            r"prec: \d\.\d{{4}}, rec: \d\.\d{{4}}, train \d+\.\d s, val \d+\.\d s"
        )
        assert len(epoch_logs) == 2
        for epoch, message in enumerate(epoch_logs, start=1):
            assert re.fullmatch(epoch_line.format(epoch), message), message


class TinyConstantLoss(nn.Module):
    """A loss of 1.23e-5 at every step, about where real training is by epoch 5."""

    def forward(self, logits, target):
        return logits.sum() * 0 + 1.23e-5


def test_small_loss_is_logged_in_scientific_notation(caplog):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config(tmpdir) | {"log_every": 1}
        ds = _make_synthetic_dataset(num_samples=4)
        with caplog.at_level("INFO", logger="training.trainer"):
            Trainer(SimpleModel(), TinyConstantLoss(), ds, ds, config).train()
        messages = [r.getMessage() for r in caplog.records]
        step_logs = [m for m in messages if m.startswith("Step ")]
        epoch_logs = [m for m in messages if m.startswith("Epoch ")]
        assert len(step_logs) == 2 and len(epoch_logs) == 1
        for message in step_logs + epoch_logs:
            assert "-- loss: 1.230e-05, " in message, message


def test_ensure_compile_supported_reports_failure(monkeypatch):
    def broken(_fn):
        raise RuntimeError("Failed to find C compiler")

    monkeypatch.setattr("training.trainer.torch.compile", broken)
    with pytest.raises(RuntimeError, match="build-essential"):
        ensure_compile_supported(torch.device("cpu"))


def test_ensure_compile_supported_names_the_fix_and_chains_cause(monkeypatch):
    cause = RuntimeError("Failed to find C compiler")

    def broken(_fn):
        raise cause

    monkeypatch.setattr("training.trainer.torch.compile", broken)
    with pytest.raises(RuntimeError, match="compile_model: false") as excinfo:
        ensure_compile_supported(torch.device("cpu"))
    assert excinfo.value.__cause__ is cause


def test_ensure_compile_supported_passes_when_compile_works(monkeypatch):
    monkeypatch.setattr("training.trainer.torch.compile", lambda fn: fn)
    ensure_compile_supported(torch.device("cpu"))  # no error


def test_load_model_weights_accepts_checkpoint_and_bare_state_dict(tmp_path):
    source = SimpleModel()
    torch.save({"epoch": 1, "model_state_dict": source.state_dict()}, tmp_path / "c.pt")
    torch.save(source.state_dict(), tmp_path / "bare.pt")
    for name in ("c.pt", "bare.pt"):
        target = SimpleModel()
        load_model_weights(target, tmp_path / name, torch.device("cpu"))
        for key, value in source.state_dict().items():
            assert torch.equal(target.state_dict()[key], value), (name, key)


def test_load_model_weights_wrong_architecture_raises(tmp_path):
    path = tmp_path / "c.pt"
    torch.save({"model_state_dict": SimpleModel().state_dict()}, path)
    with pytest.raises(RuntimeError, match="loading state_dict"):
        load_model_weights(nn.Conv2d(9, 3, 3), path, torch.device("cpu"))


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
