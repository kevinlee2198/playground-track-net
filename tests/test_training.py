import glob as glob_module
import math
import os
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
        trainer2.load_checkpoint(ckpt_path)
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


def test_resume_continues_from_next_epoch():
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
        second.load_checkpoint(os.path.join(tmpdir, "checkpoints", "test", "latest.pt"))
        assert second.global_step == 2
        second.train()
        assert second.current_epoch == 2
        assert second.global_step == 4  # 2 restored + 2 new


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
        step_logs = [r for r in caplog.records if r.getMessage().startswith("Step ")]
        assert len(step_logs) == 4  # 2 epochs x 2 steps, one line per step


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
