# TrackNet V2 Training Subsystem

**Feature Name:** Training loop, metrics, config, and checkpointing for TrackNet V2
**Goal:** Implement a complete, tested training pipeline that trains TrackNet V2 with AMP, TensorBoard logging, checkpoint management, and detection metrics (precision/recall/F1 at 4px threshold).
**Architecture:** Config-driven training loop with AdamW + MultiStepLR, bf16 mixed precision, torch.compile, and TensorBoard. Detection uses TP/FP/FN per frame at 4px Euclidean threshold.
**Tech Stack:** Python 3.12+, PyTorch 2.10+, PyYAML, TensorBoard, pytest

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement each task via TDD -- write failing test first, implement code to pass, verify, commit.

---

## File Map

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Default training hyperparameters (optimizer, lr, schedule, batch size, epochs, etc.) |
| `training/__init__.py` | Package init, exports Trainer and functions |
| `training/trainer.py` | Training loop: epoch iteration, AMP, optimizer step, LR scheduling, checkpointing, TensorBoard logging, seed setting |
| `training/evaluate.py` | Heatmap-to-position extraction, TP/FP/FN computation at 4px threshold, precision/recall/F1/accuracy aggregation |
| `tests/test_training.py` | Tests for config loading, metrics, trainer loop, checkpoint save/load |

---

## Dependencies on Other Subsystems

The training subsystem depends on interfaces from other subsystems. During development, we will create **minimal stubs/mocks** in tests so we can develop and test training independently.

- **Model:** `TrackNet(in_channels=9)` from `models/tracknet.py` -- an `nn.Module` that takes `(B, 9, 288, 512)` input and produces `(B, 3, 288, 512)` sigmoid output
- **Loss:** `WBCELoss()` from `models/losses.py` -- an `nn.Module` loss function taking `(predictions, targets)` both of shape `(B, 3, H, W)`
- **Data:** `TrackNetDataset` from `data/dataset.py` -- a PyTorch `Dataset` returning `(frames_tensor, heatmaps_tensor)` tuples
- **Config:** Loaded from YAML with PyYAML (`configs/default.yaml`)

---

## Task 1: Default Training Config (configs/default.yaml)

**Files:** `configs/default.yaml`, `tests/test_training.py` (config loading test)

**Steps:**

- [ ] **1a.** Write test for config loading from YAML

  `tests/test_training.py`:
  ```python
  import yaml
  import os

  def test_config_loads_from_yaml():
      config_path = os.path.join(
          os.path.dirname(__file__), "..", "configs", "default.yaml"
      )
      with open(config_path) as f:
          config = yaml.safe_load(f)

      # Verify all required keys exist with correct values
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
  ```

  Run: `uv run pytest tests/test_training.py::test_config_loads_from_yaml -x` (should FAIL -- file does not exist)

- [ ] **1b.** Create `configs/default.yaml`

  ```yaml
  # TrackNet V2 Training Configuration

  # Model
  in_channels: 9
  seq_len: 3
  input_size: [512, 288]  # [width, height]
  heatmap_radius: 30

  # Optimizer
  optimizer: AdamW
  learning_rate: 1.0e-4

  # LR Schedule
  lr_schedule:
    name: MultiStepLR
    milestones: [20, 25]
    gamma: 0.1

  # Training
  batch_size: 2
  epochs: 30
  seed: 42

  # DataLoader
  num_workers: 4
  pin_memory: true
  persistent_workers: true

  # Mixed Precision
  amp_dtype: bfloat16

  # torch.compile
  compile_model: true

  # Checkpointing
  checkpoint_dir: checkpoints
  experiment_name: default

  # TensorBoard
  log_dir: runs

  # Detection
  detection_threshold: 0.5
  distance_threshold: 4  # pixels, Euclidean distance for TP/FP/FN
  ```

  Run: `uv run pytest tests/test_training.py::test_config_loads_from_yaml -x` (should PASS)

- [ ] **1c.** Commit: `feat(training): add default training config and config loading test`

---

## Task 2: Core Detection Metrics (training/evaluate.py)

**Files:** `training/evaluate.py`, `training/__init__.py`, `tests/test_training.py`

**Steps:**

- [ ] **2a.** Write tests for detection metrics with synthetic data

  Add to `tests/test_training.py`:
  ```python
  import torch
  from training.evaluate import (
      heatmap_to_position,
      compute_detection_metrics,
      aggregate_metrics,
  )

  def test_heatmap_to_position_with_ball():
      """A heatmap with a bright spot should return the centroid position."""
      heatmap = torch.zeros(288, 512)
      # Place a ball-like blob at (x=100, y=150)
      for dy in range(-5, 6):
          for dx in range(-5, 6):
              if dx * dx + dy * dy <= 25:
                  heatmap[150 + dy, 100 + dx] = 0.9
      x, y, detected = heatmap_to_position(heatmap, threshold=0.5)
      assert detected is True
      assert abs(x - 100) < 2
      assert abs(y - 150) < 2

  def test_heatmap_to_position_no_ball():
      """An all-zero heatmap should return detected=False."""
      heatmap = torch.zeros(288, 512)
      x, y, detected = heatmap_to_position(heatmap, threshold=0.5)
      assert detected is False

  def test_compute_detection_metrics_true_positive():
      """Prediction within 4px of GT is a TP."""
      pred = (100.0, 150.0, True)
      gt = (102.0, 151.0, True)
      tp, fp, fn = compute_detection_metrics(pred, gt, distance_threshold=4)
      assert tp == 1 and fp == 0 and fn == 0

  def test_compute_detection_metrics_false_positive():
      """Prediction far from GT is FP + FN."""
      pred = (100.0, 150.0, True)
      gt = (200.0, 200.0, True)
      tp, fp, fn = compute_detection_metrics(pred, gt, distance_threshold=4)
      assert tp == 0 and fp == 1 and fn == 1

  def test_compute_detection_metrics_false_negative():
      """No prediction when GT exists is a FN."""
      pred = (0.0, 0.0, False)
      gt = (100.0, 150.0, True)
      tp, fp, fn = compute_detection_metrics(pred, gt, distance_threshold=4)
      assert tp == 0 and fp == 0 and fn == 1

  def test_compute_detection_metrics_true_negative():
      """No prediction and no GT ball => TN (all zeros)."""
      pred = (0.0, 0.0, False)
      gt = (0.0, 0.0, False)
      tp, fp, fn = compute_detection_metrics(pred, gt, distance_threshold=4)
      assert tp == 0 and fp == 0 and fn == 0

  def test_compute_detection_metrics_fp_no_gt():
      """Prediction when no GT ball => FP."""
      pred = (100.0, 150.0, True)
      gt = (0.0, 0.0, False)
      tp, fp, fn = compute_detection_metrics(pred, gt, distance_threshold=4)
      assert tp == 0 and fp == 1 and fn == 0

  def test_aggregate_metrics():
      """Test precision/recall/F1 aggregation."""
      # 7 TP, 2 FP, 1 FN
      metrics = aggregate_metrics(tp=7, fp=2, fn=1)
      assert abs(metrics["precision"] - 7 / 9) < 1e-6
      assert abs(metrics["recall"] - 7 / 8) < 1e-6
      expected_f1 = 2 * (7 / 9) * (7 / 8) / ((7 / 9) + (7 / 8))
      assert abs(metrics["f1"] - expected_f1) < 1e-6

  def test_aggregate_metrics_zero_division():
      """Zero TP should yield 0 precision/recall/F1 without errors."""
      metrics = aggregate_metrics(tp=0, fp=0, fn=0)
      assert metrics["precision"] == 0.0
      assert metrics["recall"] == 0.0
      assert metrics["f1"] == 0.0
  ```

  Run: `uv run pytest tests/test_training.py -k "test_heatmap or test_compute or test_aggregate" -x` (should FAIL -- module does not exist)

- [ ] **2b.** Create `training/__init__.py`

  ```python
  from training.evaluate import (
      heatmap_to_position,
      compute_detection_metrics,
      aggregate_metrics,
      evaluate_epoch,
  )
  from training.trainer import Trainer

  __all__ = [
      "heatmap_to_position",
      "compute_detection_metrics",
      "aggregate_metrics",
      "evaluate_epoch",
      "Trainer",
  ]
  ```

  Note: This will be created now but will fail to import until `trainer.py` exists. Create a placeholder in step 2c.

- [ ] **2c.** Create placeholder `training/trainer.py`

  ```python
  class Trainer:
      pass
  ```

- [ ] **2d.** Implement `training/evaluate.py`

  ```python
  from __future__ import annotations

  import math

  import torch


  def heatmap_to_position(
      heatmap: torch.Tensor, threshold: float = 0.5
  ) -> tuple[float, float, bool]:
      """Extract ball position from a single-frame heatmap.

      Args:
          heatmap: (H, W) tensor with values in [0, 1].
          threshold: Minimum value to consider as ball detection.

      Returns:
          (x, y, detected): x/y coordinates of the centroid and whether
          a ball was detected.
      """
      binary = (heatmap > threshold).float()
      if binary.sum() == 0:
          return 0.0, 0.0, False

      # Find centroid of all above-threshold pixels
      ys, xs = torch.where(binary > 0)
      x = xs.float().mean().item()
      y = ys.float().mean().item()
      return x, y, True


  def compute_detection_metrics(
      pred: tuple[float, float, bool],
      gt: tuple[float, float, bool],
      distance_threshold: float = 4.0,
  ) -> tuple[int, int, int]:
      """Compute TP/FP/FN for a single frame.

      Args:
          pred: (x, y, detected) predicted position.
          gt: (x, y, detected) ground truth position.
          distance_threshold: Maximum Euclidean distance for a true positive.

      Returns:
          (tp, fp, fn) counts.
      """
      pred_x, pred_y, pred_detected = pred
      gt_x, gt_y, gt_detected = gt

      if not pred_detected and not gt_detected:
          return 0, 0, 0
      if pred_detected and not gt_detected:
          return 0, 1, 0
      if not pred_detected and gt_detected:
          return 0, 0, 1

      # Both detected -- check distance
      dist = math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
      if dist <= distance_threshold:
          return 1, 0, 0  # True positive
      else:
          return 0, 1, 1  # Both FP and FN


  def aggregate_metrics(
      tp: int, fp: int, fn: int
  ) -> dict[str, float]:
      """Compute precision, recall, and F1 from TP/FP/FN counts."""
      precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
      recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
      f1 = (
          2 * precision * recall / (precision + recall)
          if (precision + recall) > 0
          else 0.0
      )
      return {"precision": precision, "recall": recall, "f1": f1}


  @torch.no_grad()
  def evaluate_epoch(
      model: torch.nn.Module,
      dataloader: torch.utils.data.DataLoader,
      device: torch.device,
      detection_threshold: float = 0.5,
      distance_threshold: float = 4.0,
  ) -> dict[str, float]:
      """Run full-epoch detection and return precision/recall/F1.

      Args:
          model: Trained model returning (B, 3, H, W) sigmoid heatmaps.
          dataloader: Val/test DataLoader yielding (frames, heatmaps).
          device: Device to run inference on.
          detection_threshold: Heatmap threshold for ball detection.
          distance_threshold: Euclidean pixel distance for TP matching.

      Returns:
          Dict with keys: precision, recall, f1, tp, fp, fn.
      """
      model.eval()
      total_tp, total_fp, total_fn = 0, 0, 0

      for frames, gt_heatmaps in dataloader:
          frames = frames.to(device)
          gt_heatmaps = gt_heatmaps.to(device)

          preds = model(frames)

          batch_size = preds.shape[0]
          num_frames = preds.shape[1]

          for b in range(batch_size):
              for f in range(num_frames):
                  pred_pos = heatmap_to_position(
                      preds[b, f], threshold=detection_threshold
                  )
                  gt_pos = heatmap_to_position(
                      gt_heatmaps[b, f], threshold=detection_threshold
                  )
                  tp, fp, fn = compute_detection_metrics(
                      pred_pos, gt_pos, distance_threshold=distance_threshold
                  )
                  total_tp += tp
                  total_fp += fp
                  total_fn += fn

      metrics = aggregate_metrics(total_tp, total_fp, total_fn)
      metrics["tp"] = total_tp
      metrics["fp"] = total_fp
      metrics["fn"] = total_fn
      return metrics
  ```

  Run: `uv run pytest tests/test_training.py -k "test_heatmap or test_compute or test_aggregate" -x` (should PASS)

- [ ] **2e.** Commit: `feat(training): add detection metrics with heatmap position extraction and TP/FP/FN`

---

## Task 3: evaluate_epoch Integration Test

**Files:** `tests/test_training.py`, `training/evaluate.py`

**Steps:**

- [ ] **3a.** Write test for `evaluate_epoch` with a synthetic model and dataloader

  Add to `tests/test_training.py`:
  ```python
  import torch.nn as nn
  from torch.utils.data import DataLoader, TensorDataset
  from training.evaluate import evaluate_epoch

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

  def test_evaluate_epoch_all_tp():
      """When model output matches GT, all detections should be TP."""
      gt1 = torch.stack([
          _make_synthetic_heatmap(100, 150),
          _make_synthetic_heatmap(200, 100),
          _make_synthetic_heatmap(300, 200),
      ])
      gt2 = torch.stack([
          _make_synthetic_heatmap(50, 50),
          _make_synthetic_heatmap(400, 250),
          _make_synthetic_heatmap(250, 144),
      ])

      frames = torch.randn(2, 9, 288, 512)
      gt_heatmaps = torch.stack([gt1, gt2])

      dataset = TensorDataset(frames, gt_heatmaps)
      dataloader = DataLoader(dataset, batch_size=2)

      model = PerfectModel(outputs=[gt_heatmaps])
      device = torch.device("cpu")

      metrics = evaluate_epoch(model, dataloader, device)
      assert metrics["tp"] == 6
      assert metrics["fp"] == 0
      assert metrics["fn"] == 0
      assert metrics["f1"] == 1.0
  ```

  Run: `uv run pytest tests/test_training.py::test_evaluate_epoch_all_tp -x` (should PASS)

- [ ] **3b.** Commit: `test(training): add evaluate_epoch integration test with synthetic model`

---

## Task 4: Training Loop (training/trainer.py)

**Files:** `training/trainer.py`, `tests/test_training.py`

**Steps:**

- [ ] **4a.** Write test for trainer running 1 epoch on tiny synthetic data (CPU)

  Add to `tests/test_training.py`:
  ```python
  import os
  import tempfile
  import yaml
  from training.trainer import Trainer

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

  def test_trainer_runs_one_epoch():
      """Trainer should complete 1 epoch without errors on CPU."""
      with tempfile.TemporaryDirectory() as tmpdir:
          config = {
              "optimizer": "AdamW",
              "learning_rate": 1e-3,
              "lr_schedule": {
                  "name": "MultiStepLR",
                  "milestones": [20, 25],
                  "gamma": 0.1,
              },
              "batch_size": 2,
              "epochs": 1,
              "input_size": [64, 32],
              "seq_len": 3,
              "seed": 42,
              "num_workers": 0,
              "pin_memory": False,
              "persistent_workers": False,
              "amp_dtype": "float32",  # No AMP on CPU
              "compile_model": False,
              "checkpoint_dir": os.path.join(tmpdir, "checkpoints"),
              "experiment_name": "test_run",
              "log_dir": os.path.join(tmpdir, "runs"),
              "detection_threshold": 0.5,
              "distance_threshold": 4,
          }

          model = SimpleModel()
          loss_fn = SimpleLoss()
          train_dataset = _make_synthetic_dataset(num_samples=4, h=32, w=64)
          val_dataset = _make_synthetic_dataset(num_samples=2, h=32, w=64)

          trainer = Trainer(
              model=model,
              loss_fn=loss_fn,
              train_dataset=train_dataset,
              val_dataset=val_dataset,
              config=config,
          )
          trainer.train()

          assert trainer.current_epoch == 1
  ```

  Run: `uv run pytest tests/test_training.py::test_trainer_runs_one_epoch -x` (should FAIL -- Trainer is a stub)

- [ ] **4b.** Implement `training/trainer.py`

  ```python
  from __future__ import annotations

  import logging
  import os
  import random

  import numpy as np
  import torch
  import torch.nn as nn
  from torch.utils.data import DataLoader, Dataset
  from torch.utils.tensorboard import SummaryWriter

  from training.evaluate import evaluate_epoch

  logger = logging.getLogger(__name__)


  class Trainer:
      """Training loop for TrackNet with AMP, logging, and checkpointing."""

      def __init__(
          self,
          model: nn.Module,
          loss_fn: nn.Module,
          train_dataset: Dataset,
          val_dataset: Dataset,
          config: dict,
      ) -> None:
          self.config = config
          self.device = torch.device(
              "cuda" if torch.cuda.is_available() else "cpu"
          )

          # Reproducibility
          self._set_seed(config["seed"])

          # Model
          self.model = model.to(self.device)
          if config.get("compile_model", False) and self.device.type == "cuda":
              self.model = torch.compile(self.model)

          # Loss
          self.loss_fn = loss_fn.to(self.device)

          # DataLoaders
          loader_workers = config.get("num_workers", 4)
          use_persistent = (
              config.get("persistent_workers", True)
              if loader_workers > 0
              else False
          )
          self.train_loader = DataLoader(
              train_dataset,
              batch_size=config["batch_size"],
              shuffle=True,
              num_workers=loader_workers,
              pin_memory=config.get("pin_memory", True),
              persistent_workers=use_persistent,
          )
          self.val_loader = DataLoader(
              val_dataset,
              batch_size=config["batch_size"],
              shuffle=False,
              num_workers=loader_workers,
              pin_memory=config.get("pin_memory", True),
              persistent_workers=use_persistent,
          )

          # Optimizer
          self.optimizer = torch.optim.AdamW(
              self.model.parameters(), lr=config["learning_rate"]
          )

          # LR Scheduler
          sched_config = config["lr_schedule"]
          self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
              self.optimizer,
              milestones=sched_config["milestones"],
              gamma=sched_config["gamma"],
          )

          # Mixed precision
          amp_dtype_str = config.get("amp_dtype", "bfloat16")
          if amp_dtype_str == "bfloat16":
              self.amp_dtype = torch.bfloat16
          elif amp_dtype_str == "float16":
              self.amp_dtype = torch.float16
          else:
              self.amp_dtype = None

          # TensorBoard
          log_dir = os.path.join(
              config["log_dir"], config.get("experiment_name", "default")
          )
          self.writer = SummaryWriter(log_dir=log_dir)

          # Checkpointing
          self.checkpoint_dir = os.path.join(
              config["checkpoint_dir"],
              config.get("experiment_name", "default"),
          )
          os.makedirs(self.checkpoint_dir, exist_ok=True)

          # State
          self.current_epoch = 0
          self.best_f1 = 0.0

      def _set_seed(self, seed: int) -> None:
          random.seed(seed)
          np.random.seed(seed)
          torch.manual_seed(seed)
          if torch.cuda.is_available():
              torch.cuda.manual_seed_all(seed)

      def _train_one_epoch(self) -> float:
          """Run one training epoch. Returns average loss."""
          self.model.train()
          total_loss = 0.0
          num_batches = 0

          for frames, heatmaps in self.train_loader:
              frames = frames.to(self.device)
              heatmaps = heatmaps.to(self.device)

              self.optimizer.zero_grad(set_to_none=True)

              if self.amp_dtype is not None:
                  with torch.amp.autocast(
                      self.device.type, dtype=self.amp_dtype
                  ):
                      preds = self.model(frames)
                      loss = self.loss_fn(preds, heatmaps)
              else:
                  preds = self.model(frames)
                  loss = self.loss_fn(preds, heatmaps)

              loss.backward()
              self.optimizer.step()

              total_loss += loss.item()
              num_batches += 1

          return total_loss / max(num_batches, 1)

      def _save_checkpoint(self, filename: str) -> str:
          """Save model checkpoint. Returns the file path."""
          path = os.path.join(self.checkpoint_dir, filename)
          torch.save(
              {
                  "epoch": self.current_epoch,
                  "model_state_dict": self.model.state_dict(),
                  "optimizer_state_dict": self.optimizer.state_dict(),
                  "scheduler_state_dict": self.scheduler.state_dict(),
                  "best_f1": self.best_f1,
              },
              path,
          )
          return path

      def load_checkpoint(self, path: str) -> None:
          """Load a checkpoint from disk."""
          checkpoint = torch.load(
              path, map_location=self.device, weights_only=True
          )
          self.model.load_state_dict(checkpoint["model_state_dict"])
          self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
          self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
          self.current_epoch = checkpoint["epoch"]
          self.best_f1 = checkpoint["best_f1"]

      def train(self) -> None:
          """Run the full training loop."""
          epochs = self.config["epochs"]

          for epoch in range(self.current_epoch, epochs):
              self.current_epoch = epoch + 1

              # Train
              train_loss = self._train_one_epoch()

              # Validate
              val_metrics = evaluate_epoch(
                  self.model,
                  self.val_loader,
                  self.device,
                  detection_threshold=self.config.get(
                      "detection_threshold", 0.5
                  ),
                  distance_threshold=self.config.get(
                      "distance_threshold", 4.0
                  ),
              )

              # LR step
              self.scheduler.step()

              # Log to TensorBoard
              self.writer.add_scalar(
                  "Loss/train", train_loss, self.current_epoch
              )
              self.writer.add_scalar(
                  "Metrics/precision",
                  val_metrics["precision"],
                  self.current_epoch,
              )
              self.writer.add_scalar(
                  "Metrics/recall",
                  val_metrics["recall"],
                  self.current_epoch,
              )
              self.writer.add_scalar(
                  "Metrics/f1", val_metrics["f1"], self.current_epoch
              )
              self.writer.add_scalar(
                  "LR",
                  self.scheduler.get_last_lr()[0],
                  self.current_epoch,
              )

              logger.info(
                  "Epoch %d/%d -- loss: %.4f, F1: %.4f, prec: %.4f, rec: %.4f",
                  self.current_epoch,
                  epochs,
                  train_loss,
                  val_metrics["f1"],
                  val_metrics["precision"],
                  val_metrics["recall"],
              )

              # Save latest checkpoint
              self._save_checkpoint("latest.pt")

              # Save best checkpoint
              if val_metrics["f1"] > self.best_f1:
                  self.best_f1 = val_metrics["f1"]
                  self._save_checkpoint("best.pt")

          self.writer.close()
  ```

  Run: `uv run pytest tests/test_training.py::test_trainer_runs_one_epoch -x` (should PASS)

- [ ] **4c.** Commit: `feat(training): implement training loop with AMP, LR scheduling, TensorBoard, and checkpointing`

---

## Task 5: Checkpoint Save/Load Round-Trip Test

**Files:** `tests/test_training.py`

**Steps:**

- [ ] **5a.** Write checkpoint round-trip test

  Add to `tests/test_training.py`:
  ```python
  def test_checkpoint_save_load_roundtrip():
      """Saving and loading a checkpoint should restore trainer state."""
      with tempfile.TemporaryDirectory() as tmpdir:
          config = {
              "optimizer": "AdamW",
              "learning_rate": 1e-3,
              "lr_schedule": {
                  "name": "MultiStepLR",
                  "milestones": [20, 25],
                  "gamma": 0.1,
              },
              "batch_size": 2,
              "epochs": 2,
              "input_size": [64, 32],
              "seq_len": 3,
              "seed": 42,
              "num_workers": 0,
              "pin_memory": False,
              "persistent_workers": False,
              "amp_dtype": "float32",
              "compile_model": False,
              "checkpoint_dir": os.path.join(tmpdir, "checkpoints"),
              "experiment_name": "test_ckpt",
              "log_dir": os.path.join(tmpdir, "runs"),
              "detection_threshold": 0.5,
              "distance_threshold": 4,
          }

          model = SimpleModel()
          loss_fn = SimpleLoss()
          train_ds = _make_synthetic_dataset(num_samples=4, h=32, w=64)
          val_ds = _make_synthetic_dataset(num_samples=2, h=32, w=64)

          trainer = Trainer(
              model=model,
              loss_fn=loss_fn,
              train_dataset=train_ds,
              val_dataset=val_ds,
              config=config,
          )
          trainer.train()

          # Save state for comparison
          original_epoch = trainer.current_epoch
          original_best_f1 = trainer.best_f1
          original_state = {
              k: v.clone()
              for k, v in trainer.model.state_dict().items()
          }

          # Load into a fresh trainer
          model2 = SimpleModel()
          trainer2 = Trainer(
              model=model2,
              loss_fn=SimpleLoss(),
              train_dataset=train_ds,
              val_dataset=val_ds,
              config=config,
          )
          ckpt_path = os.path.join(
              tmpdir, "checkpoints", "test_ckpt", "latest.pt"
          )
          trainer2.load_checkpoint(ckpt_path)

          assert trainer2.current_epoch == original_epoch
          assert trainer2.best_f1 == original_best_f1
          for key in original_state:
              assert torch.equal(
                  trainer2.model.state_dict()[key], original_state[key]
              )
  ```

  Run: `uv run pytest tests/test_training.py::test_checkpoint_save_load_roundtrip -x` (should PASS)

- [ ] **5b.** Commit: `test(training): add checkpoint save/load round-trip test`

---

## Task 6: Package Exports and TensorBoard Verification

**Files:** `training/__init__.py`, `tests/test_training.py`

**Steps:**

- [ ] **6a.** Write test for clean imports and TensorBoard event creation

  Add to `tests/test_training.py`:
  ```python
  import glob as glob_module

  def test_training_package_imports():
      """All public APIs should be importable from the training package."""
      from training import (
          Trainer,
          heatmap_to_position,
          compute_detection_metrics,
          aggregate_metrics,
          evaluate_epoch,
      )
      assert Trainer is not None
      assert heatmap_to_position is not None
      assert compute_detection_metrics is not None
      assert aggregate_metrics is not None
      assert evaluate_epoch is not None

  def test_tensorboard_logs_created():
      """Training should create TensorBoard event files."""
      with tempfile.TemporaryDirectory() as tmpdir:
          config = {
              "optimizer": "AdamW",
              "learning_rate": 1e-3,
              "lr_schedule": {
                  "name": "MultiStepLR",
                  "milestones": [20, 25],
                  "gamma": 0.1,
              },
              "batch_size": 2,
              "epochs": 1,
              "input_size": [64, 32],
              "seq_len": 3,
              "seed": 42,
              "num_workers": 0,
              "pin_memory": False,
              "persistent_workers": False,
              "amp_dtype": "float32",
              "compile_model": False,
              "checkpoint_dir": os.path.join(tmpdir, "checkpoints"),
              "experiment_name": "test_tb",
              "log_dir": os.path.join(tmpdir, "runs"),
              "detection_threshold": 0.5,
              "distance_threshold": 4,
          }

          model = SimpleModel()
          loss_fn = SimpleLoss()
          train_ds = _make_synthetic_dataset(num_samples=4, h=32, w=64)
          val_ds = _make_synthetic_dataset(num_samples=2, h=32, w=64)

          trainer = Trainer(
              model=model,
              loss_fn=loss_fn,
              train_dataset=train_ds,
              val_dataset=val_ds,
              config=config,
          )
          trainer.train()

          log_dir = os.path.join(tmpdir, "runs", "test_tb")
          event_files = glob_module.glob(
              os.path.join(log_dir, "events.out.tfevents.*")
          )
          assert len(event_files) > 0
  ```

  Run: `uv run pytest tests/test_training.py -k "test_training_package or test_tensorboard" -x` (should PASS)

- [ ] **6b.** Run full test suite

  Run: `uv run pytest tests/test_training.py -v` (ALL tests should PASS)

- [ ] **6c.** Commit: `test(training): add package import and TensorBoard verification tests`

---

## Summary of Deliverables

After all tasks are complete:

| File | Contents |
|------|----------|
| `configs/default.yaml` | All hyperparameters from spec: AdamW, lr=1e-4, MultiStepLR(20,25), batch=2, epochs=30, bf16, torch.compile, seed=42 |
| `training/__init__.py` | Exports: Trainer, heatmap_to_position, compute_detection_metrics, aggregate_metrics, evaluate_epoch |
| `training/trainer.py` | Full training loop: AMP (bf16), torch.compile, AdamW+MultiStepLR, zero_grad(set_to_none=True), pin_memory, persistent_workers, TensorBoard (loss/F1/precision/recall/LR per epoch), checkpoint save (best by val F1 + latest), seed setting |
| `training/evaluate.py` | heatmap_to_position (threshold+centroid), compute_detection_metrics (4px Euclidean TP/FP/FN), aggregate_metrics (precision/recall/F1), evaluate_epoch (full dataloader pass) |
| `tests/test_training.py` | 13 tests: config YAML loading, heatmap position (ball/no-ball), detection metrics (TP/FP/FN/TN/FP-no-GT), aggregate metrics (normal/zero-division), evaluate_epoch integration, trainer 1-epoch, checkpoint round-trip, package imports, TensorBoard events |

All tests run on CPU with small synthetic data (no GPU required).
