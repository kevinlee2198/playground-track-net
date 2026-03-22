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
