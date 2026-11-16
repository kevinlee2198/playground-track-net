from __future__ import annotations

import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from training.evaluate import evaluate_epoch

logger = logging.getLogger(__name__)


def ensure_compile_supported(device: torch.device) -> None:
    """Compile and run a trivial function on ``device`` to check that
    ``torch.compile`` works here.

    Compilation is lazy, so the probe has to be called: a missing C compiler
    only shows up on the first compiled call.

    Raises:
        RuntimeError: If compiling or running the probe fails, chained from
            the original error.
    """

    def probe(x: torch.Tensor) -> torch.Tensor:
        return x * 2 + 1

    try:
        torch.compile(probe)(torch.ones(4, device=device))
    except Exception as err:
        raise RuntimeError(
            f"torch.compile does not work on this machine ({type(err).__name__}). "
            "Install a C compiler (on Debian/Ubuntu: sudo apt install "
            "build-essential) or set compile_model: false in the config."
        ) from err


def experiment_dir(config: dict, base_key: str) -> str:
    """``config[base_key]`` joined with the experiment name ("default" if unset).

    ``base_key`` is ``"checkpoint_dir"`` or ``"log_dir"``.
    """
    return os.path.join(config[base_key], config.get("experiment_name", "default"))


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reproducibility
        self._set_seed(config["seed"])

        # Model. Compile in place so state_dict keys stay unprefixed.
        self.model = model.to(self.device)
        if config.get("compile_model", False) and self.device.type == "cuda":
            ensure_compile_supported(self.device)
            self.model.compile()

        # Loss
        self.loss_fn = loss_fn.to(self.device)

        # DataLoaders
        loader_workers = config.get("num_workers", 4)
        use_persistent = (
            config.get("persistent_workers", True) if loader_workers > 0 else False
        )
        # drop_last: a short final batch would make torch.compile recompile.
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
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
        elif amp_dtype_str == "float32":
            self.amp_dtype = None
        else:
            raise ValueError(
                f"Unsupported amp_dtype {amp_dtype_str!r}: use 'bfloat16' or "
                "'float32' (float16 needs a GradScaler, which Trainer does not use)"
            )

        # TensorBoard
        self.writer = SummaryWriter(log_dir=experiment_dir(config, "log_dir"))

        # Checkpointing
        self.checkpoint_dir = experiment_dir(config, "checkpoint_dir")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Run length and step logging
        self.max_steps: int | None = config.get("max_steps")
        self.log_every: int = config.get("log_every", 100)

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_f1 = 0.0

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _reached_max_steps(self) -> bool:
        return self.max_steps is not None and self.global_step >= self.max_steps

    def _train_one_epoch(self) -> float:
        """Run one training epoch. Returns average loss.

        The model must accept ``model(frames, return_logits=True)``. Only the
        forward pass runs under autocast; the loss is computed from float32
        logits outside it. Stops early once ``max_steps`` optimizer steps have
        been taken in total.

        Raises:
            FloatingPointError: If the loss at any step is NaN or infinite.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        window_loss = 0.0
        window_steps = 0

        for frames, heatmaps in self.train_loader:
            frames = frames.to(self.device)
            heatmaps = heatmaps.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None,
            ):
                logits = self.model(frames, return_logits=True)
            loss = self.loss_fn(logits.float(), heatmaps)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at step {self.global_step + 1}"
                )

            loss.backward()
            self.optimizer.step()
            self.global_step += 1

            step_loss = loss.item()
            total_loss += step_loss
            num_batches += 1
            window_loss += step_loss
            window_steps += 1

            if self.global_step % self.log_every == 0:
                self._log_step(window_loss / window_steps)
                window_loss = 0.0
                window_steps = 0

            if self._reached_max_steps():
                break

        return total_loss / max(num_batches, 1)

    def _log_step(self, mean_loss: float) -> None:
        """Log the mean loss of the steps since the last step log."""
        self.writer.add_scalar("Loss/train_step", mean_loss, self.global_step)
        if self.device.type == "cuda":
            peak_mib = torch.cuda.max_memory_allocated(self.device) / 2**20
            logger.info(
                "Step %d -- loss: %.4f, peak memory: %.0f MiB",
                self.global_step,
                mean_loss,
                peak_mib,
            )
        else:
            logger.info("Step %d -- loss: %.4f", self.global_step, mean_loss)

    @property
    def _unwrapped_model(self) -> nn.Module:
        """The model without a ``torch.compile`` wrapper, whose state_dict keys
        carry no ``_orig_mod.`` prefix."""
        return getattr(self.model, "_orig_mod", self.model)

    def _save_checkpoint(self, filename: str) -> str:
        """Save a checkpoint atomically. Returns the file path.

        Writes to ``<path>.tmp`` and renames it over ``path``, so an
        interrupted save never leaves a truncated checkpoint behind.
        """
        path = os.path.join(self.checkpoint_dir, filename)
        tmp_path = path + ".tmp"
        torch.save(
            {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "model_state_dict": self._unwrapped_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_f1": self.best_f1,
                "config": self.config,
            },
            tmp_path,
        )
        os.replace(tmp_path, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint from disk, so ``train()`` resumes from the next
        epoch. ``max_steps`` still counts the restored steps."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self._unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_f1 = checkpoint["best_f1"]

    def train(self) -> None:
        """Run the full training loop.

        Stops after ``epochs`` epochs or, if set, once ``max_steps`` optimizer
        steps have been taken; the epoch in which the limit is reached still
        validates and checkpoints.
        """
        epochs = self.config["epochs"]

        for epoch in range(self.current_epoch, epochs):
            if self._reached_max_steps():
                break
            self.current_epoch = epoch + 1
            epoch_start = time.perf_counter()

            # Train
            train_loss = self._train_one_epoch()

            # Validate
            val_metrics = evaluate_epoch(
                self.model,
                self.val_loader,
                self.device,
                detection_threshold=self.config.get("detection_threshold", 0.5),
                distance_threshold=self.config.get("distance_threshold", 4.0),
            )

            # LR step
            self.scheduler.step()

            # Log to TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, self.current_epoch)
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
            self.writer.add_scalar("Metrics/f1", val_metrics["f1"], self.current_epoch)
            self.writer.add_scalar(
                "Metrics/f1_strict", val_metrics["f1_strict"], self.current_epoch
            )
            self.writer.add_scalar(
                "Metrics/accuracy", val_metrics["accuracy"], self.current_epoch
            )
            self.writer.add_scalar(
                "LR",
                self.scheduler.get_last_lr()[0],
                self.current_epoch,
            )

            logger.info(
                "Epoch %d/%d -- loss: %.4f, F1: %.4f, prec: %.4f, rec: %.4f, %.1f s",
                self.current_epoch,
                epochs,
                train_loss,
                val_metrics["f1"],
                val_metrics["precision"],
                val_metrics["recall"],
                time.perf_counter() - epoch_start,
            )

            # Save best checkpoint first, so latest.pt records the updated best_f1
            if val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                self._save_checkpoint("best.pt")

            # Save latest checkpoint
            self._save_checkpoint("latest.pt")

        self.writer.close()
