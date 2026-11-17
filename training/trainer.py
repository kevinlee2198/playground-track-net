from __future__ import annotations

import logging
import math
import os
import random
import time
from collections.abc import Sized
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from data import mixup_batch
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


# Scalar types (besides dict and list) a weights_only=True load restores.
_PLAIN_TYPES = (str, int, float, bool, type(None))


def _check_plain_config(value: object, path: str = "") -> None:
    """Check that ``value`` holds only dicts with str keys, lists, str, int,
    float, bool and None, at any depth.

    Raises:
        ValueError: Naming the key path (e.g. ``data.roots[0]``) and the type
            of the first value or key that is anything else.
    """
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(
                    f"config key {key!r} under {path or 'the top level'} is "
                    f"{type(key).__name__}, not str; checkpoints store the config "
                    "and load it with weights_only=True"
                )
            _check_plain_config(item, f"{path}.{key}" if path else key)
    elif type(value) is list:
        for i, item in enumerate(value):
            _check_plain_config(item, f"{path}[{i}]")
    elif type(value) not in _PLAIN_TYPES:
        raise ValueError(
            f"config value at {path} is {type(value).__name__}; checkpoints "
            "store the config and load it with weights_only=True, so it may "
            "hold only dict, list, str, int, float, bool and None"
        )


_OPTIMIZERS: dict[str, type[torch.optim.Adam]] = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
}


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
        """
        Raises:
            ValueError: If ``config`` holds a value a ``weights_only`` load
                cannot restore, ``log_every`` is not a positive int,
                ``mixup_alpha`` is not a finite number >= 0, the train set is
                smaller than one batch, or ``optimizer`` is not ``AdamW`` or
                ``Adam``.
        """
        _check_plain_config(config)
        self.config = config

        # Run length and step logging
        self.max_steps: int | None = config.get("max_steps")
        self.log_every: int = config.get("log_every", 100)
        if type(self.log_every) is not int or self.log_every <= 0:  # not bool either
            raise ValueError(
                f"log_every must be a positive int, got {self.log_every!r}"
            )

        # Sample mixup on training batches; 0 turns it off. Checked with the
        # other settings, before the log and checkpoint directories exist.
        self.mixup_alpha: float = config["mixup_alpha"]
        if type(self.mixup_alpha) not in (int, float) or not (
            math.isfinite(self.mixup_alpha) and self.mixup_alpha >= 0
        ):
            raise ValueError(
                "mixup_alpha must be a finite number >= 0 (0 turns mixup off), "
                f"got {self.mixup_alpha!r}"
            )

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
        if len(self.train_loader) == 0:
            raise ValueError(
                f"the train set has {len(cast(Sized, train_dataset))} samples, "
                f"fewer than batch_size {config['batch_size']}: with drop_last "
                "no batch would be trained"
            )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=loader_workers,
            pin_memory=config.get("pin_memory", True),
            persistent_workers=use_persistent,
        )

        # Optimizer. weight_decay is decoupled for AdamW and an L2 penalty for Adam.
        optimizer_name = config["optimizer"]
        if optimizer_name not in _OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer {optimizer_name!r}: use one of "
                f"{', '.join(_OPTIMIZERS)}"
            )
        self.optimizer = _OPTIMIZERS[optimizer_name](
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
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

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_f1 = 0.0
        self.epoch_complete = True  # whether current_epoch ran to its last batch

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

        The model must accept ``model(frames, return_logits=True)``. With
        ``mixup_alpha`` above 0 each batch is mixed by ``mixup_batch`` on the
        device before the forward pass. Only the forward pass runs under
        autocast; the loss is computed from float32 logits outside it. Stops
        early once ``max_steps`` optimizer steps have been taken in total, and
        sets ``epoch_complete`` to whether every batch of the epoch was
        trained.

        Raises:
            FloatingPointError: If the loss at any step is NaN or infinite.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        window_loss = 0.0
        window_steps = 0
        window_start = time.perf_counter()

        for frames, heatmaps in self.train_loader:
            frames = frames.to(self.device)
            heatmaps = heatmaps.to(self.device)
            if self.mixup_alpha > 0:
                frames, heatmaps = mixup_batch(frames, heatmaps, self.mixup_alpha)

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
                now = time.perf_counter()
                self._log_step(
                    window_loss / window_steps, window_steps / (now - window_start)
                )
                window_loss = 0.0
                window_steps = 0
                window_start = now

            if self._reached_max_steps():
                break

        self.epoch_complete = num_batches == len(self.train_loader)
        return total_loss / max(num_batches, 1)

    def _log_step(self, mean_loss: float, steps_per_s: float) -> None:
        """Log the mean loss and throughput of the steps in the current window.

        The window runs from the last step log to this one, and restarts at
        each epoch: the first log of an epoch covers only that epoch's steps
        (and includes the data loader's start-up).
        """
        self.writer.add_scalar("Loss/train_step", mean_loss, self.global_step)
        if self.device.type == "cuda":
            peak_mib = torch.cuda.max_memory_allocated(self.device) / 2**20
            logger.info(
                "Step %d -- loss: %.3e, %.1f steps/s, peak memory: %.0f MiB",
                self.global_step,
                mean_loss,
                steps_per_s,
                peak_mib,
            )
        else:
            logger.info(
                "Step %d -- loss: %.3e, %.1f steps/s",
                self.global_step,
                mean_loss,
                steps_per_s,
            )

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
                "epoch_complete": self.epoch_complete,
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

    def load_checkpoint(self, path: str) -> dict:
        """Load a checkpoint from disk, so ``train()`` resumes from the next
        epoch. ``max_steps`` still counts the restored steps.

        Logs a warning if the checkpoint's epoch was cut short by
        ``max_steps``: the rest of that epoch's data is not trained.

        Returns:
            The config the checkpoint was saved with.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self._unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_f1 = checkpoint["best_f1"]
        self.epoch_complete = checkpoint.get("epoch_complete", True)
        if not self.epoch_complete:
            logger.warning(
                "%s: epoch %d was cut short at step %d by max_steps; the resumed "
                "run continues from epoch %d without the rest of that epoch's data",
                path,
                self.current_epoch,
                self.global_step,
                self.current_epoch + 1,
            )
        return checkpoint["config"]

    def train(self) -> None:
        """Run the full training loop.

        Stops after ``epochs`` epochs or, if set, once ``max_steps`` optimizer
        steps have been taken; the epoch in which the limit is reached still
        validates and checkpoints. The TensorBoard writer is closed however
        the loop ends.
        """
        try:
            self._train_epochs()
        finally:
            self.writer.close()

    def _train_epochs(self) -> None:
        epochs = self.config["epochs"]
        if self._reached_max_steps():
            logger.info(
                "Nothing left to do: global_step %d has reached max_steps %d",
                self.global_step,
                self.max_steps,
            )
            return
        if self.current_epoch >= epochs:
            logger.info(
                "Nothing left to do: epoch %d of %d epochs is already done",
                self.current_epoch,
                epochs,
            )
            return

        for epoch in range(self.current_epoch, epochs):
            if self._reached_max_steps():
                break
            self.current_epoch = epoch + 1

            # Train
            train_start = time.perf_counter()
            train_loss = self._train_one_epoch()
            train_seconds = time.perf_counter() - train_start

            # Validate
            val_start = time.perf_counter()
            val_metrics = evaluate_epoch(
                self.model,
                self.val_loader,
                self.device,
                detection_threshold=self.config.get("detection_threshold", 0.5),
                distance_threshold=self.config.get("distance_threshold", 4.0),
            )
            val_seconds = time.perf_counter() - val_start

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
                "Epoch %d/%d -- loss: %.3e, F1: %.4f, prec: %.4f, rec: %.4f, "
                "train %.1f s, val %.1f s",
                self.current_epoch,
                epochs,
                train_loss,
                val_metrics["f1"],
                val_metrics["precision"],
                val_metrics["recall"],
                train_seconds,
                val_seconds,
            )

            # Save best checkpoint first, so latest.pt records the updated best_f1
            if val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                self._save_checkpoint("best.pt")

            # Save latest checkpoint
            self._save_checkpoint("latest.pt")
