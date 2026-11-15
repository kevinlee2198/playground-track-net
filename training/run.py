"""Train or evaluate a TrackNet model on the dataset a config names."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from data import (
    Compose,
    FrameColorJitter,
    HorizontalFlip,
    TrackNetDataset,
    build_dataset,
    discover_tennis_clips,
    split_clips,
)
from models import TrackNet, WBCEFocalLoss, load_model_weights
from training.evaluate import OUTCOMES, evaluate_epoch
from training.trainer import Trainer

logger = logging.getLogger(__name__)

EVAL_STRIDE = 3  # validation and test score each frame once
DISTANCE_PX = 4.0  # true-positive distance of both evaluation protocols


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_datasets(config: dict, splits: Sequence[str]) -> dict[str, ConcatDataset]:
    """Datasets for the named splits ("train", "val", "test") of ``config["data"]``.

    Train uses ``train_stride`` and flip + colour-jitter augmentation; val and
    test use stride 3 and no augmentation.
    """
    data = config["data"]
    if data["format"] != "tennis":
        raise ValueError(
            f"unknown data.format {data['format']!r}: only 'tennis' is supported"
        )
    clips = split_clips(
        discover_tennis_clips(data["root"]), data["val_groups"], data["test_groups"]
    )
    width, height = config["input_size"]
    datasets = {}
    for split in splits:
        if split == "train":
            stride = config["train_stride"]
            transform = Compose([HorizontalFlip(), FrameColorJitter()])
        else:
            stride, transform = EVAL_STRIDE, None
        datasets[split] = build_dataset(
            clips[split],
            stride=stride,
            radius=config["heatmap_radius"],
            height=height,
            width=width,
            transform=transform,
        )
        logger.info(
            "%s: %d clips, %d windows", split, len(clips[split]), len(datasets[split])
        )
    return datasets


def run_training(config: dict, resume: str | Path | None = None) -> None:
    """Train ``TrackNet()`` on the train split, validating on the val split.

    ``config`` must hold only plain YAML types: the trainer stores it in every
    checkpoint. With ``resume``, training continues from that checkpoint.
    """
    datasets = _build_datasets(config, ("train", "val"))
    # The model is initialised before the Trainer seeds, so seed it here too.
    torch.manual_seed(config["seed"])
    trainer = Trainer(
        TrackNet(), WBCEFocalLoss(), datasets["train"], datasets["val"], config
    )
    if resume is not None:
        trainer.load_checkpoint(str(resume))
        logger.info("Resumed from %s at step %d", resume, trainer.global_step)
    trainer.train()


def _source_res_distance(dataset: ConcatDataset, width: int, height: int) -> float:
    """``DISTANCE_PX`` in source-video pixels, expressed in heatmap pixels.

    Raises:
        ValueError: If the clips differ in source size, or resizing to
            ``width`` x ``height`` changes the aspect ratio.
    """
    sizes = {cast(TrackNetDataset, d).source_size for d in dataset.datasets}
    if len(sizes) != 1:
        raise ValueError(f"clips in the split differ in source size: {sorted(sizes)}")
    ((source_width, source_height),) = sizes
    if width * source_height != height * source_width:
        raise ValueError(
            f"resizing {source_width}x{source_height} frames to {width}x{height} "
            "changes the aspect ratio, so no single source-pixel distance applies"
        )
    return DISTANCE_PX * width / source_width


def run_evaluation(config: dict, weights: str | Path, split: str) -> dict:
    """Evaluate ``weights`` on the "val" or "test" split under two protocols.

    Both count a detection within 4 px of the label as a true positive:
    ``model_res_4px`` measures in heatmap pixels at model resolution,
    ``source_res_4px`` in pixels of the source video.

    Returns:
        {"split", "frames": frames scored, "model_res_4px": metrics,
        "source_res_4px": metrics}, each metrics dict as ``evaluate_epoch``
        returns it.

    Raises:
        ValueError: If the split's clips differ in source size or the resize
            to ``input_size`` changes their aspect ratio.
    """
    dataset = _build_datasets(config, (split,))[split]
    width, height = config["input_size"]
    distances = {
        "model_res_4px": DISTANCE_PX,
        "source_res_4px": _source_res_distance(dataset, width, height),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrackNet()
    load_model_weights(model, weights, device)
    model.to(device)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    metrics = {
        protocol: evaluate_epoch(
            model,
            loader,
            device,
            detection_threshold=config["detection_threshold"],
            distance_threshold=distance,
        )
        for protocol, distance in distances.items()
    }
    frames = sum(metrics["model_res_4px"][outcome] for outcome in OUTCOMES)
    return {"split": split, "frames": frames, **metrics}
