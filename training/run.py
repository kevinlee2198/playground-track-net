"""Train or evaluate a TrackNet model on the dataset a config names, or check its labels."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from data import (
    Compose,
    FrameColorJitter,
    HorizontalFlip,
    Label,
    TrackNetDataset,
    build_dataset,
    discover_tennis_clips,
    source_to_model,
    split_clips,
    visibility_counts,
)
from models import TrackNet, WBCEFocalLoss, load_model_weights
from training.evaluate import OUTCOMES, evaluate_epoch
from training.trainer import Trainer, experiment_dir

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

    Raises:
        FileExistsError: If ``resume`` is not given but this experiment already
            has a ``latest.pt``, which a fresh run would overwrite. Checked
            before anything is built or written.
    """
    if resume is None:
        latest = Path(experiment_dir(config, "checkpoint_dir")) / "latest.pt"
        if latest.exists():
            raise FileExistsError(
                f"{latest} already exists and a fresh run would overwrite it: "
                f"pass --resume {latest} to continue that run, or set a different "
                "experiment_name to start a new one"
            )
    datasets = _build_datasets(config, ("train", "val"))
    # The model is initialised before the Trainer seeds, so seed it here too.
    torch.manual_seed(config["seed"])
    trainer = Trainer(
        TrackNet(), WBCEFocalLoss(), datasets["train"], datasets["val"], config
    )
    if resume is not None:
        _warn_config_changes(trainer.load_checkpoint(str(resume)), config)
        logger.info("Resumed from %s at step %d", resume, trainer.global_step)
    trainer.train()


# Keys a resumed run is expected to change: they only set where it stops.
_RUN_LENGTH_KEYS = ("epochs", "max_steps")


def _warn_config_changes(stored: dict, current: dict) -> None:
    """Log one warning listing each top-level key whose value differs between
    the checkpoint's config and the current one, except the run-length keys."""
    changes = [
        f"{key}: {stored.get(key)!r} -> {current.get(key)!r}"
        for key in sorted(stored.keys() | current.keys())
        if key not in _RUN_LENGTH_KEYS and stored.get(key) != current.get(key)
    ]
    if changes:
        logger.warning(
            "Resuming with a config that differs from the checkpoint's: %s",
            "; ".join(changes),
        )


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


SPLITS = ("train", "val", "test")
MARKER_BGR = (255, 0, 255)  # magenta: stands out on court colours and a yellow ball
MARKER_RADIUS = 6  # px at model resolution
CROP_SIZE = 64  # px at model resolution, centred on the label
CROP_ZOOM = 4


def write_label_overlays(
    config: dict, out_dir: str | Path, count: int = 12
) -> list[Path]:
    """Write PNGs of visible-ball labels drawn on their frames, for a human to check.

    The datasets are built exactly as for training, so a clip that would break
    training raises here with the same error. ``count`` visible-ball frames are
    shared out across the train, val and test splits in turn (each split gets one
    before any gets two) and evenly spaced within each split; if the splits hold
    fewer, all of them are written. Each PNG shows the frame at model resolution
    with a hollow circle at the rescaled label, beside a 4x enlargement of the
    64x64 region centred on the label. Files are named
    ``<split>_<game>_<clip>_<frame>.png``.

    Logs, per split, the clip count, the frame count, how many label rows
    hold each raw visibility value and how many visible labels lie outside
    the source frame.

    Returns:
        The written paths: train first, then val, then test, each in clip and
        frame order.
    """
    candidates: dict[str, list[tuple[TrackNetDataset, int]]] = {}
    for split, dataset in _build_datasets(config, SPLITS).items():
        clips = [cast(TrackNetDataset, d) for d in dataset.datasets]
        visibility: Counter[int] = Counter()
        for clip in clips:
            visibility.update(visibility_counts(clip.label_path))
        logger.info(
            "%s: %d clips, %d frames, visibility %s, "
            "%d visible labels outside the frame",
            split,
            len(clips),
            sum(clip.num_frames for clip in clips),
            dict(sorted(visibility.items())),
            sum(_outside_frame(clip, label) for clip in clips for label in clip.labels),
        )
        candidates[split] = [
            (clip, i)
            for clip in clips
            for i, label in enumerate(clip.labels)
            if label.visible
        ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shares = _round_robin_shares(count, {s: len(c) for s, c in candidates.items()})
    paths = []
    for split, n in shares.items():
        frames = candidates[split]
        for k in range(n):
            clip, i = frames[(2 * k + 1) * len(frames) // (2 * n)]  # evenly spaced
            game, clip_name = clip.frames_dir.parent.name, clip.frames_dir.name
            stem = clip.frame_paths[i].stem
            path = out_dir / f"{split}_{game}_{clip_name}_{stem}.png"
            if not cv2.imwrite(str(path), _label_overlay(clip, i)):
                raise OSError(f"failed to write {path}")
            paths.append(path)
    if len(paths) < count:
        logger.warning(
            "only %d visible-ball frames in the dataset, fewer than the %d asked for",
            len(paths),
            count,
        )
    return paths


def _round_robin_shares(total: int, capacities: dict[str, int]) -> dict[str, int]:
    """Deal ``total`` items out one at a time to each key in turn, up to its capacity."""
    shares = dict.fromkeys(capacities, 0)
    while total > 0 and any(shares[k] < capacities[k] for k in shares):
        for key in shares:
            if total > 0 and shares[key] < capacities[key]:
                shares[key] += 1
                total -= 1
    return shares


def _outside_frame(clip: TrackNetDataset, label: Label) -> bool:
    """Whether ``label`` is visible but lies outside ``clip``'s source frame."""
    source_width, source_height = clip.source_size
    return label.visible and not (
        0 <= label.x < source_width and 0 <= label.y < source_height
    )


def _label_overlay(clip: TrackNetDataset, i: int) -> np.ndarray:
    """Frame ``i`` at model resolution with its label circled, beside an enlarged crop.

    A label outside the frame is still drawn, at its true position, which is
    off the frame or in the crop's padding, and a warning is logged; training
    accepts such labels too.
    """
    width, height = clip.width, clip.height
    source_width, source_height = clip.source_size
    label = clip.labels[i]
    if _outside_frame(clip, label):
        logger.warning(
            "%s: label of frame %d at (%s, %s) lies outside the %dx%d frame",
            clip.label_path,
            label.frame,
            label.x,
            label.y,
            source_width,
            source_height,
        )
    # The same mapping TrackNetDataset applies to its heatmap targets.
    x = source_to_model(label.x, source_width, width)
    y = source_to_model(label.y, source_height, height)
    cx, cy = round(x), round(y)
    # Crop origin in padded coordinates; clamped so the crop stays inside the
    # padded frame when the label lies beyond its edge.
    ox, oy = min(max(cx, 0), width), min(max(cy, 0), height)

    image = cv2.imread(str(clip.frame_paths[i]))
    if image is None:
        raise RuntimeError(f"Failed to read image: {clip.frame_paths[i]}")
    frame = cv2.resize(image, (width, height))

    # Pad so a label near the edge still sits at the centre of its crop.
    half = CROP_SIZE // 2
    padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_CONSTANT)
    crop = padded[oy : oy + CROP_SIZE, ox : ox + CROP_SIZE]
    enlarged = cv2.resize(
        crop, None, fx=CROP_ZOOM, fy=CROP_ZOOM, interpolation=cv2.INTER_NEAREST
    )
    # Each model pixel p covers enlarged pixels [4p, 4p + 3]; its centre is 4p + 1.5.
    centre = (
        round((x - ox + half) * CROP_ZOOM + (CROP_ZOOM - 1) / 2),
        round((y - oy + half) * CROP_ZOOM + (CROP_ZOOM - 1) / 2),
    )
    cv2.circle(enlarged, centre, MARKER_RADIUS * CROP_ZOOM, MARKER_BGR, 1)
    cv2.circle(frame, (cx, cy), MARKER_RADIUS, MARKER_BGR, 1)

    side = CROP_SIZE * CROP_ZOOM
    canvas = np.zeros((max(height, side), width + side, 3), dtype=np.uint8)
    canvas[:height, :width] = frame
    canvas[:side, width:] = enlarged
    return canvas
