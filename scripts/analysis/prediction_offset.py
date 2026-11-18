"""Mean offset of predictions from labels on un-augmented clips of one split,
with the input as it is and mirrored left to right.

A constant offset that changes when the input is mirrored points at something
the model ties to the picture's orientation. Offsets are prediction minus
label in model-resolution pixels; detections more than 10 px from the label
are left out.

Usage, from the repo root:
    PYTHONPATH=. uv run python scripts/analysis/prediction_offset.py \
        configs/default.yaml checkpoints/<run>/best.pt <train|val> <every_nth_window>
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import torch

from data import (
    TrackNetDataset,
    discover_clips,
    require_frames,
    source_to_model,
    split_clips,
)
from models import load_model
from training.evaluate import heatmap_to_position
from training.run import checkpoint_background, load_config

NEAR_PX = 10.0


def format_offsets(offsets: list[tuple[float, float]], mirrored: bool) -> str:
    values = np.array(offsets)
    return (
        f"mirrored={mirrored!s:5s} n {len(values):5d}  "
        f"dx {values[:, 0].mean():+.3f}  dy {values[:, 1].mean():+.3f}"
    )


def main() -> None:
    config_path, weights, split, nth = sys.argv[1:5]
    config = load_config(config_path)
    width, height = config["input_size"]
    data = config["data"]
    splits = split_clips(discover_clips(data), data["val_groups"], data["test_groups"])
    clips = splits[split]
    require_frames(clips)  # as build_dataset would: the clips are built directly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stored = load_model(weights, device)
    model.eval()
    background = checkpoint_background(config, stored, weights)

    # (group, mirrored) -> [(dx, dy), ...]
    offsets: dict[tuple[str, bool], list[tuple[float, float]]] = defaultdict(list)
    for clip_info in clips:
        clip = TrackNetDataset(
            clip_info.frames_dir,
            clip_info.label_path,
            height=height,
            width=width,
            radius=config["heatmap_radius"],
            stride=3,
            background=background,
        )
        source_width, source_height = clip.source_size
        for window in range(0, len(clip), int(nth)):
            frames, _targets = clip[window]
            start = clip.window_starts[window]
            for mirrored in (False, True):
                # Mirrors every image of the window, a background too.
                inputs = frames.flip(-1) if mirrored else frames
                with torch.no_grad():
                    heatmaps = model(inputs.unsqueeze(0).to(device))[0].float().cpu()
                for slot, heatmap in enumerate(heatmaps):
                    label = clip.labels[start + slot]
                    x, y, detected = heatmap_to_position(heatmap)
                    if not (label.visible and detected):
                        continue
                    label_x = source_to_model(label.x, source_width, width)
                    label_y = source_to_model(label.y, source_height, height)
                    if mirrored:
                        label_x = (width - 1) - label_x
                    dx, dy = x - label_x, y - label_y
                    if np.hypot(dx, dy) <= NEAR_PX:
                        offsets[(clip_info.group, mirrored)].append((dx, dy))

    for mirrored in (False, True):
        pooled = [
            v for (_, m), values in offsets.items() if m == mirrored for v in values
        ]
        print(f"{split} {format_offsets(pooled, mirrored)}")
    # At least 7 wide, the width tennis's game names always had.
    pad = max([7, *(len(group) for group, _ in offsets)])
    for (group, mirrored), values in sorted(offsets.items()):
        print(f"  {group:{pad}s} {format_offsets(values, mirrored)}")


if __name__ == "__main__":
    main()
