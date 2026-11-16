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

from data import TrackNetDataset, discover_tennis_clips, source_to_model, split_clips
from models import TrackNet, load_model_weights
from training.evaluate import heatmap_to_position
from training.run import load_config

NEAR_PX = 10.0


def main() -> None:
    config_path, weights, split, nth = sys.argv[1:5]
    config = load_config(config_path)
    width, height = config["input_size"]
    data = config["data"]
    clips = split_clips(
        discover_tennis_clips(data["root"]), data["val_groups"], data["test_groups"]
    )[split]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrackNet()
    load_model_weights(model, weights, device)
    model.to(device).eval()

    # (game, mirrored) -> [(dx, dy), ...]
    offsets: dict[tuple[str, bool], list[tuple[float, float]]] = defaultdict(list)
    for clip_info in clips:
        clip = TrackNetDataset(
            clip_info.frames_dir,
            clip_info.label_path,
            height=height,
            width=width,
            radius=config["heatmap_radius"],
            stride=3,
        )
        source_width, source_height = clip.source_size
        for window in range(0, len(clip), int(nth)):
            frames, _targets = clip[window]
            start = clip.window_starts[window]
            for mirrored in (False, True):
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
        values = np.array(
            [v for (_, m), group in offsets.items() if m == mirrored for v in group]
        )
        print(
            f"{split} mirrored={mirrored!s:5s} n {len(values):5d}  "
            f"dx {values[:, 0].mean():+.3f}  dy {values[:, 1].mean():+.3f}"
        )
    for (game, mirrored), group in sorted(offsets.items()):
        values = np.array(group)
        print(
            f"  {game:7s} mirrored={mirrored!s:5s} n {len(values):5d}  "
            f"dx {values[:, 0].mean():+.3f}  dy {values[:, 1].mean():+.3f}"
        )


if __name__ == "__main__":
    main()
