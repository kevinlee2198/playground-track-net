"""Where is the ball in the picture relative to its label? No model involved.

For every second clearly visible frame, crop the source frame (33 px square)
around the label, remove the background with the median of the six
neighbouring frames' crops, threshold the difference at 30 and take the
difference-weighted centroid of the blob nearest the label. Blobs of fewer
than 6 or more than 200 pixels, or more than 8 px from the label, are left
out. Offsets are blob minus label in source pixels.

The offset is then regressed on the ball's speed (from the neighbouring
labels) and its distance from the frame centre, per axis: the constant is the
offset of a still ball at the centre of the frame.

Usage, from the repo root (reads only the dataset):
    .venv/bin/python -I scripts/analysis/label_vs_ball.py datasets/tennis game1 [game2 ...]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

HALF = 16  # crop half-size
NEIGHBOURS = 3  # frames each side for the background median
STEP = 2
DIFF_THRESHOLD = 30
MIN_BLOB, MAX_BLOB = 6, 200
MAX_OFFSET = 8.0
MOVING = 3.0  # px per frame


def measure_clip(clip: Path) -> list[tuple[float, ...]]:
    """(dx, dy, vx, vy, x, y, width, height) per measured frame; the speed is
    NaN when a neighbouring frame has no clearly visible ball."""
    with open(clip / "Label.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    cache: dict[int, np.ndarray] = {}

    def frame(i: int) -> np.ndarray:
        if i not in cache:
            image = cv2.imread(str(clip / rows[i]["file name"]))
            if image is None:
                raise RuntimeError(f"cannot read {clip / rows[i]['file name']}")
            cache[i] = image.astype(np.int16)
            cache.pop(i - 3 * NEIGHBOURS, None)
        return cache[i]

    def position(i: int) -> tuple[float, float] | None:
        if rows[i]["visibility"] != "1":
            return None
        return float(rows[i]["x-coordinate"]), float(rows[i]["y-coordinate"])

    measured = []
    for i in range(NEIGHBOURS, len(rows) - NEIGHBOURS, STEP):
        here = position(i)
        if here is None:
            continue
        x, y = here
        before, after = position(i - 1), position(i + 1)
        if before and after:
            vx, vy = (after[0] - before[0]) / 2, (after[1] - before[1]) / 2
        else:
            vx = vy = float("nan")
        xi, yi = round(x), round(y)
        height, width = frame(i).shape[:2]
        if not (HALF <= xi < width - HALF and HALF <= yi < height - HALF):
            continue
        crop = (slice(yi - HALF, yi + HALF + 1), slice(xi - HALF, xi + HALF + 1))
        background = np.median(
            [
                frame(j)[crop]
                for j in range(i - NEIGHBOURS, i + NEIGHBOURS + 1)
                if j != i
            ],
            axis=0,
        )
        diff = np.abs(frame(i)[crop] - background).max(axis=2)
        labeled, count = ndimage.label(diff > DIFF_THRESHOLD)
        if count == 0:
            continue
        centres = ndimage.center_of_mass(diff, labeled, range(1, count + 1))
        nearest = int(np.argmin([np.hypot(cy - HALF, cx - HALF) for cy, cx in centres]))
        size = int((labeled == nearest + 1).sum())
        cy, cx = centres[nearest]
        dx, dy = (xi - HALF + cx) - x, (yi - HALF + cy) - y
        if MIN_BLOB <= size <= MAX_BLOB and np.hypot(dx, dy) <= MAX_OFFSET:
            measured.append((dx, dy, vx, vy, x, y, width, height))
    return measured


def report(game: str, values: np.ndarray) -> None:
    dx, dy, vx, vy, x, y, width, height = values.T
    print(
        f"{game}: n {len(values)}  mean offset dx {dx.mean():+.2f} dy {dy.mean():+.2f}"
    )
    known = ~np.isnan(vx)
    for name, selected, axis, offset in (
        ("moving left", known & (vx < -MOVING), "dx", dx),
        ("moving right", known & (vx > MOVING), "dx", dx),
        ("moving up", known & (vy < -MOVING), "dy", dy),
        ("moving down", known & (vy > MOVING), "dy", dy),
    ):
        print(
            f"  {name:12s} n {selected.sum():5d}  {axis} {offset[selected].mean():+.2f}"
        )
    for axis, offset, speed, centred in (
        ("dx", dx, vx, x - (width - 1) / 2),
        ("dy", dy, vy, y - (height - 1) / 2),
    ):
        design = np.c_[np.ones(known.sum()), speed[known], centred[known]]
        coefficients, *_ = np.linalg.lstsq(design, offset[known], rcond=None)
        residual = offset[known] - design @ coefficients
        errors = np.sqrt(np.diag(np.linalg.inv(design.T @ design)) * residual.var())
        print(
            f"  {axis} = {coefficients[0]:+.2f} (±{errors[0]:.2f}) "
            f"{coefficients[1]:+.3f} (±{errors[1]:.3f}) per px/frame of speed "
            f"{coefficients[2] * 1000:+.2f} (±{errors[2] * 1000:.2f}) per 1000 px from the centre"
        )


def main() -> None:
    root = Path(sys.argv[1])
    for game in sys.argv[2:]:
        measured = [
            row
            for clip in sorted((root / game).iterdir())
            if (clip / "Label.csv").exists()
            for row in measure_clip(clip)
        ]
        report(game, np.array(measured))


if __name__ == "__main__":
    main()
