"""Error analysis of a checkpoint on the validation split.

Scores every validation frame against its label coordinates (not the centroid
of the target disc, which is what ``evaluate`` uses), writes one row per frame
to a CSV and prints summary tables. Positions are in model-resolution pixels.

Usage, from the repo root:
    PYTHONPATH=. scripts/capped.sh uv run python scripts/analysis/val_errors.py \
        configs/default.yaml checkpoints/<run>/best.pt <out.csv>
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from data import TrackNetDataset, source_to_model
from models import load_model
from training.evaluate import aggregate_metrics, classify_frame
from training.run import _build_datasets, checkpoint_background, load_config

THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
NEAR_PX = 10.0  # a detection further than this from the label is on something else
FIELDS = (
    "clip frame slot raw_vis visible gx gy detected max "
    "bx by wx wy ax ay size peak blobs"
).split()


def raw_visibility(label_path) -> dict[int, int]:
    """The dataset's own visibility value (0-3) per frame id."""
    with open(label_path, newline="") as f:
        return {
            int(row["file name"].split(".")[0]): int(row["visibility"])
            for row in csv.DictReader(f)
        }


def decode(heatmap: np.ndarray, threshold: float) -> dict | None:
    """The largest component above ``threshold``: binary centroid (bx, by; the
    shipped decoder), value-weighted centroid (wx, wy), argmax (ax, ay), size,
    peak and the number of components. None when nothing is above it."""
    binary = heatmap > threshold
    labeled, count = ndimage.label(binary)
    if count == 0:
        return None
    sizes = ndimage.sum(binary, labeled, range(1, count + 1))
    largest = int(np.argmax(sizes)) + 1
    mask = labeled == largest
    by, bx = ndimage.center_of_mass(mask)
    wy, wx = ndimage.center_of_mass(np.where(mask, heatmap, 0.0))
    ay, ax = np.unravel_index(np.argmax(np.where(mask, heatmap, -1.0)), heatmap.shape)
    return {
        "bx": bx,
        "by": by,
        "wx": wx,
        "wy": wy,
        "ax": float(ax),
        "ay": float(ay),
        "size": int(sizes[largest - 1]),
        "peak": float(heatmap[mask].max()),
        "blobs": count,
    }


def outcome(
    position: dict | None, row: dict, limit: float, shift_x: float = 0.0
) -> str:
    """``classify_frame`` on the binary centroid of ``position`` (None when
    nothing was detected), moved ``shift_x`` px in x."""
    if position is None:
        prediction = (0.0, 0.0, False)
    else:
        prediction = (position["bx"] + shift_x, position["by"], True)
    label = (row["gx"], row["gy"], bool(row["visible"]))
    return classify_frame(prediction, label, limit)


def count_outcomes(rows: list[dict], limit: float, shift_x: float = 0.0) -> Counter:
    """Outcomes of scored rows, each judged on its own detection at 0.5."""
    return Counter(
        outcome(row if row["detected"] else None, row, limit, shift_x) for row in rows
    )


def format_counts(counts: Counter) -> str:
    metrics = aggregate_metrics(counts)
    return (
        f"tp {counts['tp']:4d} fp1 {counts['fp1']:3d} fp2 {counts['fp2']:3d} "
        f"fn {counts['fn']:3d} tn {counts['tn']:3d}  "
        f"P {metrics['precision']:.4f} R {metrics['recall']:.4f} "
        f"F1 {metrics['f1']:.4f}"
    )


def label_offset(row: dict, centroid: str = "b") -> tuple[float, float]:
    """Detected minus labelled position for one of the centroids in ``decode``
    (b, w or a)."""
    return row[centroid + "x"] - row["gx"], row[centroid + "y"] - row["gy"]


def is_near(row: dict) -> bool:
    """Whether the binary centroid is within ``NEAR_PX`` of the label."""
    return math.hypot(*label_offset(row)) <= NEAR_PX


def score_frames(config: dict, weights: str) -> tuple[list[dict], dict[float, Counter]]:
    """One row per validation frame, plus 4 px outcome counts per threshold."""
    width, height = config["input_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stored = load_model(weights, device)
    model.eval()
    background = checkpoint_background(config, stored, weights)
    val = _build_datasets(config | {"background_frame": background}, ("val",))["val"]

    rows: list[dict] = []
    sweep = {threshold: Counter() for threshold in THRESHOLDS}
    for clip in val.datasets:
        assert isinstance(clip, TrackNetDataset)
        name = f"{clip.frames_dir.parent.name}/{clip.frames_dir.name}"
        raw = raw_visibility(clip.label_path)
        source_width, source_height = clip.source_size
        starts = iter(clip.window_starts)
        with torch.no_grad():
            for frames, _targets in DataLoader(clip, batch_size=2, num_workers=0):
                heatmaps = model(frames.to(device)).float().cpu().numpy()
                for window in heatmaps:
                    start = next(starts)
                    for slot, heatmap in enumerate(window):
                        label = clip.labels[start + slot]
                        # A label that is not visible has no position to report.
                        gx = gy = math.nan
                        if label.visible:
                            gx = source_to_model(label.x, source_width, width)
                            gy = source_to_model(label.y, source_height, height)
                        row = {
                            "clip": name,
                            "frame": label.frame,
                            "slot": slot,
                            "raw_vis": raw[label.frame],
                            "visible": int(label.visible),
                            "gx": gx,
                            "gy": gy,
                            "max": float(heatmap.max()),
                        }
                        for threshold in THRESHOLDS:
                            position = decode(heatmap, threshold)
                            sweep[threshold][outcome(position, row, 4.0)] += 1
                        position = decode(heatmap, 0.5)
                        row["detected"] = int(position is not None)
                        rows.append(row | (position or {}))
    return rows, sweep


def report(
    rows: list[dict], sweep: dict[float, Counter], width: int, height: int
) -> None:
    print(f"frames scored: {len(rows)}")
    for limit in (4.0, 1.6):
        print(f"within {limit} px: {format_counts(count_outcomes(rows, limit))}")

    hits = [r for r in rows if r["detected"] and r["visible"]]
    print(f"\ndetections on a visible ball: {len(hits)}")
    for centroid, title in (
        ("b", "binary centroid"),
        ("w", "weighted centroid"),
        ("a", "argmax"),
    ):
        dx, dy = np.array([label_offset(r, centroid) for r in hits]).T
        distance = np.hypot(dx, dy)
        near = distance <= NEAR_PX
        within = "  ".join(
            f"<={t}: {np.mean(distance <= t):.3f}" for t in (1.0, 1.6, 2.0, 3.0, 4.0)
        )
        print(
            f"{title:18s} median {np.median(distance):.2f}  "
            f"p75 {np.percentile(distance, 75):.2f}  p90 {np.percentile(distance, 90):.2f}  "
            f"p95 {np.percentile(distance, 95):.2f}   {within}"
        )
        print(
            f"{'':18s} within {NEAR_PX:.0f} px (n {near.sum()}): "
            f"dx mean {dx[near].mean():+.3f} sd {dx[near].std():.3f}  "
            f"dy mean {dy[near].mean():+.3f} sd {dy[near].std():.3f}"
        )

    near_hits = [r for r in hits if is_near(r)]
    shift = -float(np.mean([label_offset(r)[0] for r in near_hits]))
    print(
        f"\nshifting predictions {shift:+.2f} px in x (fitted on these frames, so "
        f"optimistic):\n  within 1.6 px: {format_counts(count_outcomes(rows, 1.6, shift))}"
    )

    print("\nby raw visibility (0 absent, 1 visible, 2 hard to see, 3 occluded):")
    for visibility in sorted({r["raw_vis"] for r in rows}):
        selected = [r for r in rows if r["raw_vis"] == visibility]
        detected = sum(r["detected"] for r in selected)
        print(
            f"  {visibility}: {len(selected):4d} frames, detected {detected:4d} "
            f"({detected / len(selected):.3f})"
        )

    print("\nby slot in the 3-frame window (4 px):")
    for slot in range(3):
        selected = [r for r in rows if r["slot"] == slot]
        print(f"  slot {slot}: {format_counts(count_outcomes(selected, 4.0))}")

    print("\nper clip (4 px), with the mean horizontal error of near detections:")
    per_clip = defaultdict(list)
    for r in rows:
        per_clip[r["clip"]].append(r)
    for name, selected in per_clip.items():
        dx = [label_offset(r)[0] for r in near_hits if r["clip"] == name]
        print(
            f"  {name:13s} {format_counts(count_outcomes(selected, 4.0))}  "
            f"dx {np.mean(dx):+.2f}"
        )

    print("\ndetection threshold sweep (4 px):")
    for threshold, counts in sweep.items():
        print(f"  {threshold:.1f}: {format_counts(counts)}")

    misses = [r for r in rows if r["visible"] and not r["detected"]]
    peaks = np.array([r["max"] for r in misses])
    edge = sum(
        min(r["gx"], width - 1 - r["gx"], r["gy"], height - 1 - r["gy"]) < 8
        for r in misses
    )
    print(
        f"\nmisses: {len(misses)}; heatmap peak under 0.1: {int((peaks < 0.1).sum())}, "
        f"0.1 to 0.5: {int((peaks >= 0.1).sum())}; within 8 px of the frame edge: {edge}"
    )
    false_alarms = [r for r in rows if r["detected"] and not r["visible"]]
    print(
        f"false alarms without a labelled ball: {len(false_alarms)}, by raw visibility "
        f"{dict(Counter(r['raw_vis'] for r in false_alarms))}; median blob size "
        f"{np.median([r['size'] for r in false_alarms]):.0f} px against "
        f"{np.median([r['size'] for r in hits]):.0f} px on visible balls"
    )


def main() -> None:
    config_path, weights, out_csv = sys.argv[1:4]
    config = load_config(config_path)
    if config["data"]["format"] != "tennis":
        raise ValueError(
            f"{config_path} has data.format {config['data']['format']!r}: this "
            "script reads the tennis Label.csv columns and supports only 'tennis'"
        )
    rows, sweep = score_frames(config, weights)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report(rows, sweep, *config["input_size"])


if __name__ == "__main__":
    main()
