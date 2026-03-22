import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

FIXTURE_LABELS: list[tuple[int, int, int, int]] = [
    (0, 1, 100, 80),
    (1, 1, 110, 85),
    (2, 1, 120, 90),
    (3, 0, 0, 0),
    (4, 1, 140, 100),
    (5, 1, 150, 105),
    (6, 1, 160, 110),
    (7, 2, 170, 115),
    (8, 1, 180, 120),
]


@pytest.fixture
def sample_frames_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Create a 9-frame dataset with synthetic images and a CSV label file."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    rng = np.random.RandomState(42)

    for frame_idx, vis, bx, by in FIXTURE_LABELS:
        img = rng.randint(0, 256, (288, 512, 3), dtype=np.uint8)
        if vis > 0:
            cv2.circle(img, (bx, by), 5, (255, 255, 255), -1)
        cv2.imwrite(str(frames_dir / f"{frame_idx:05d}.jpg"), img)

    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Visibility", "X", "Y"])
        for row in FIXTURE_LABELS:
            writer.writerow(row)

    return frames_dir, csv_path
