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


TENNIS_LABEL_ROWS = [  # file name, visibility, x, y, status — 1280x720 coordinates
    ("0000.jpg", 1, 640, 360, 0),
    ("0001.jpg", 1, 650, 355, 0),
    ("0002.jpg", 2, 660, 350, 0),
    ("0003.jpg", 0, "", "", ""),
    ("0004.jpg", 3, 680, 340, 0),
    ("0005.jpg", 1, 690, 335, 0),
]
TENNIS_CLIPS = ("game1/Clip1", "game1/Clip2", "game2/Clip1", "game3/Clip1")


@pytest.fixture
def tennis_root(tmp_path: Path) -> Path:
    """Create a miniature TrackNet tennis dataset: game*/Clip*/ with JPEGs + Label.csv.

    Frames are 1280x720; frame ``i`` is a uniform grey of level ``20 * i``.
    """
    root = tmp_path / "tennis"
    for clip in TENNIS_CLIPS:
        clip_dir = root / clip
        clip_dir.mkdir(parents=True)
        for i, (file_name, *_) in enumerate(TENNIS_LABEL_ROWS):
            img = np.full((720, 1280, 3), 20 * i, dtype=np.uint8)
            cv2.imwrite(str(clip_dir / file_name), img)
        with open(clip_dir / "Label.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["file name", "visibility", "x-coordinate", "y-coordinate", "status"]
            )
            writer.writerows(TENNIS_LABEL_ROWS)
    return root
