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


BADMINTON_SIZE = (320, 180)  # width, height: 16:9 like the real 1280x720 videos
BADMINTON_LABEL_ROWS = [  # Frame, Visibility, X, Y in 320x180 coordinates
    (0, 1, 160, 90),
    (1, 1, 165, 85),
    (2, 0, 0, 0),
    (3, 1, 175, 75),
    (4, 1, 180, 70),
    (5, 1, 185, 65),
]
# <part>/<match>/<rally> -> video frames. Match names repeat across parts, and
# match10 sorts after match2 only in natural order.
BADMINTON_RALLIES = {
    "Professional/match10/1_01_01": 9,  # 3 trailing video frames without labels
    "Professional/match2/1_01_10": 6,
    "Professional/match2/1_01_02": 6,
    "Amateur/match2/1_01_01": 6,
    "Test/match1/1_01_01": 6,
}


def badminton_grey(i: int) -> int:
    """The uniform grey level of video frame ``i`` in the badminton fixture."""
    return 25 * i


def write_badminton_video(path: Path, num_frames: int) -> None:
    """An mp4v video of ``num_frames`` uniform grey frames, frame i at
    ``badminton_grey(i)``. mp4v is lossy: decoded levels are a few units off."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = BADMINTON_SIZE
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter.fourcc(*"mp4v"), 25.0, (width, height)
    )
    assert writer.isOpened(), f"cannot write {path}"
    for i in range(num_frames):
        writer.write(np.full((height, width, 3), badminton_grey(i), dtype=np.uint8))
    writer.release()


def write_badminton_labels(
    path: Path, rows: list[tuple[int, int, int, int]] = BADMINTON_LABEL_ROWS
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Visibility", "X", "Y"])
        writer.writerows(rows)


@pytest.fixture
def badminton_root(tmp_path: Path) -> Path:
    """Create a miniature CoachAI badminton dataset, before frame extraction:
    <part>/<match>/video/<rally>.mp4 and <part>/<match>/csv/<rally>_ball.csv.

    Every rally has the six label rows of ``BADMINTON_LABEL_ROWS``; its video
    has the number of frames ``BADMINTON_RALLIES`` gives.
    """
    root = tmp_path / "badminton"
    for rally, num_frames in BADMINTON_RALLIES.items():
        match, name = rally.rsplit("/", 1)
        write_badminton_video(root / match / "video" / f"{name}.mp4", num_frames)
        write_badminton_labels(root / match / "csv" / f"{name}_ball.csv")
    return root
