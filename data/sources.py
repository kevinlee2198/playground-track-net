"""Find labelled clips on disk, split them by group, and build one dataset over many."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import ConcatDataset

from data.dataset import TrackNetDataset

LABEL_FILE = "Label.csv"


@dataclass(frozen=True)
class Clip:
    frames_dir: Path
    label_path: Path
    group: str  # the unit splits are made on; for tennis, the game directory name


def _natural_key(path: Path) -> list[str | int]:
    """Sort key that orders digit runs numerically: game2 before game10."""
    return [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)
    ]


def _sorted_dirs(parent: Path) -> list[Path]:
    return sorted((p for p in parent.iterdir() if p.is_dir()), key=_natural_key)


def discover_tennis_clips(root: str | Path) -> list[Clip]:
    """root/<game>/<clip>/Label.csv, in natural order (game2 before game10, Clip2 before Clip10).

    A clip is a directory directly under a game directory that holds a ``Label.csv``;
    anything else at either level is ignored.

    Raises:
        FileNotFoundError: If ``root`` is not a directory.
        ValueError: If ``root`` holds no clips, for example because it points one
            level too deep.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")
    clips = [
        Clip(frames_dir=clip_dir, label_path=clip_dir / LABEL_FILE, group=game_dir.name)
        for game_dir in _sorted_dirs(root)
        for clip_dir in _sorted_dirs(game_dir)
        if (clip_dir / LABEL_FILE).is_file()
    ]
    if not clips:
        raise ValueError(
            f"no clips found under {root}: expected root/<game>/<clip>/{LABEL_FILE}"
        )
    return clips


def split_clips(
    clips: Sequence[Clip], val_groups: Sequence[str], test_groups: Sequence[str]
) -> dict[str, list[Clip]]:
    """Keys "train", "val", "test". Train is every group not named."""
    val, test = set(val_groups), set(test_groups)
    named = val | test
    unknown = named - {c.group for c in clips}
    if unknown:
        raise ValueError(f"unknown groups: {sorted(unknown)}")
    overlap = val & test
    if overlap:
        raise ValueError(f"groups in both val and test: {sorted(overlap)}")
    splits = {
        "train": [c for c in clips if c.group not in named],
        "val": [c for c in clips if c.group in val],
        "test": [c for c in clips if c.group in test],
    }
    if not splits["train"]:
        raise ValueError("train split is empty: every group is named in val or test")
    return splits


def build_dataset(
    clips: Sequence[Clip],
    *,
    stride: int,
    radius: float,
    height: int = 288,
    width: int = 512,
    transform: Callable | None = None,
) -> ConcatDataset:
    """One ``TrackNetDataset`` per clip, concatenated in the order given."""
    return ConcatDataset(
        [
            TrackNetDataset(
                c.frames_dir,
                c.label_path,
                height=height,
                width=width,
                radius=radius,
                stride=stride,
                transform=transform,
            )
            for c in clips
        ]
    )
