"""Find labelled clips on disk, split them by group, and build one dataset over many."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import ConcatDataset

from data.dataset import TrackNetDataset

LABEL_FILE = "Label.csv"
BADMINTON_LABEL_SUFFIX = "_ball.csv"


@dataclass(frozen=True)
class Clip:
    frames_dir: Path
    label_path: Path
    # The unit splits are made on: for tennis the game directory name, for
    # badminton "<part>/<match>".
    group: str
    # The --root to give `main.py extract-frames` when the frames are decoded
    # from video (badminton); None when the dataset ships them (tennis).
    extract_root: Path | None = None


@dataclass(frozen=True)
class BadmintonRally:
    """One rally of the CoachAI badminton dataset: its video, labels and frames.

    ``match_dir`` is ``root/<part>/<match>``; the rally's files are
    ``video/<name>.mp4``, ``csv/<name>_ball.csv`` and, once ``extract-frames``
    has run, ``frames/<name>/``.
    """

    match_dir: Path
    name: str

    @property
    def group(self) -> str:
        return f"{self.match_dir.parent.name}/{self.match_dir.name}"

    @property
    def video_path(self) -> Path:
        return self.match_dir / "video" / f"{self.name}.mp4"

    @property
    def label_path(self) -> Path:
        return self.match_dir / "csv" / f"{self.name}{BADMINTON_LABEL_SUFFIX}"

    @property
    def frames_dir(self) -> Path:
        return self.match_dir / "frames" / self.name


def _natural_key(path: Path) -> list[str | int]:
    """Sort key that orders digit runs numerically: game2 before game10."""
    return [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)
    ]


def _sorted_dirs(parent: Path) -> list[Path]:
    return sorted((p for p in parent.iterdir() if p.is_dir()), key=_natural_key)


def _check_root(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")


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
    _check_root(root)
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


def find_badminton_rallies(root: str | Path) -> list[BadmintonRally]:
    """Every ``root/<part>/<match>/csv/<rally>_ball.csv``, as a rally, in natural
    order (match2 before match10). Neither the video nor the frames need exist.

    Raises:
        FileNotFoundError: If ``root`` is not a directory.
        ValueError: If ``root`` holds no label files.
    """
    root = Path(root)
    _check_root(root)
    rallies = [
        BadmintonRally(match_dir, label.name.removesuffix(BADMINTON_LABEL_SUFFIX))
        for part_dir in _sorted_dirs(root)
        for match_dir in _sorted_dirs(part_dir)
        for label in sorted(
            (match_dir / "csv").glob(f"*{BADMINTON_LABEL_SUFFIX}"), key=_natural_key
        )
        if label.is_file()
    ]
    if not rallies:
        raise ValueError(
            f"no rallies found under {root}: expected "
            f"root/<part>/<match>/csv/<rally>{BADMINTON_LABEL_SUFFIX} with the "
            "video at root/<part>/<match>/video/<rally>.mp4"
        )
    return rallies


def discover_badminton_clips(root: str | Path) -> list[Clip]:
    """One clip per ``root/<part>/<match>/csv/<rally>_ball.csv``, in natural order,
    reading frames from ``root/<part>/<match>/frames/<rally>/``; its group is
    ``"<part>/<match>"``, since match names repeat across parts.

    The frames need not exist yet (``main.py extract-frames`` writes them):
    ``build_dataset`` checks them for the clips it builds, so a split that is
    not built need not be extracted.

    Raises:
        FileNotFoundError: If ``root`` is not a directory.
        ValueError: If ``root`` holds no label files.
    """
    root = Path(root)
    return [
        Clip(rally.frames_dir, rally.label_path, rally.group, extract_root=root)
        for rally in find_badminton_rallies(root)
    ]


def require_frames(clips: Sequence[Clip]) -> None:
    """Check that every clip's frames directory exists.

    Raises:
        FileNotFoundError: For the first clip without one, naming it and, for
            frames decoded from video, the ``extract-frames`` command to run.
    """
    for clip in clips:
        if not clip.frames_dir.is_dir():
            message = (
                f"{clip.group}/{clip.frames_dir.name}: no frames directory "
                f"{clip.frames_dir} for {clip.label_path}"
            )
            if clip.extract_root is not None:
                message += (
                    f": run `main.py extract-frames --root {clip.extract_root}` first"
                )
            raise FileNotFoundError(message)


# data.format -> the function that discovers that dataset's clips under data.root
_DISCOVER_BY_FORMAT: dict[str, Callable[[str | Path], list[Clip]]] = {
    "tennis": discover_tennis_clips,
    "badminton": discover_badminton_clips,
}


def discover_clips(data: dict) -> list[Clip]:
    """The clips under ``data["root"]``, found as ``data["format"]`` lays them out.

    ``data`` is a config's ``data`` section.

    Raises:
        ValueError: If the format is not one of the supported ones, or (from
            the format's discovery function) the root holds no clips.
        FileNotFoundError: As the format's discovery function raises it.
    """
    discover = _DISCOVER_BY_FORMAT.get(data["format"])
    if discover is None:
        supported = ", ".join(repr(f) for f in _DISCOVER_BY_FORMAT)
        raise ValueError(
            f"unknown data.format {data['format']!r}: supported formats are {supported}"
        )
    return discover(data["root"])


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
    background: bool = False,
) -> ConcatDataset:
    """One ``TrackNetDataset`` per clip, concatenated in the order given.

    Raises:
        FileNotFoundError: As ``require_frames`` raises it, before any clip
            is read.
    """
    require_frames(clips)
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
                background=background,
            )
            for c in clips
        ]
    )
