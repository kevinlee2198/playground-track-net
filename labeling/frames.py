"""The frames a labelling session shows: a directory of images, or a video's frames
decoded once to JPEG files."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from data.extract import TEMP_SUFFIX, frame_file_name
from inference.video_preprocess import _decode, _image_files, _read_image

JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]
# Written last into an extraction: the source record of the video it came from
MANIFEST = "source.json"
NUMBER = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class FrameSet:
    paths: list[Path]  # frame i is paths[i]
    width: int  # of every frame, in pixels
    height: int
    # What the frames are, for telling whether a label CSV was made on them: a
    # video's name, size, modification time and frame count, or a directory's
    # resolved path and its frame files' names and sizes, in order
    source: dict


def prepare_frames(source: str | Path, frames_dir: str | Path | None) -> FrameSet:
    """The frames of ``source``: a directory's images in name order, or a video's
    frames, decoded in order into ``frames_dir`` (frame i is the i-th frame that
    decodes, never found by seeking).

    A directory whose image names are all numbers must have them in numeric
    order when sorted by name (``0001.png``, not ``1.png ... 10.png``).

    A video is decoded into ``<frames_dir>.partial`` and renamed into place when
    complete, with a manifest holding the video's source record. A finished
    ``frames_dir`` is reused; a leftover ``.partial`` is removed and redone. A
    ``frames_dir`` that holds anything else, or the frames of another video (or
    of this one before it changed), is an error and left as it is.
    """
    source = Path(source)
    if source.is_dir():
        if frames_dir is not None:
            raise ValueError("--frames-dir applies only to a video, not a directory")
        paths = _image_files(source)
        _require_numeric_order(source, paths)
        record = {
            "kind": "directory",
            "path": str(source.resolve()),
            "count": len(paths),
            "files": [[p.name, p.stat().st_size] for p in paths],
        }
    else:
        if frames_dir is None:
            raise ValueError("a video needs a frames directory")
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            _extract(source, frames_dir)
        paths, record = _finished_extraction(source, frames_dir)
    height, width = _read_image(paths[0]).shape[:2]
    return FrameSet(paths, width, height, record)


def record_differences(then: dict, now: dict) -> list[str]:
    """What differs between two source records, one phrase per key."""
    keys = ["kind"] if then.get("kind") != now.get("kind") else [*then, *now]
    differences = []
    for key in dict.fromkeys(keys):
        a, b = then.get(key), now.get(key)
        if a == b:
            continue
        if isinstance(a, list) and isinstance(b, list):  # name the first difference
            i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            i = min(len(a), len(b)) if i is None else i
            a = a[i] if i < len(a) else "nothing"
            b = b[i] if i < len(b) else "nothing"
            key = f"{key}[{i}]"
        differences.append(f"{key} was {a!r}, is {b!r}")
    return differences


def _require_numeric_order(directory: Path, paths: list[Path]) -> None:
    """Refuse numbered frame names whose name order is not their numeric order."""
    if not all(NUMBER.fullmatch(p.stem) for p in paths):
        return
    numbers = [int(p.stem) for p in paths]
    for i in range(1, len(paths)):
        if numbers[i] <= numbers[i - 1]:
            raise ValueError(
                f"the frames in {directory} are taken in name order, which puts "
                f"{paths[i - 1].name} before {paths[i].name}: zero-pad the numbers "
                "(0001.png, 0002.png, ...) so that name order is frame order"
            )


def _video_record(video: Path) -> dict:
    stat = video.stat()
    return {
        "kind": "video",
        "name": video.name,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _extract(video: Path, frames_dir: Path) -> None:
    record = _video_record(video)  # also fails early on a missing file
    temp = frames_dir.with_name(frames_dir.name + TEMP_SUFFIX)
    if temp.exists():
        shutil.rmtree(temp)  # the leftover of an interrupted extraction
    temp.mkdir(parents=True)
    try:
        count = 0
        for image in _decode(video):
            path = temp / frame_file_name(count)
            if not cv2.imwrite(str(path), image, JPEG_PARAMS):
                raise OSError(f"failed to write {path}")
            count += 1
        if count == 0:
            raise ValueError(f"{video} decodes no frames")
        (temp / MANIFEST).write_text(json.dumps({**record, "frames": count}) + "\n")
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    temp.rename(frames_dir)
    print(f"Decoded {count} frames of {video} into {frames_dir}")


def _finished_extraction(video: Path, frames_dir: Path) -> tuple[list[Path], dict]:
    """The frame files of a finished extraction of ``video`` in ``frames_dir``,
    and its manifest: the video's source record."""
    fix = "pass another --frames-dir or remove it"
    try:
        manifest = json.loads((frames_dir / MANIFEST).read_text())
        count = int(manifest["frames"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise FileExistsError(
            f"{frames_dir} exists but is not a finished extraction ({e}): {fix}"
        ) from e
    if differences := record_differences(
        {k: v for k, v in manifest.items() if k != "frames"}, _video_record(video)
    ):
        raise FileExistsError(
            f"{frames_dir} holds the frames of another video, or of {video.name} "
            f"before it changed ({'; '.join(differences)}): {fix}"
        )
    paths = [frames_dir / frame_file_name(i) for i in range(count)]
    expected = {p.name for p in paths} | {MANIFEST}
    present = {p.name for p in frames_dir.iterdir()}
    if present != expected or any(p.stat().st_size == 0 for p in paths):
        raise FileExistsError(
            f"{frames_dir} does not hold exactly the {count} frames its "
            f"{MANIFEST} lists, none empty: {fix}"
        )
    return paths, manifest
