"""Hand labels of one clip, kept in a CSV of the ``internal`` layout that is
rewritten atomically after every change, beside a record of the frames it
labels."""

from __future__ import annotations

import csv
import errno
import fcntl
import io
import json
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from labeling.frames import record_differences

HEADER = ["Frame", "Visibility", "X", "Y"]
# 1 ball visible at X,Y; 2 ball there but hard to see, at X,Y; 0 no ball in the
# frame; 3 the labeller cannot tell (left out when scoring). 0 and 3 have no X,Y.
WITH_POSITION = (1, 2)
VISIBILITIES = (0, 1, 2, 3)
NEW_FILE_MODE = 0o644


@dataclass(frozen=True)
class HandLabel:
    visibility: int
    x: float | None  # source pixels, pixel centres at integers; None for 0 and 3
    y: float | None


def check_frame(frame: int, num_frames: int) -> None:
    if not 0 <= frame < num_frames:
        raise ValueError(f"frame {frame} is outside 0-{num_frames - 1}")


def check_label(
    frame: int, label: HandLabel, num_frames: int, width: int, height: int
) -> HandLabel:
    """``label`` with its coordinates rounded to two decimals, or ValueError.
    Checks the types too, since a label may come from JSON."""
    check_frame(frame, num_frames)
    vis = label.visibility
    if type(vis) is not int or vis not in VISIBILITIES:
        raise ValueError(f"visibility must be 0, 1, 2 or 3, got {vis!r}")
    if vis not in WITH_POSITION:
        if label.x is not None or label.y is not None:
            raise ValueError(f"visibility {vis} takes no coordinates")
        return label
    coords = []
    for name, value, size in (("x", label.x, width), ("y", label.y, height)):
        # Not bool, which is an int; the range check also refuses NaN and infinity
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= size - 1
        ):
            raise ValueError(
                f"visibility {vis} needs {name} in 0-{size - 1}, got {value!r}"
            )
        coords.append(round(value, 2))
    return HandLabel(vis, *coords)


def sibling(path: Path, suffix: str) -> Path:
    """``path`` with ``suffix`` added to its whole name: labels.csv.lock."""
    return path.with_name(path.name + suffix)


def write_atomically(path: Path, text: str) -> None:
    """Replace ``path`` with ``text``, durably: through a temporary file in the
    same directory, synced, renamed over it, then the directory synced. An
    existing file keeps its mode; a new one gets 0644. On failure ``path`` is
    unchanged or holds ``text`` complete, and no temporary file is left."""
    try:
        mode = path.stat().st_mode & 0o7777
    except FileNotFoundError:
        mode = NEW_FILE_MODE
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            os.fchmod(f.fileno(), mode)
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        _sync_directory(path.parent)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def _sync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    except OSError as e:
        # EINVAL: the file system cannot sync a directory (some network and
        # WSL drives); the rename itself has been made
        if e.errno != errno.EINVAL:
            raise
    finally:
        os.close(fd)


@contextmanager
def exclusive_lock(out: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``<out>.lock`` (created if missing, and left
    in place) while the block runs, so that one run at a time uses ``out``."""
    lock = sibling(out, ".lock")
    with open(lock, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise OSError(
                f"{out} is in use by another label run (it holds {lock}): "
                "stop that run first"
            ) from None
        yield


class LabelStore:
    """The labels of a clip of ``num_frames`` frames of ``width`` x ``height``,
    whose frames are described by the source record ``source``.

    A new CSV is created holding the header only, after ``<path>.source.json``
    holding ``source``. An existing CSV is loaded (any problem in it raised as
    a ValueError naming the line) only if its ``.source.json`` matches
    ``source``. Changes from several threads are made one at a time, and none
    after ``close``.
    """

    labels: dict[int, HandLabel]  # by frame; replaced, never changed in place

    def __init__(
        self, path: str | Path, num_frames: int, width: int, height: int, source: dict
    ):
        self.path = Path(path)
        self.num_frames, self.width, self.height = num_frames, width, height
        self._lock = threading.Lock()
        self._closed = False
        record = sibling(self.path, ".source.json")
        if self.path.exists():
            self._check_record(record, source)
            self.labels = self._load()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_atomically(record, json.dumps(source, indent=1) + "\n")
            self._write({})

    def set(self, frame: int, label: HandLabel) -> HandLabel:
        label = self._check(frame, label)
        with self._lock:
            self._require_open()
            self._write({**self.labels, frame: label})
        return label

    def delete(self, frame: int) -> None:
        check_frame(frame, self.num_frames)
        with self._lock:
            self._require_open()
            if frame in self.labels:
                self._write({k: v for k, v in self.labels.items() if k != frame})

    def close(self) -> None:
        """Wait for a change being written, then refuse any other."""
        with self._lock:
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise OSError("the label tool is stopping")

    def _check(self, frame: int, label: HandLabel) -> HandLabel:
        return check_label(frame, label, self.num_frames, self.width, self.height)

    def _check_record(self, record: Path, source: dict) -> None:
        fix = "pass another --out, or the frames it was made on"
        try:
            then = json.loads(record.read_text())
        except FileNotFoundError:
            raise ValueError(
                f"{self.path} has no {record.name}, so which frames it labels is "
                f"unknown: {fix}"
            ) from None
        except ValueError as e:
            raise ValueError(f"{record} is not JSON ({e}): {fix}") from None
        if differences := record_differences(then, source):
            raise ValueError(
                f"{self.path} labels other frames ({'; '.join(differences)}): {fix}"
            )

    def _write(self, labels: dict[int, HandLabel]) -> None:
        """Replace the file with ``labels``, then keep them; on failure the
        labels kept do not change."""
        text = io.StringIO()
        writer = csv.writer(text, lineterminator="\n")
        writer.writerow(HEADER)
        for frame, label in sorted(labels.items()):
            x, y = label.x, label.y
            xy = ["", ""] if x is None or y is None else [f"{x:.2f}", f"{y:.2f}"]
            writer.writerow([frame, label.visibility, *xy])
        write_atomically(self.path, text.getvalue())
        self.labels = labels

    def _load(self) -> dict[int, HandLabel]:
        labels: dict[int, HandLabel] = {}
        with open(self.path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header != HEADER:
                raise ValueError(
                    f"{self.path}, line 1: header {header!r}, expected {HEADER!r}"
                )
            for row in reader:
                try:
                    if len(row) != 4:
                        raise ValueError(f"{len(row)} fields, expected 4")
                    frame, vis = int(row[0]), int(row[1])
                    x, y = (float(v) if v else None for v in row[2:])
                    if frame in labels:
                        raise ValueError(f"duplicate frame {frame}")
                    labels[frame] = self._check(frame, HandLabel(vis, x, y))
                except ValueError as e:
                    raise ValueError(
                        f"{self.path}, line {reader.line_num}: {e}"
                    ) from None
        return labels
