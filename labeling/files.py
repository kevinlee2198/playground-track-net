"""Durable file writes and the lock that keeps one label run per CSV."""

from __future__ import annotations

import errno
import fcntl
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

NEW_FILE_MODE = 0o644


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
