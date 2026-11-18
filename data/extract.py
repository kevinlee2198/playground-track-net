"""Decode the badminton dataset's rally videos into the frame images training reads."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from data.labels import load_labels
from data.sources import BadmintonRally, find_badminton_rallies

logger = logging.getLogger(__name__)

TEMP_SUFFIX = ".partial"  # frames/<rally>.partial: a rally being extracted


@dataclass(frozen=True)
class ExtractionSummary:
    extracted: int  # rallies decoded and written
    skipped: int  # rallies whose frames were already in place
    frames_written: int
    surplus_dropped: int  # video frames past the last label row, not written
    # Extracted rallies whose container frame count differs from the frames decoded
    count_mismatches: int


def frame_file_name(frame: int) -> str:
    """``NNNN.jpg``: the frame id zero-padded to at least four digits."""
    return f"{frame:04d}.jpg"


def extract_badminton_frames(
    root: str | Path, jpeg_quality: int = 95
) -> ExtractionSummary:
    """Write each rally's labelled video frames to ``<part>/<match>/frames/<rally>/``.

    Each video is decoded once, in order; video frame ``i`` is written as the
    JPEG of label row ``Frame == i`` at source resolution. Frames past the
    last label row are dropped and counted. A rally is written into
    ``frames/<rally>.partial`` and renamed into place when complete, so an
    interrupted run can be re-run: a rally whose frames directory already
    holds exactly the expected files, none of them empty, is skipped, and a
    leftover ``.partial`` directory is removed and redone. Two runs must not
    be pointed at the same root at once: the second would remove the first's
    ``.partial`` directory as a leftover.

    Each extracted rally's frame count as the video container reports it is
    compared with the frames decoded (written plus dropped); a difference is
    logged as a warning and counted, and changes nothing that is written.
    Extraction stops at the first rally that raises; rallies finished before
    it stay in place.

    Raises:
        ValueError: If ``jpeg_quality`` is outside 0-100, a label file's frame
            ids are not exactly 0..N-1, or a video decodes fewer frames than
            its label file has rows.
        FileExistsError: If a rally's frames directory exists with other
            contents than the expected files, or one of them is empty. It is
            left as it is.
        OSError: If a video cannot be opened or an image cannot be written.
        And as ``find_badminton_rallies`` raises.
    """
    if not 0 <= jpeg_quality <= 100:
        raise ValueError(f"jpeg_quality must be in 0-100, got {jpeg_quality}")
    rallies = find_badminton_rallies(root)
    extracted = skipped = frames_written = surplus_dropped = count_mismatches = 0
    for k, rally in enumerate(rallies, 1):
        num_frames = _label_row_count(rally)
        if rally.frames_dir.exists():
            problem = _frames_dir_problem(rally, num_frames)
            if problem is not None:
                raise FileExistsError(
                    f"{rally.frames_dir} exists but {problem}: remove it and "
                    "re-run extract-frames"
                )
            skipped += 1
            continue
        surplus, reported = _extract_rally(rally, num_frames, jpeg_quality)
        decoded = num_frames + surplus
        extracted += 1
        frames_written += num_frames
        surplus_dropped += surplus
        if reported != decoded:
            count_mismatches += 1
            logger.warning(
                "%s/%s: the video container reports %d frames, but %d were decoded",
                rally.group,
                rally.name,
                reported,
                decoded,
            )
        logger.info(
            "[%d/%d] %s/%s: %d frames",
            k,
            len(rallies),
            rally.group,
            rally.name,
            num_frames,
        )
        if surplus:
            logger.info(
                "%s/%s: dropped %d video frames past the last label row",
                rally.group,
                rally.name,
                surplus,
            )
    logger.info(
        "extract-frames: %d rallies extracted, %d skipped (already extracted), "
        "%d frames written, %d surplus video frames dropped, "
        "%d rallies whose container frame count differs from the decoded count",
        extracted,
        skipped,
        frames_written,
        surplus_dropped,
        count_mismatches,
    )
    return ExtractionSummary(
        extracted, skipped, frames_written, surplus_dropped, count_mismatches
    )


def _label_row_count(rally: BadmintonRally) -> int:
    """The number of label rows, checked to have frame ids exactly 0..N-1."""
    frames = [label.frame for label in load_labels(rally.label_path)]  # sorted
    if not frames:
        raise ValueError(f"{rally.label_path}: no label rows")
    if frames != list(range(len(frames))):
        i = next(i for i, frame in enumerate(frames) if frame != i)
        raise ValueError(
            f"{rally.label_path}: frame ids are not exactly 0..{len(frames) - 1}, "
            f"one row each: in ascending order, frame {i} was expected and "
            f"{frames[i]} found"
        )
    return len(frames)


def _frames_dir_problem(rally: BadmintonRally, num_frames: int) -> str | None:
    """Why the rally's existing frames directory is not a finished extraction
    (completing "<frames_dir> exists but ..."), or None if it is one."""
    if not rally.frames_dir.is_dir():
        return "is not a directory"
    files = {p.name: p for p in rally.frames_dir.iterdir()}
    if files.keys() != {frame_file_name(i) for i in range(num_frames)}:
        return f"does not hold exactly the {num_frames} frames of {rally.label_path}"
    empty = sorted(name for name, p in files.items() if p.stat().st_size == 0)
    if empty:
        return f"its file {empty[0]} is empty"
    return None


def _extract_rally(
    rally: BadmintonRally, num_frames: int, jpeg_quality: int
) -> tuple[int, int]:
    """Decode the rally's video into its frames directory; return the number of
    surplus video frames dropped and the frame count the container reports."""
    temp = rally.frames_dir.with_name(rally.frames_dir.name + TEMP_SUFFIX)
    if temp.exists():
        logger.info("%s: removing the leftover of an interrupted run", temp)
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    capture = cv2.VideoCapture(str(rally.video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"cannot open video {rally.video_path}")
        reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        for i in range(num_frames):
            ok, image = capture.read()
            if not ok:
                raise ValueError(
                    f"{rally.group}/{rally.name}: {rally.video_path} decodes only "
                    f"{i} frames but {rally.label_path} has {num_frames} label rows"
                )
            path = temp / frame_file_name(i)
            if not cv2.imwrite(str(path), image, params):
                raise OSError(f"failed to write {path}")
        surplus = 0
        while capture.grab():
            surplus += 1
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    finally:
        capture.release()
    temp.rename(rally.frames_dir)
    return surplus, reported
