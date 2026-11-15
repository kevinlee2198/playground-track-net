"""Ball-position label loading for the CSV layouts used by our datasets."""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Column names for (frame, visibility, x, y), keyed by layout. Extra columns are ignored.
_LAYOUTS: dict[str, tuple[str, str, str, str]] = {
    "internal": ("Frame", "Visibility", "X", "Y"),
    "tennis": ("file name", "visibility", "x-coordinate", "y-coordinate"),
}


@dataclass(frozen=True)
class Label:
    frame: int  # frame id; for the tennis layout, int(stem of "file name")
    visible: bool
    x: float  # source-video pixels; 0.0 when not visible
    y: float


def load_labels(
    path: str | Path, visible_flags: tuple[int, ...] = (1, 2)
) -> list[Label]:
    """Sorted by frame. Layout is detected from the header row.

    A row is visible when its visibility value is in ``visible_flags``. Visible rows
    must have numeric coordinates; rows that are not visible report ``x = y = 0.0``
    whatever the file holds.
    """
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        frame_col, vis_col, x_col, y_col = _detect_columns(reader.fieldnames, path)

        labels: list[Label] = []
        seen: set[int] = set()
        for row in reader:
            where = f"{path}, line {reader.line_num}"
            try:
                frame = int(Path(row[frame_col]).stem)
                visible = int(row[vis_col]) in visible_flags
            except (TypeError, ValueError) as e:
                raise ValueError(f"{where}: cannot parse frame/visibility: {e}") from e
            if frame in seen:
                raise ValueError(f"{where}: duplicate frame {frame}")
            seen.add(frame)

            if visible:
                x, y = _parse_coord(row[x_col]), _parse_coord(row[y_col])
                if x is None or y is None:
                    raise ValueError(
                        f"{where}: visible frame {frame} has missing or invalid "
                        f"coordinates ({row[x_col]!r}, {row[y_col]!r})"
                    )
            else:
                x = y = 0.0
            labels.append(Label(frame=frame, visible=visible, x=x, y=y))

    labels.sort(key=lambda label: label.frame)
    return labels


def visibility_counts(path: str | Path) -> dict[int, int]:
    """Number of rows for each raw visibility value, in ascending value order.

    Unlike ``load_labels``, values are not mapped to visible / not visible, so a
    value the loader would silently treat as "not visible" shows up here.
    """
    path = Path(path)
    counts: Counter[int] = Counter()
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        _, vis_col, _, _ = _detect_columns(reader.fieldnames, path)
        for row in reader:
            try:
                counts[int(row[vis_col])] += 1
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{path}, line {reader.line_num}: cannot parse visibility: {e}"
                ) from e
    return dict(sorted(counts.items()))


def _detect_columns(
    fieldnames: Sequence[str] | None, path: Path
) -> tuple[str, str, str, str]:
    present = set(fieldnames or ())
    for columns in _LAYOUTS.values():
        if present.issuperset(columns):
            return columns
    expected = " or ".join(repr(", ".join(cols)) for cols in _LAYOUTS.values())
    raise ValueError(
        f"{path}: unrecognised header {fieldnames!r}; expected columns {expected}"
    )


def _parse_coord(value: str | None) -> float | None:
    """Return the coordinate as a float, or None if it is empty, NaN or non-numeric."""
    try:
        coord = float(value or "")
    except ValueError:
        return None
    return None if math.isnan(coord) else coord
