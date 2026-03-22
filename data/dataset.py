from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from data.heatmap import generate_heatmap

FRAMES_PER_SAMPLE = 3
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class TrackNetDataset(Dataset):
    """Dataset yielding triplets of consecutive frames with corresponding heatmaps."""

    def __init__(
        self,
        frames_dir: str | Path,
        label_path: str | Path,
        height: int = 288,
        width: int = 512,
        radius: int = 30,
        transform: Callable | None = None,
    ) -> None:
        self.frames_dir = Path(frames_dir)
        self.height = height
        self.width = width
        self.radius = radius
        self.transform = transform

        self.labels = _load_labels(label_path)
        self.frame_paths = sorted(
            p for p in self.frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        self.num_frames = len(self.frame_paths)

        if len(self.labels) != self.num_frames:
            raise ValueError(
                f"Label count ({len(self.labels)}) != frame count ({self.num_frames}). "
                f"CSV and frames directory must have the same number of entries."
            )

    def __len__(self) -> int:
        return (self.num_frames + FRAMES_PER_SAMPLE - 1) // FRAMES_PER_SAMPLE

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * FRAMES_PER_SAMPLE
        last = self.num_frames - 1
        frame_indices = [
            min(max(start + offset, 0), last)
            for offset in range(FRAMES_PER_SAMPLE)
        ]

        frame_tensors = []
        heatmap_list = []
        for fi in frame_indices:
            frame_tensors.append(self._read_frame(fi))

            label = self.labels[fi]
            heatmap_list.append(
                generate_heatmap(
                    x=label["x"],
                    y=label["y"],
                    visibility=label["visibility"],
                    height=self.height,
                    width=self.width,
                    radius=self.radius,
                )
            )

        frames = torch.cat(frame_tensors, dim=0)
        heatmaps = torch.stack(heatmap_list, dim=0)

        if self.transform is not None:
            frames, heatmaps = self.transform(frames, heatmaps)
        return frames, heatmaps

    def _read_frame(self, fi: int) -> torch.Tensor:
        """Read a single frame from disk and return a (C, H, W) float32 tensor."""
        img = cv2.imread(str(self.frame_paths[fi]))
        if img is None:
            raise RuntimeError(f"Failed to read image: {self.frame_paths[fi]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def _load_labels(label_path: str | Path) -> list[dict]:
    """Read the CSV label file and return rows sorted by frame number."""
    with open(label_path) as f:
        reader = csv.DictReader(f)
        labels = [
            {
                "frame": int(row["Frame"]),
                "visibility": int(row["Visibility"]),
                "x": int(row["X"]),
                "y": int(row["Y"]),
            }
            for row in reader
        ]
    labels.sort(key=lambda r: r["frame"])
    return labels
