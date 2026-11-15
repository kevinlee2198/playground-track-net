from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.heatmap import generate_heatmap
from data.labels import load_labels

FRAMES_PER_SAMPLE = 3
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
STRIDES = (1, 3)


class TrackNetDataset(Dataset):
    """Dataset yielding windows of three consecutive frames with their heatmaps.

    Frame files are joined to labels by integer file stem (``0004.jpg`` is frame 4).
    Label coordinates are in source-video pixels and are rescaled to
    ``width`` x ``height``; the source size is read from the first frame.

    Windows start every ``stride`` frames and never run past the end of the clip,
    so trailing frames that do not fill a window are dropped. A clip shorter than
    three frames gives one window padded by repeating its last frame.
    """

    def __init__(
        self,
        frames_dir: str | Path,
        label_path: str | Path,
        height: int = 288,
        width: int = 512,
        radius: float = 2.5,
        stride: int = 3,
        transform: Callable | None = None,
    ) -> None:
        if stride not in STRIDES:
            raise ValueError(f"stride must be one of {STRIDES}, got {stride}")
        self.frames_dir = Path(frames_dir)
        self.height = height
        self.width = width
        self.radius = radius
        self.stride = stride
        self.transform = transform

        self.labels = load_labels(label_path)
        frame_files = {
            int(p.stem): p
            for p in self.frames_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        }
        label_frames = {label.frame for label in self.labels}
        if label_frames != frame_files.keys():
            missing_files = sorted(label_frames - frame_files.keys())
            unlabelled = sorted(frame_files.keys() - label_frames)
            raise ValueError(
                f"{self.frames_dir}: frame files and labels in {label_path} differ; "
                f"labelled frames without a file: {missing_files[:10]}, "
                f"files without a label: {unlabelled[:10]}"
            )
        if not self.labels:
            raise ValueError(f"{self.frames_dir}: clip has no frames")
        self.frame_paths = [frame_files[label.frame] for label in self.labels]
        self.num_frames = len(self.frame_paths)

        source_height, source_width = self._imread(0).shape[:2]
        self.source_size: tuple[int, int] = (source_width, source_height)
        self._scale_x = width / source_width
        self._scale_y = height / source_height

        last_start = max(self.num_frames - FRAMES_PER_SAMPLE + 1, 1)
        self.window_starts = list(range(0, last_start, stride))

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.window_starts[idx]
        last = self.num_frames - 1
        frame_indices = [
            min(start + offset, last) for offset in range(FRAMES_PER_SAMPLE)
        ]

        frame_tensors = []
        heatmap_list = []
        for fi in frame_indices:
            frame_tensors.append(self._read_frame(fi))

            label = self.labels[fi]
            heatmap_list.append(
                generate_heatmap(
                    x=label.x * self._scale_x,
                    y=label.y * self._scale_y,
                    visibility=int(label.visible),
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

    def _imread(self, fi: int) -> np.ndarray:
        """Read a frame from disk as a BGR uint8 array at its source size."""
        img = cv2.imread(str(self.frame_paths[fi]))
        if img is None:
            raise RuntimeError(f"Failed to read image: {self.frame_paths[fi]}")
        return img

    def _read_frame(self, fi: int) -> torch.Tensor:
        """Read a single frame from disk and return a (C, H, W) float32 tensor."""
        img = cv2.cvtColor(self._imread(fi), cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
