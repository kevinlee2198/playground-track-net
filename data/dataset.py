from __future__ import annotations
import csv
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
from data.heatmap import generate_heatmap


class TrackNetDataset(Dataset):
    def __init__(self, frames_dir: str | Path, label_path: str | Path,
                 height: int = 288, width: int = 512, radius: int = 30, transform=None):
        self.frames_dir = Path(frames_dir)
        self.height = height
        self.width = width
        self.radius = radius
        self.transform = transform
        self.labels: list[dict] = []
        with open(label_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.labels.append({
                    "frame": int(row["Frame"]), "visibility": int(row["Visibility"]),
                    "x": int(row["X"]), "y": int(row["Y"]),
                })
        self.labels.sort(key=lambda r: r["frame"])
        self.frame_paths = sorted(self.frames_dir.glob("*.*"))
        self.num_frames = len(self.frame_paths)
        self._num_samples = (self.num_frames + 2) // 3

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * 3
        frame_indices = [min(max(i, 0), self.num_frames - 1) for i in [start, start + 1, start + 2]]
        frame_tensors = []
        for fi in frame_indices:
            img = cv2.imread(str(self.frame_paths[fi]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (self.height, self.width):
                img = cv2.resize(img, (self.width, self.height))
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            frame_tensors.append(tensor)
        frames = torch.cat(frame_tensors, dim=0)
        heatmap_list = []
        for fi in frame_indices:
            label = self.labels[fi]
            hm = generate_heatmap(x=label["x"], y=label["y"], visibility=label["visibility"],
                                  height=self.height, width=self.width, radius=self.radius)
            heatmap_list.append(hm)
        heatmaps = torch.stack(heatmap_list, dim=0)
        if self.transform is not None:
            frames, heatmaps = self.transform(frames, heatmaps)
        return frames, heatmaps
