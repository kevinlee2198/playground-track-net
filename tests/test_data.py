import torch
import numpy as np
import pytest
from data.heatmap import generate_heatmap
from data.dataset import TrackNetDataset


class TestGenerateHeatmap:
    def test_visible_ball_returns_circle(self):
        heatmap = generate_heatmap(x=256, y=144, visibility=1, height=288, width=512, radius=30)
        assert heatmap.shape == (288, 512)
        assert heatmap.dtype == torch.float32
        assert heatmap[144, 256] == 1.0
        assert heatmap[0, 0] == 0.0

    def test_invisible_ball_returns_zeros(self):
        heatmap = generate_heatmap(x=0, y=0, visibility=0, height=288, width=512, radius=30)
        assert heatmap.shape == (288, 512)
        assert heatmap.sum() == 0.0

    def test_partially_occluded_still_labeled(self):
        heatmap = generate_heatmap(x=100, y=100, visibility=2, height=288, width=512, radius=30)
        assert heatmap[100, 100] == 1.0
        assert heatmap.sum() > 0.0

    def test_circle_radius(self):
        heatmap = generate_heatmap(x=256, y=144, visibility=1, height=288, width=512, radius=30)
        assert heatmap[144, 256 + 30] == 1.0
        assert heatmap[144, 256 + 31] == 0.0

    def test_ball_at_edge_clips_to_image(self):
        heatmap = generate_heatmap(x=5, y=5, visibility=1, height=288, width=512, radius=30)
        assert heatmap.shape == (288, 512)
        assert heatmap[5, 5] == 1.0
        centered = generate_heatmap(x=256, y=144, visibility=1, height=288, width=512, radius=30)
        assert heatmap.sum() < centered.sum()

    def test_values_are_binary(self):
        heatmap = generate_heatmap(x=200, y=100, visibility=1, height=288, width=512, radius=30)
        unique_vals = torch.unique(heatmap)
        assert all(v in (0.0, 1.0) for v in unique_vals)


class TestTrackNetDataset:
    def test_length_with_nine_frames(self, sample_frames_dir):
        frames_dir, csv_path = sample_frames_dir
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        assert len(ds) == 3

    def test_getitem_shapes(self, sample_frames_dir):
        frames_dir, csv_path = sample_frames_dir
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        frames, heatmaps = ds[0]
        assert frames.shape == (9, 288, 512)
        assert heatmaps.shape == (3, 288, 512)

    def test_getitem_dtypes(self, sample_frames_dir):
        frames_dir, csv_path = sample_frames_dir
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        frames, heatmaps = ds[0]
        assert frames.dtype == torch.float32
        assert heatmaps.dtype == torch.float32

    def test_frame_values_normalized(self, sample_frames_dir):
        frames_dir, csv_path = sample_frames_dir
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        frames, _ = ds[0]
        assert frames.min() >= 0.0
        assert frames.max() <= 1.0

    def test_invisible_frame_heatmap_is_zeros(self, sample_frames_dir):
        frames_dir, csv_path = sample_frames_dir
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        _, heatmaps = ds[1]
        assert heatmaps[0].sum() == 0.0
        assert heatmaps[1].sum() > 0.0

    def test_all_samples_accessible(self, sample_frames_dir):
        frames_dir, csv_path = sample_frames_dir
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        for i in range(len(ds)):
            frames, heatmaps = ds[i]
            assert frames.shape == (9, 288, 512)
            assert heatmaps.shape == (3, 288, 512)


import csv
import cv2


class TestDatasetBoundaryPadding:
    def test_single_frame_dataset(self, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        img = np.random.randint(0, 256, (288, 512, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "00000.jpg"), img)
        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Visibility", "X", "Y"])
            writer.writerow([0, 1, 256, 144])
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        assert len(ds) == 1
        frames, heatmaps = ds[0]
        assert frames.shape == (9, 288, 512)
        assert torch.equal(frames[:3], frames[3:6])
        assert torch.equal(frames[:3], frames[6:9])

    def test_four_frames_produces_two_samples(self, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        labels = []
        for i in range(4):
            img = np.random.randint(0, 256, (288, 512, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), img)
            labels.append((i, 1, 100 + i * 10, 80 + i * 5))
        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Visibility", "X", "Y"])
            for row in labels:
                writer.writerow(row)
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        assert len(ds) == 2
        frames, heatmaps = ds[1]
        assert frames.shape == (9, 288, 512)
        assert torch.equal(frames[3:6], frames[6:9])
