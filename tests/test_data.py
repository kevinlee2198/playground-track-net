import torch
import numpy as np
import pytest
from data.heatmap import generate_heatmap
from data.dataset import TrackNetDataset
from data.transforms import HorizontalFlip, FrameColorJitter, Mixup


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


class TestHorizontalFlip:
    def test_flip_reverses_width_dimension(self):
        frames = torch.arange(9 * 4 * 6, dtype=torch.float32).reshape(9, 4, 6)
        heatmaps = torch.arange(3 * 4 * 6, dtype=torch.float32).reshape(3, 4, 6)
        flip = HorizontalFlip(p=1.0)
        f_frames, f_heatmaps = flip(frames, heatmaps)
        assert torch.equal(f_frames, frames.flip(-1))
        assert torch.equal(f_heatmaps, heatmaps.flip(-1))

    def test_no_flip_when_p_zero(self):
        frames = torch.randn(9, 288, 512)
        heatmaps = torch.randn(3, 288, 512)
        flip = HorizontalFlip(p=0.0)
        f_frames, f_heatmaps = flip(frames, heatmaps)
        assert torch.equal(f_frames, frames)
        assert torch.equal(f_heatmaps, heatmaps)

    def test_shapes_preserved(self):
        frames = torch.randn(9, 288, 512)
        heatmaps = torch.randn(3, 288, 512)
        flip = HorizontalFlip(p=1.0)
        f_frames, f_heatmaps = flip(frames, heatmaps)
        assert f_frames.shape == frames.shape
        assert f_heatmaps.shape == heatmaps.shape


class TestFrameColorJitter:
    def test_heatmaps_unchanged(self):
        frames = torch.rand(9, 288, 512)
        heatmaps = torch.rand(3, 288, 512)
        jitter = FrameColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        _, out_heatmaps = jitter(frames, heatmaps)
        assert torch.equal(out_heatmaps, heatmaps)

    def test_frames_modified(self):
        torch.manual_seed(0)
        frames = torch.full((9, 288, 512), 0.5)
        heatmaps = torch.zeros(3, 288, 512)
        jitter = FrameColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        out_frames, _ = jitter(frames, heatmaps)
        assert out_frames.shape == frames.shape

    def test_output_clamped_to_01(self):
        frames = torch.rand(9, 288, 512)
        heatmaps = torch.zeros(3, 288, 512)
        jitter = FrameColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        out_frames, _ = jitter(frames, heatmaps)
        assert out_frames.min() >= 0.0
        assert out_frames.max() <= 1.0


class TestMixup:
    def test_mixup_blends_frames_and_heatmaps(self):
        torch.manual_seed(42)
        frames_a = torch.ones(9, 4, 6)
        heatmaps_a = torch.ones(3, 4, 6)
        frames_b = torch.zeros(9, 4, 6)
        heatmaps_b = torch.zeros(3, 4, 6)
        mixup = Mixup(alpha=1.0)
        f_out, h_out = mixup(frames_a, heatmaps_a, frames_b, heatmaps_b)
        assert f_out.min() >= 0.0
        assert f_out.max() <= 1.0
        lam = f_out[0, 0, 0].item()
        assert torch.allclose(f_out, torch.full_like(f_out, lam))
        assert torch.allclose(h_out, torch.full_like(h_out, lam))

    def test_mixup_output_shapes(self):
        frames_a = torch.randn(9, 288, 512)
        heatmaps_a = torch.randn(3, 288, 512)
        frames_b = torch.randn(9, 288, 512)
        heatmaps_b = torch.randn(3, 288, 512)
        mixup = Mixup(alpha=1.0)
        f_out, h_out = mixup(frames_a, heatmaps_a, frames_b, heatmaps_b)
        assert f_out.shape == (9, 288, 512)
        assert h_out.shape == (3, 288, 512)

    def test_mixup_lambda_from_beta_distribution(self):
        torch.manual_seed(0)
        mixup = Mixup(alpha=1.0)
        frames_a = torch.ones(9, 2, 2)
        heatmaps_a = torch.ones(3, 2, 2)
        frames_b = torch.zeros(9, 2, 2)
        heatmaps_b = torch.zeros(3, 2, 2)
        lambdas = []
        for _ in range(100):
            f_out, _ = mixup(frames_a, heatmaps_a, frames_b, heatmaps_b)
            lambdas.append(f_out[0, 0, 0].item())
        mean_lam = sum(lambdas) / len(lambdas)
        assert 0.3 < mean_lam < 0.7
