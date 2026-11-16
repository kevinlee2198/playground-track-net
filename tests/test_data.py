import csv
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from data.background import background_indices, median_frame
from data.dataset import TrackNetDataset
from data.heatmap import generate_heatmap, model_to_source, source_to_model
from data.labels import load_labels, visibility_counts
from data.sources import build_dataset, discover_tennis_clips, split_clips
from data.transforms import Compose, FrameColorJitter, HorizontalFlip, Mixup
from inference.video_preprocess import preprocess_frame


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


class TestGenerateHeatmap:
    def test_visible_ball_returns_circle(self):
        heatmap = generate_heatmap(
            x=256, y=144, visibility=1, height=288, width=512, radius=30
        )
        assert heatmap.shape == (288, 512)
        assert heatmap.dtype == torch.float32
        assert heatmap[144, 256] == 1.0
        assert heatmap[0, 0] == 0.0

    def test_invisible_ball_returns_zeros(self):
        heatmap = generate_heatmap(
            x=0, y=0, visibility=0, height=288, width=512, radius=30
        )
        assert heatmap.shape == (288, 512)
        assert heatmap.sum() == 0.0

    def test_partially_occluded_still_labeled(self):
        heatmap = generate_heatmap(
            x=100, y=100, visibility=2, height=288, width=512, radius=30
        )
        assert heatmap[100, 100] == 1.0
        assert heatmap.sum() > 0.0

    def test_circle_radius(self):
        heatmap = generate_heatmap(
            x=256, y=144, visibility=1, height=288, width=512, radius=30
        )
        assert heatmap[144, 256 + 30] == 1.0
        assert heatmap[144, 256 + 31] == 0.0

    def test_ball_at_edge_clips_to_image(self):
        heatmap = generate_heatmap(
            x=5, y=5, visibility=1, height=288, width=512, radius=30
        )
        assert heatmap.shape == (288, 512)
        assert heatmap[5, 5] == 1.0
        centered = generate_heatmap(
            x=256, y=144, visibility=1, height=288, width=512, radius=30
        )
        assert heatmap.sum() < centered.sum()

    def test_values_are_binary(self):
        heatmap = generate_heatmap(
            x=200, y=100, visibility=1, height=288, width=512, radius=30
        )
        unique_vals = torch.unique(heatmap)
        assert all(v in (0.0, 1.0) for v in unique_vals)

    def test_default_radius_is_2_5(self):
        assert generate_heatmap(x=256, y=144, visibility=1).sum() == 21

    def test_float_centre_at_edge_is_clipped(self):
        hm = generate_heatmap(x=511.84, y=0.2, visibility=1)
        assert hm[0, 511] == 1.0
        assert hm[:, :500].sum() == 0

    def test_label_outside_frame_gives_empty_target(self):
        assert generate_heatmap(x=-50.0, y=10.0, visibility=1).sum() == 0


class TestCoordinateMapping:
    @pytest.mark.parametrize(
        ("model_size", "source_size"), [(512, 1280), (288, 720), (128, 1280)]
    )
    def test_model_to_source_to_model_is_identity(self, model_size, source_size):
        for x in (-3.25, 0.0, 0.5, 17.3, 255.7, model_size - 1.0):
            src = model_to_source(x, model_size, source_size)
            assert abs(source_to_model(src, source_size, model_size) - x) < 1e-9

    def test_pixel_centres_map_onto_pixel_centres(self):
        # The centre of the frame is the centre of the frame at any resolution.
        assert source_to_model(639.5, 1280, 512) == pytest.approx(255.5)
        assert model_to_source(255.5, 512, 1280) == pytest.approx(639.5)


def _intensity_centroid(image: torch.Tensor) -> tuple[float, float]:
    """(x, y) centroid of a (C, H, W) image weighted by its summed intensity."""
    weight = image.sum(dim=0).double()
    ys = torch.arange(weight.shape[0], dtype=torch.float64).unsqueeze(1)
    xs = torch.arange(weight.shape[1], dtype=torch.float64).unsqueeze(0)
    total = weight.sum()
    return (
        float((weight * xs).sum() / total),
        float((weight * ys).sum() / total),
    )


class TestTargetAlignsWithResizedBall:
    """The heatmap target must sit on the ball as the resized frame shows it."""

    SOURCE_BALL = (640, 360)  # pixel of a 1280x720 frame the disc is centred on

    @pytest.fixture
    def disc_clip(self, tmp_path: Path) -> tuple[Path, Path]:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.circle(img, self.SOURCE_BALL, 6, (255, 255, 255), -1)
        cv2.imwrite(str(frames_dir / "0000.png"), img)  # lossless
        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Visibility", "X", "Y"])
            writer.writerow([0, 1, *self.SOURCE_BALL])
        return frames_dir, csv_path

    def _mapped_label(self) -> tuple[float, float]:
        x, y = self.SOURCE_BALL
        return source_to_model(x, 1280, 512), source_to_model(y, 720, 288)

    def test_target_centred_on_resized_ball(self, disc_clip):
        frames, heatmaps = TrackNetDataset(*disc_clip)[0]
        x, y = self._mapped_label()
        ball_x, ball_y = _intensity_centroid(frames[:3])
        assert abs(ball_x - x) < 0.05 and abs(ball_y - y) < 0.05, (ball_x, ball_y)
        assert torch.equal(heatmaps[0], generate_heatmap(x, y, visibility=1))

    def test_target_centred_on_resized_ball_after_flip(self, disc_clip):
        dataset = TrackNetDataset(*disc_clip, transform=HorizontalFlip(p=1.0))
        frames, heatmaps = dataset[0]
        x, y = self._mapped_label()
        ball_x, ball_y = _intensity_centroid(frames[:3])
        assert abs(ball_x - (511 - x)) < 0.05, ball_x  # mirrored: x -> 511 - x
        assert abs(ball_y - y) < 0.05, ball_y
        flipped = generate_heatmap(x, y, visibility=1).flip(-1)
        assert torch.equal(heatmaps[0], flipped)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _write_sample_dataset(
    frames_dir: Path,
    csv_path: Path,
    labels: list[tuple[int, int, int, int]],
) -> None:
    """Write synthetic frames and a CSV label file for testing."""
    frames_dir.mkdir(exist_ok=True)
    for frame_idx, vis, bx, by in labels:
        img = np.random.randint(0, 256, (288, 512, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"{frame_idx:05d}.jpg"), img)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Visibility", "X", "Y"])
        for row in labels:
            writer.writerow(row)


class TestLoadLabels:
    @staticmethod
    def _write_csv(path: Path, header: list[str], rows: list[tuple]) -> Path:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    @staticmethod
    def _assert_load_fails(path: Path, match: str) -> None:
        """load_labels raises a ValueError matching ``match`` that names the file."""
        with pytest.raises(ValueError, match=match) as exc_info:
            load_labels(path)
        assert str(path) in str(exc_info.value)

    def test_tennis_layout(self, tennis_root):
        labels = load_labels(tennis_root / "game1" / "Clip1" / "Label.csv")
        visible = [label.visible for label in labels]
        assert [label.frame for label in labels] == [0, 1, 2, 3, 4, 5]
        assert visible == [True, True, True, False, False, True]
        assert (labels[0].x, labels[0].y) == (640.0, 360.0)
        assert (labels[3].x, labels[3].y) == (0.0, 0.0)
        assert (labels[4].x, labels[4].y) == (0.0, 0.0)

    def test_visibility_counts_are_raw_values(self, tennis_root):
        counts = visibility_counts(tennis_root / "game1" / "Clip1" / "Label.csv")
        assert counts == {0: 1, 1: 3, 2: 1, 3: 1}

    def test_internal_layout(self, sample_frames_dir):
        _, csv_path = sample_frames_dir
        visible = [label.visible for label in load_labels(csv_path)]
        assert visible == [True, True, True, False, True, True, True, True, True]

    def test_visible_row_without_coordinates_raises(self, tmp_path):
        path = self._write_csv(
            tmp_path / "Label.csv",
            ["file name", "visibility", "x-coordinate", "y-coordinate", "status"],
            [("0000.jpg", 1, 640, 360, 0), ("0001.jpg", 1, "", "", "")],
        )
        self._assert_load_fails(path, match="coordinates")

    @pytest.mark.parametrize("coord", ["inf", "-inf"])
    def test_visible_row_with_infinite_coordinate_raises(self, tmp_path, coord):
        path = self._write_csv(
            tmp_path / "Label.csv",
            ["file name", "visibility", "x-coordinate", "y-coordinate", "status"],
            [("0000.jpg", 1, 640, 360, 0), ("0001.jpg", 1, coord, 355, 0)],
        )
        self._assert_load_fails(path, match="coordinates")

    def test_duplicate_frame_raises(self, tmp_path):
        path = self._write_csv(
            tmp_path / "labels.csv",
            ["Frame", "Visibility", "X", "Y"],
            [(0, 1, 100, 80), (1, 1, 110, 85), (1, 1, 120, 90)],
        )
        self._assert_load_fails(path, match="duplicate")

    def test_unknown_header_raises(self, tmp_path):
        path = self._write_csv(
            tmp_path / "labels.csv",
            ["frame_id", "vis", "cx", "cy"],
            [(0, 1, 100, 80)],
        )
        self._assert_load_fails(path, match="header")


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

    def test_labels_rescaled_from_source_resolution(self, tennis_root):
        clip = tennis_root / "game1" / "Clip1"
        ds = TrackNetDataset(clip, clip / "Label.csv")
        assert ds.source_size == (1280, 720)
        _, heatmaps = ds[0]
        ys, xs = torch.where(heatmaps[0] > 0)
        # (640 + 0.5) * 512 / 1280 - 0.5 = 255.7; (360 + 0.5) * 288 / 720 - 0.5 = 143.7.
        # The 20-pixel disc drawn there has its centroid at (255.75, 143.75).
        assert abs(xs.float().mean().item() - 255.7) <= 0.1
        assert abs(ys.float().mean().item() - 143.7) <= 0.1
        assert heatmaps[0].sum() == 20  # radius 2.5 at offset (0.7, 0.7) covers 20 px

    def test_occluded_and_invisible_frames_have_empty_targets(self, tennis_root):
        clip = tennis_root / "game1" / "Clip1"
        _, heatmaps = TrackNetDataset(clip, clip / "Label.csv")[1]  # frames 3, 4, 5
        # Frame 5 at (690, 335) maps to (275.7, 133.7): a 20-pixel disc.
        assert (
            heatmaps[0].sum() == 0
            and heatmaps[1].sum() == 0
            and heatmaps[2].sum() == 20
        )

    def test_stride_1_window_count_and_overlap(self, tennis_root):
        clip = tennis_root / "game1" / "Clip1"
        ds = TrackNetDataset(clip, clip / "Label.csv", stride=1)
        assert len(ds) == 4
        assert torch.equal(ds[1][0][:3], ds[0][0][3:6])
        assert not torch.equal(ds[1][0][:3], ds[0][0][:3])

    @pytest.mark.parametrize("stride", [1, 3])
    def test_short_clip_gives_one_padded_window(self, tmp_path, stride):
        _write_sample_dataset(
            tmp_path / "frames",
            tmp_path / "labels.csv",
            labels=[(0, 1, 100, 80), (1, 1, 300, 200)],
        )
        ds = TrackNetDataset(
            tmp_path / "frames", tmp_path / "labels.csv", stride=stride
        )
        assert len(ds) == 1
        frames, heatmaps = ds[0]
        assert frames.shape == (9, 288, 512)
        assert not torch.equal(frames[:3], frames[3:6])
        assert torch.equal(frames[3:6], frames[6:9])
        assert torch.equal(heatmaps[1], heatmaps[2])

    def test_invalid_stride_raises(self, tennis_root):
        clip = tennis_root / "game1" / "Clip1"
        with pytest.raises(ValueError, match="stride"):
            TrackNetDataset(clip, clip / "Label.csv", stride=2)

    def test_frame_label_mismatch_raises(self, tennis_root):
        clip = tennis_root / "game1" / "Clip1"
        (clip / "0005.jpg").unlink()
        with pytest.raises(ValueError, match="Clip1"):
            TrackNetDataset(clip, clip / "Label.csv")

    def test_image_without_label_row_raises(self, tennis_root):
        clip = tennis_root / "game1" / "Clip1"
        cv2.imwrite(str(clip / "0006.jpg"), np.zeros((720, 1280, 3), np.uint8))
        with pytest.raises(ValueError, match=r"Clip1.*files without a label: \[6\]"):
            TrackNetDataset(clip, clip / "Label.csv")

    def test_frames_match_inference_preprocessing_on_colour_images(self, tmp_path):
        # Random colours, so swapping BGR and RGB on either side changes the
        # tensor; 1280x720, so both sides resize.
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        rng = np.random.default_rng(0)
        paths = []
        for i in range(3):
            path = frames_dir / f"{i:04d}.png"
            cv2.imwrite(str(path), rng.integers(0, 256, (720, 1280, 3), np.uint8))
            paths.append(path)
        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Visibility", "X", "Y"])
            writer.writerows((i, 0, 0, 0) for i in range(3))

        frames, _ = TrackNetDataset(frames_dir, csv_path)[0]
        expected = []
        for path in paths:
            bgr = cv2.imread(str(path))  # as infer reads an image directory
            assert bgr is not None
            expected.append(preprocess_frame(bgr))
        assert torch.equal(frames, torch.from_numpy(np.concatenate(expected)))

    def test_image_with_non_integer_name_raises_naming_it(self, tennis_root):
        clip = tennis_root / "game1" / "Clip1"
        cv2.imwrite(str(clip / "cover.jpg"), np.zeros((720, 1280, 3), np.uint8))
        with pytest.raises(ValueError, match="cover.jpg"):
            TrackNetDataset(clip, clip / "Label.csv")


class TestDatasetBoundaryPadding:
    def test_single_frame_dataset(self, tmp_path):
        _write_sample_dataset(
            tmp_path / "frames",
            tmp_path / "labels.csv",
            labels=[(0, 1, 256, 144)],
        )
        ds = TrackNetDataset(
            frames_dir=tmp_path / "frames", label_path=tmp_path / "labels.csv"
        )
        assert len(ds) == 1
        frames, heatmaps = ds[0]
        assert frames.shape == (9, 288, 512)
        assert torch.equal(frames[:3], frames[3:6])
        assert torch.equal(frames[:3], frames[6:9])

    def test_four_frames_gives_one_window_without_tail_padding(self, tmp_path):
        labels = [(i, 1, 100 + i * 10, 80 + i * 5) for i in range(4)]
        _write_sample_dataset(
            tmp_path / "frames",
            tmp_path / "labels.csv",
            labels=labels,
        )
        ds = TrackNetDataset(
            frames_dir=tmp_path / "frames", label_path=tmp_path / "labels.csv"
        )
        assert len(ds) == 1
        frames, _ = ds[0]
        assert frames.shape == (9, 288, 512)
        assert not torch.equal(frames[3:6], frames[6:9])


# ---------------------------------------------------------------------------
# Median background frame
# ---------------------------------------------------------------------------


class TestBackgroundIndices:
    def test_short_clip_uses_every_frame(self):
        assert background_indices(5) == [0, 1, 2, 3, 4]

    def test_long_clip_spaces_indices_from_first_to_last(self):
        indices = background_indices(101, 50)
        assert len(indices) == 50
        assert indices[0] == 0 and indices[-1] == 100
        assert indices == sorted(set(indices))  # increasing, without repeats

    def test_single_frame(self):
        assert background_indices(1) == [0]


class TestMedianFrame:
    def test_uint8_median(self):
        frames = [np.full((2, 3, 3), v, dtype=np.uint8) for v in (10, 20, 250)]
        median = median_frame(frames)
        assert median.dtype == np.uint8
        assert median.shape == (2, 3, 3)
        assert np.all(median == 20)

    def test_integer_median_is_rounded(self):
        # The median of 1 and 2 is 1.5; rounding half to even gives 2.
        frames = [np.full((2, 3), v, dtype=np.uint8) for v in (1, 2)]
        assert np.all(median_frame(frames) == 2)

    def test_float_inputs_keep_float32(self):
        frames = [np.full((2, 3), v, dtype=np.float32) for v in (0.1, 0.2, 0.9)]
        median = median_frame(frames)
        assert median.dtype == np.float32
        assert np.allclose(median, 0.2)


def _write_moving_square_clip(
    tmp_path: Path, num_frames: int
) -> tuple[Path, Path, np.ndarray]:
    """A clip whose frames are one fixed random scene with a bright 6x6 square
    at a different place in each frame. Returns (frames_dir, csv_path, scene),
    the scene as an RGB uint8 array."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    rng = np.random.default_rng(0)
    scene = rng.integers(0, 200, (288, 512, 3), dtype=np.uint8)  # RGB
    labels = []
    for i in range(num_frames):
        img = scene.copy()
        x, y = 20 + 50 * i, 30 + 25 * i
        img[y : y + 6, x : x + 6] = 255
        # PNG is lossless, so the median recovers the scene exactly.
        cv2.imwrite(
            str(frames_dir / f"{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )
        labels.append((i, 1, x + 3, y + 3))
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Visibility", "X", "Y"])
        writer.writerows(labels)
    return frames_dir, csv_path, scene


class TestDatasetBackground:
    def test_background_is_the_static_scene_in_every_window(self, tmp_path):
        frames_dir, csv_path, scene = _write_moving_square_clip(tmp_path, 9)
        with_bg = TrackNetDataset(frames_dir, csv_path, stride=1, background=True)
        without = TrackNetDataset(frames_dir, csv_path, stride=1)
        expected = torch.from_numpy(scene).permute(2, 0, 1).float() / 255.0
        first_background = with_bg[0][0][9:]
        for i in range(len(with_bg)):
            frames, heatmaps = with_bg[i]
            plain_frames, plain_heatmaps = without[i]
            assert frames.shape == (12, 288, 512)
            assert frames.dtype == torch.float32
            assert torch.allclose(frames[9:], expected, atol=1 / 255)
            assert torch.equal(frames[9:], first_background)
            assert torch.equal(frames[:9], plain_frames)
            assert torch.equal(heatmaps, plain_heatmaps)

    def test_background_stored_as_uint8_at_model_resolution(self, tmp_path):
        frames_dir, csv_path, _ = _write_moving_square_clip(tmp_path, 9)
        ds = TrackNetDataset(frames_dir, csv_path, background=True)
        assert ds.background is not None
        assert ds.background.dtype == np.uint8
        assert ds.background.shape == (288, 512, 3)

    def test_background_off_by_default(self, tmp_path):
        frames_dir, csv_path, _ = _write_moving_square_clip(tmp_path, 9)
        ds = TrackNetDataset(frames_dir, csv_path)
        assert ds.background is None
        assert ds[0][0].shape == (9, 288, 512)

    def test_two_frame_clip_gives_one_window_with_background(self, tmp_path):
        _write_sample_dataset(
            tmp_path / "frames",
            tmp_path / "labels.csv",
            labels=[(0, 1, 100, 80), (1, 1, 300, 200)],
        )
        ds = TrackNetDataset(
            tmp_path / "frames", tmp_path / "labels.csv", background=True
        )
        assert len(ds) == 1
        frames, _ = ds[0]
        assert frames.shape == (12, 288, 512)

    def test_build_dataset_passes_background_to_every_clip(self, tennis_root):
        clips = discover_tennis_clips(tennis_root)
        dataset = build_dataset(clips, stride=3, radius=2.5, background=True)
        clips_ds = [cast(TrackNetDataset, d) for d in dataset.datasets]
        assert all(d.background is not None for d in clips_ds)
        assert dataset[0][0].shape == (12, 288, 512)

    @staticmethod
    def _write_static_clip(tmp_path: Path, num_frames: int) -> tuple[Path, Path]:
        """A clip whose frames are all one random image, which is therefore
        its background, and is not left-right symmetric."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        img = np.random.default_rng(0).integers(0, 256, (288, 512, 3), dtype=np.uint8)
        assert not np.array_equal(img, img[:, ::-1])
        for i in range(num_frames):
            cv2.imwrite(str(frames_dir / f"{i:04d}.png"), img)
        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Visibility", "X", "Y"])
            writer.writerows((i, 1, 100, 80) for i in range(num_frames))
        return frames_dir, csv_path

    def test_colour_jitter_applies_to_the_background_too(self, tmp_path):
        frames_dir, csv_path = self._write_static_clip(tmp_path, 6)
        plain = TrackNetDataset(frames_dir, csv_path, background=True)
        jittered = TrackNetDataset(
            frames_dir, csv_path, background=True, transform=FrameColorJitter()
        )
        torch.manual_seed(0)
        frames, _ = jittered[0]
        assert not torch.allclose(frames, plain[0][0])  # the draw is not the identity
        for k in range(3):
            assert torch.allclose(frames[9:], frames[3 * k : 3 * k + 3], atol=1e-6), k

    def test_flip_applies_to_the_background_too(self, tmp_path):
        frames_dir, csv_path = self._write_static_clip(tmp_path, 6)
        plain = TrackNetDataset(frames_dir, csv_path, background=True)
        flipped = TrackNetDataset(
            frames_dir, csv_path, background=True, transform=HorizontalFlip(p=1.0)
        )
        background = plain[0][0][9:]
        assert torch.equal(flipped[0][0][9:], background.flip(-1))


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class TestHorizontalFlip:
    def test_flip_reverses_width_dimension(self):
        frames = torch.arange(9 * 4 * 6, dtype=torch.float32).reshape(9, 4, 6)
        heatmaps = torch.arange(3 * 4 * 6, dtype=torch.float32).reshape(3, 4, 6)
        flip = HorizontalFlip(p=1.0)
        f_frames, f_heatmaps = flip(frames, heatmaps)
        assert torch.equal(f_frames, frames.flip(-1))
        assert torch.equal(f_heatmaps, heatmaps.flip(-1))

    def test_flip_covers_the_background_channels(self):
        frames = torch.arange(12 * 4 * 6, dtype=torch.float32).reshape(12, 4, 6)
        heatmaps = torch.zeros(3, 4, 6)
        f_frames, _ = HorizontalFlip(p=1.0)(frames, heatmaps)
        assert torch.equal(f_frames, frames.flip(-1))

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

    def test_same_jitter_applied_to_all_three_frames(self):
        torch.manual_seed(0)
        frame = torch.rand(3, 32, 64)
        out, _ = FrameColorJitter()(frame.repeat(3, 1, 1), torch.zeros(3, 32, 64))
        a, b, c = out.chunk(3, dim=0)
        assert torch.allclose(a, b) and torch.allclose(b, c)
        assert not torch.allclose(a, frame)

    def test_same_jitter_applied_to_frames_and_background(self):
        torch.manual_seed(0)
        frame = torch.rand(3, 32, 64)
        out, _ = FrameColorJitter()(frame.repeat(4, 1, 1), torch.zeros(3, 32, 64))
        assert out.shape == (12, 32, 64)
        images = out.chunk(4, dim=0)
        assert all(torch.allclose(images[0], image) for image in images[1:])
        assert not torch.allclose(images[0], frame)

    def test_p_zero_returns_frames_unchanged(self):
        torch.manual_seed(0)
        jitter = FrameColorJitter(p=0.0)
        for _ in range(20):
            frames = torch.rand(9, 32, 64)
            out, _ = jitter(frames, torch.zeros(3, 32, 64))
            assert torch.equal(out, frames)

    def test_p_one_is_torchvision_colorjitter_with_no_extra_draw(self):
        frames = torch.rand(9, 32, 64)
        torch.manual_seed(3)
        out, _ = FrameColorJitter(p=1.0)(frames, torch.zeros(3, 32, 64))
        after = torch.rand(1)
        torch.manual_seed(3)
        batch = frames.unflatten(0, (-1, 3))
        expected = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)(batch)
        assert torch.equal(out, expected.clamp(0.0, 1.0).flatten(0, 1))
        # The same random-number stream: no draw for the p decision.
        assert torch.equal(torch.rand(1), after)

    def test_p_half_jitters_about_half_of_the_windows(self):
        torch.manual_seed(0)
        jitter = FrameColorJitter(p=0.5)
        changed = 0
        for _ in range(200):
            frames = torch.rand(9, 32, 64)
            out, _ = jitter(frames, torch.zeros(3, 32, 64))
            changed += not torch.equal(out, frames)
        assert 70 <= changed <= 130

    @pytest.mark.parametrize("p", [1.5, -0.1])
    def test_p_outside_unit_interval_raises(self, p):
        with pytest.raises(ValueError, match="p"):
            FrameColorJitter(p=p)

    def test_keeps_its_strengths_and_p(self):
        jitter = FrameColorJitter(brightness=0.1, contrast=0.2, saturation=0.4, p=0.5)
        assert jitter.jitter.brightness == pytest.approx((0.9, 1.1))
        assert jitter.jitter.contrast == pytest.approx((0.8, 1.2))
        assert jitter.jitter.saturation == pytest.approx((0.6, 1.4))
        assert jitter.p == 0.5


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


class TestCompose:
    def test_compose_applies_in_order(self):
        flip = HorizontalFlip(p=1.0)
        compose = Compose([flip, flip])
        frames = torch.randn(9, 4, 6)
        heatmaps = torch.randn(3, 4, 6)
        f_out, h_out = compose(frames, heatmaps)
        assert torch.allclose(f_out, frames)
        assert torch.allclose(h_out, heatmaps)


# ---------------------------------------------------------------------------
# Sources: clip discovery, splits, multi-clip datasets
# ---------------------------------------------------------------------------


class TestSources:
    def test_discover_tennis_clips(self, tennis_root):
        clips = discover_tennis_clips(tennis_root)
        assert [(c.group, c.frames_dir.name) for c in clips] == [
            ("game1", "Clip1"),
            ("game1", "Clip2"),
            ("game2", "Clip1"),
            ("game3", "Clip1"),
        ]
        assert all(c.label_path.name == "Label.csv" for c in clips)

    def test_natural_order(self, tmp_path):
        for game, clip in [
            ("game2", "Clip1"),
            ("game10", "Clip1"),
            ("game1", "Clip10"),
            ("game1", "Clip2"),
        ]:
            clip_dir = tmp_path / game / clip
            clip_dir.mkdir(parents=True)
            (clip_dir / "Label.csv").touch()
        clips = discover_tennis_clips(tmp_path)
        assert [(c.group, c.frames_dir.name) for c in clips] == [
            ("game1", "Clip2"),
            ("game1", "Clip10"),
            ("game2", "Clip1"),
            ("game10", "Clip1"),
        ]

    def test_split_by_group_never_shares_a_game(self, tennis_root):
        s = split_clips(
            discover_tennis_clips(tennis_root),
            val_groups=["game2"],
            test_groups=["game3"],
        )
        assert [len(s[k]) for k in ("train", "val", "test")] == [2, 1, 1]
        assert {c.group for c in s["train"]} == {"game1"}
        assert {c.group for c in s["val"]} == {"game2"}
        assert {c.group for c in s["test"]} == {"game3"}

    def test_build_dataset_lengths(self, tennis_root):
        clips = discover_tennis_clips(tennis_root)
        assert len(build_dataset(clips, stride=1, radius=2.5)) == 16  # 4 clips x 4
        assert len(build_dataset(clips, stride=3, radius=2.5)) == 8  # 4 clips x 2

    def test_unknown_group_raises(self, tennis_root):
        clips = discover_tennis_clips(tennis_root)
        with pytest.raises(ValueError, match="game9"):
            split_clips(clips, val_groups=["game9"], test_groups=[])

    def test_overlapping_groups_raises(self, tennis_root):
        clips = discover_tennis_clips(tennis_root)
        with pytest.raises(ValueError, match="game2"):
            split_clips(clips, val_groups=["game2"], test_groups=["game2"])

    def test_empty_train_raises(self, tennis_root):
        clips = discover_tennis_clips(tennis_root)
        with pytest.raises(ValueError, match="train"):
            split_clips(clips, val_groups=["game1", "game2"], test_groups=["game3"])

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            discover_tennis_clips(tmp_path / "does-not-exist")

    def test_root_without_clips_raises(self, tennis_root):
        too_deep = tennis_root / "game1"  # holds clips, not games
        with pytest.raises(ValueError, match="<game>/<clip>/Label.csv") as err:
            discover_tennis_clips(too_deep)
        assert str(too_deep) in str(err.value)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestPackageImports:
    def test_imports(self):
        from data import (
            Label,
            TrackNetDataset,
            generate_heatmap,
            load_labels,
        )

        assert TrackNetDataset is not None
        assert generate_heatmap is not None
        assert Label is not None
        assert load_labels is not None

    def test_background_helpers_exported(self):
        import data

        assert data.background_indices is background_indices
        assert data.median_frame is median_frame


class TestDataLoader:
    def test_batch_shapes(self, sample_frames_dir):
        frames_dir, csv_path = sample_frames_dir
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        loader = DataLoader(ds, batch_size=2, pin_memory=False, num_workers=0)
        batch_frames, batch_heatmaps = next(iter(loader))
        assert batch_frames.shape == (2, 9, 288, 512)
        assert batch_heatmaps.shape == (2, 3, 288, 512)
        assert batch_frames.dtype == torch.float32
        assert batch_heatmaps.dtype == torch.float32


class TestDatasetWithTransform:
    def test_transform_is_called(self, sample_frames_dir):
        frames_dir, csv_path = sample_frames_dir
        flip = HorizontalFlip(p=1.0)
        ds_no_flip = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        ds_flip = TrackNetDataset(
            frames_dir=frames_dir, label_path=csv_path, transform=flip
        )
        frames_orig, heatmaps_orig = ds_no_flip[0]
        frames_flip, heatmaps_flip = ds_flip[0]
        assert torch.equal(frames_flip, frames_orig.flip(-1))
        assert torch.equal(heatmaps_flip, heatmaps_orig.flip(-1))
