import numpy as np
import pytest
from inference.video_preprocess import extract_frames, preprocess_frame, create_sliding_windows


class TestPreprocessFrame:
    def test_resize_and_normalize(self):
        """Frame is resized to 512x288 and normalized to [0,1]."""
        frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
        result = preprocess_frame(frame)
        assert result.shape == (3, 288, 512)  # C, H, W
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert result.dtype == np.float32

    def test_preserves_content(self):
        """A solid-color frame stays solid after preprocessing."""
        frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        result = preprocess_frame(frame)
        expected = 128.0 / 255.0
        np.testing.assert_allclose(result, expected, atol=1e-2)


class TestCreateSlidingWindows:
    def test_basic_stride3(self):
        """9 frames -> 3 windows of 3, stride=3, no overlap."""
        frames = [np.zeros((3, 288, 512), dtype=np.float32) for _ in range(9)]
        windows = create_sliding_windows(frames, window_size=3, stride=3)
        assert len(windows) == 3
        for w in windows:
            assert w.shape == (9, 288, 512)  # 3 frames * 3 channels

    def test_boundary_padding_start(self):
        """With 2 frames, first window pads by duplicating frame 0."""
        frames = [np.ones((3, 288, 512), dtype=np.float32) * i for i in range(2)]
        windows = create_sliding_windows(frames, window_size=3, stride=3)
        assert len(windows) == 1
        # First 3 channels should be frame[0] (padded), next 3 frame[0], next 3 frame[1]
        np.testing.assert_array_equal(windows[0][:3], frames[0])

    def test_boundary_padding_end(self):
        """4 frames -> 2 windows. Second window pads the last frame."""
        frames = [np.ones((3, 288, 512), dtype=np.float32) * i for i in range(4)]
        windows = create_sliding_windows(frames, window_size=3, stride=3)
        assert len(windows) == 2

    def test_single_frame(self):
        """Single frame produces 1 window with padding."""
        frames = [np.zeros((3, 288, 512), dtype=np.float32)]
        windows = create_sliding_windows(frames, window_size=3, stride=3)
        assert len(windows) == 1
        assert windows[0].shape == (9, 288, 512)


class TestExtractFrames:
    def test_extract_from_image_directory(self, tmp_path):
        """Extract frames from a directory of images."""
        import cv2
        for i in range(5):
            img = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"frame_{i:04d}.png"), img)
        frames, metadata = extract_frames(str(tmp_path))
        assert len(frames) == 5
        assert metadata["original_width"] == 1280
        assert metadata["original_height"] == 720
        assert frames[0].shape == (3, 288, 512)


import cv2
from inference.postprocess import heatmap_to_coordinates


class TestHeatmapToCoordinates:
    def test_single_ball(self):
        """Detect a single ball from a synthetic heatmap."""
        heatmap = np.zeros((288, 512), dtype=np.float32)
        # Place a ball-like blob at (256, 144) -- center of heatmap
        cv2.circle(heatmap, (256, 144), 15, 1.0, -1)
        result = heatmap_to_coordinates(heatmap, orig_width=1280, orig_height=720)
        assert result is not None
        x, y, confidence = result
        # Centroid should be near (256, 144), scaled to original resolution
        expected_x = 256 * (1280 / 512)
        expected_y = 144 * (720 / 288)
        assert abs(x - expected_x) < 5.0
        assert abs(y - expected_y) < 5.0
        assert confidence > 0.5

    def test_no_ball(self):
        """Return None when heatmap is below threshold."""
        heatmap = np.full((288, 512), 0.1, dtype=np.float32)
        result = heatmap_to_coordinates(heatmap, orig_width=1280, orig_height=720)
        assert result is None

    def test_multiple_blobs_picks_largest(self):
        """When multiple blobs exist, pick the largest connected component."""
        heatmap = np.zeros((288, 512), dtype=np.float32)
        # Small blob
        cv2.circle(heatmap, (100, 50), 5, 1.0, -1)
        # Large blob -- this should be chosen
        cv2.circle(heatmap, (300, 200), 20, 1.0, -1)
        result = heatmap_to_coordinates(heatmap, orig_width=1280, orig_height=720)
        assert result is not None
        x, y, _ = result
        expected_x = 300 * (1280 / 512)
        expected_y = 200 * (720 / 288)
        assert abs(x - expected_x) < 10.0
        assert abs(y - expected_y) < 10.0

    def test_custom_threshold(self):
        """Respects custom threshold parameter."""
        heatmap = np.full((288, 512), 0.4, dtype=np.float32)
        # Below default 0.5 but above 0.3
        result_default = heatmap_to_coordinates(heatmap, orig_width=1280, orig_height=720)
        result_custom = heatmap_to_coordinates(
            heatmap, orig_width=1280, orig_height=720, threshold=0.3
        )
        assert result_default is None
        assert result_custom is not None
