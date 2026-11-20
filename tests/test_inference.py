import math
import subprocess
import sys

import cv2
import numpy as np
import pytest
from inference.video_preprocess import (
    SourceInfo,
    count_frames,
    extract_frames,
    iter_frames,
    iter_windows,
    preprocess_frame,
    create_sliding_windows,
    read_frames_at,
    read_source_info,
)
from inference.postprocess import heatmap_to_coordinates, trajectory_rectification
from inference.tracker import KalmanBallTracker
from tests.conftest import CLIP_SIZE, spy_on_video_capture, write_moving_square_clip
from utils.visualization import draw_ball_on_frame


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


def test_preprocess_frame_converts_bgr_to_rgb():
    bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # pure blue in OpenCV order
    out = preprocess_frame(bgr)
    assert out.shape == (3, 288, 512)
    assert out[2].min() == 1.0 and out[0].max() == 0.0


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

    def test_background_follows_the_frames_in_every_window(self):
        frames = [np.full((3, 72, 128), i, dtype=np.float32) for i in range(5)]
        background = np.random.default_rng(0).random((3, 72, 128), dtype=np.float32)
        windows = create_sliding_windows(frames, background=background)
        plain = create_sliding_windows(frames)
        assert len(windows) == len(plain) == 2
        for window, without in zip(windows, plain, strict=True):
            assert window.shape == (12, 72, 128)
            np.testing.assert_array_equal(window[:9], without)
            np.testing.assert_array_equal(window[9:], background)

    def test_no_background_leaves_windows_unchanged(self):
        frames = [np.full((3, 72, 128), i, dtype=np.float32) for i in range(5)]
        windows = create_sliding_windows(frames, background=None)
        assert [w.shape for w in windows] == [(9, 72, 128)] * 2
        np.testing.assert_array_equal(windows[1][:3], frames[3])
        np.testing.assert_array_equal(windows[1][6:], frames[4])  # padded at the end


class TestExtractFrames:
    def test_extract_from_image_directory(self, tmp_path):
        """Extract frames from a directory of images."""
        for i in range(5):
            img = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"frame_{i:04d}.png"), img)
        frames, metadata = extract_frames(str(tmp_path))
        assert len(frames) == 5
        assert metadata["original_width"] == 1280
        assert metadata["original_height"] == 720
        assert frames[0].shape == (3, 288, 512)


class TestIterWindows:
    @pytest.mark.parametrize("n", range(7))  # 0 to 2 full windows, 0 to 2 left over
    @pytest.mark.parametrize("with_background", [False, True])
    def test_matches_create_sliding_windows(self, n, with_background):
        rng = np.random.default_rng(n)
        frames = [rng.random((3, 8, 16), dtype=np.float32) for _ in range(n)]
        background = None
        if with_background:
            background = rng.random((3, 8, 16), dtype=np.float32)
        expected = create_sliding_windows(
            frames, window_size=3, stride=3, background=background
        )
        streamed = list(iter_windows(iter(frames), background=background))
        for k, ((window, real), want) in enumerate(
            zip(streamed, expected, strict=True)
        ):
            np.testing.assert_array_equal(window, want)
            assert real == min(3, n - 3 * k)  # frames of the clip, not padding

    def test_pulls_frames_one_window_at_a_time(self):
        pulled = []

        def frames():
            for i in range(9):
                pulled.append(i)
                yield np.full((3, 2, 2), i, dtype=np.float32)

        windows = iter_windows(frames())
        next(windows)
        assert pulled == [0, 1, 2]
        next(windows)
        assert pulled == [0, 1, 2, 3, 4, 5]


@pytest.fixture(params=["video", "dir"])
def clip(request, tmp_path):
    """An 8-frame moving-square clip, as an mp4v video or a PNG directory."""
    name = "clip.mp4" if request.param == "video" else "frames"
    return write_moving_square_clip(tmp_path / name, 8)


class TestStreamingRead:
    def test_iter_frames_yields_what_extract_frames_returns(self, clip):
        frames, _ = extract_frames(str(clip))
        streamed = list(iter_frames(clip))
        assert len(streamed) == len(frames) == 8
        for got, want in zip(streamed, frames, strict=True):
            np.testing.assert_array_equal(got, want)

    def test_source_info_matches_extract_frames_metadata(self, clip):
        _, metadata = extract_frames(str(clip))
        info = read_source_info(clip)
        assert info == SourceInfo(*CLIP_SIZE, metadata.get("fps", 30.0))
        assert (info.width, info.height) == (
            metadata["original_width"],
            metadata["original_height"],
        )

    def test_count_frames(self, clip):
        assert count_frames(clip) == 8

    def test_count_frames_ignores_the_containers_frame_count(
        self, tmp_path, monkeypatch
    ):
        video = write_moving_square_clip(tmp_path / "clip.mp4", 8)
        spy_on_video_capture(monkeypatch, frame_count=810_570)
        assert count_frames(video) == 8

    def test_read_frames_at(self, clip):
        frames, _ = extract_frames(str(clip))
        chosen = read_frames_at(clip, [0, 3, 7])
        assert len(chosen) == 3
        for got, i in zip(chosen, [0, 3, 7], strict=True):
            np.testing.assert_array_equal(got, frames[i])

    def test_read_frames_at_past_the_end_raises(self, clip):
        with pytest.raises(ValueError, match="8 frames"):
            read_frames_at(clip, [0, 8])

    def test_missing_video_raises_when_called(self, tmp_path):
        with pytest.raises(ValueError, match="Failed to open video"):
            iter_frames(tmp_path / "missing.mp4")
        with pytest.raises(ValueError, match="Failed to open video"):
            read_source_info(tmp_path / "missing.mp4")

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No image files"):
            iter_frames(tmp_path)
        with pytest.raises(ValueError, match="No image files"):
            read_source_info(tmp_path)


class TestHeatmapToCoordinates:
    def test_single_ball(self):
        """Detect a single ball from a synthetic heatmap."""
        heatmap = np.zeros((288, 512), dtype=np.float32)
        # Place a ball-like blob at (256, 144) -- center of heatmap
        cv2.circle(heatmap, (256, 144), 15, 1.0, -1)
        result = heatmap_to_coordinates(heatmap, orig_width=1280, orig_height=720)
        assert result is not None
        x, y, confidence = result
        # Model pixel (256, 144) in source pixels: x = (256 + 0.5) * 1280 / 512 - 0.5
        expected_x = 640.75
        expected_y = 360.75
        assert abs(x - expected_x) < 5.0
        assert abs(y - expected_y) < 5.0
        assert confidence > 0.5

    def test_centroid_mapped_through_pixel_centres(self):
        """A blob symmetric about model pixel (256, 144) maps to (640.75, 360.75)."""
        heatmap = np.zeros((288, 512), dtype=np.float32)
        heatmap[143:146, 255:258] = 1.0
        result = heatmap_to_coordinates(heatmap, orig_width=1280, orig_height=720)
        assert result is not None
        assert result[:2] == (640.75, 360.75)

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
        expected_x = 750.75  # (300 + 0.5) * 1280 / 512 - 0.5
        expected_y = 500.75  # (200 + 0.5) * 720 / 288 - 0.5
        assert abs(x - expected_x) < 10.0
        assert abs(y - expected_y) < 10.0

    def test_custom_threshold(self):
        """Respects custom threshold parameter."""
        heatmap = np.full((288, 512), 0.4, dtype=np.float32)
        # Below default 0.5 but above 0.3
        result_default = heatmap_to_coordinates(
            heatmap, orig_width=1280, orig_height=720
        )
        result_custom = heatmap_to_coordinates(
            heatmap, orig_width=1280, orig_height=720, threshold=0.3
        )
        assert result_default is None
        assert result_custom is not None


class TestTrajectoryRectification:
    def test_no_detections_or_no_frames(self):
        assert trajectory_rectification([]) == []
        detections = [None, None, None, None]
        assert trajectory_rectification(detections) == detections

    def test_takes_a_tuple_and_returns_a_list(self):
        detections = ((0.0, 0.0), None, (10.0, 20.0))
        assert trajectory_rectification(detections) == [
            (0.0, 0.0),
            (5.0, 10.0),
            (10.0, 20.0),
        ]

    def test_never_fills_before_the_first_or_after_the_last_detection(self):
        path = [(100.0 + 30 * t, 500.0 - 40 * t + 6 * t * t) for t in range(4)]
        detections = [None, None, *path, None, None]
        assert trajectory_rectification(detections) == detections

    @pytest.mark.parametrize("max_gap", [0, 1, 4])
    def test_fills_a_gap_of_max_gap_frames_but_not_a_longer_one(self, max_gap):
        """A gap of ``max_gap`` frames is filled whole. A gap one frame longer
        is not filled at all, not even the frames next to a detection."""
        frames = max_gap + 1
        short = [(0.0, 0.0), *[None] * max_gap, (10.0 * frames, 20.0 * frames)]
        assert trajectory_rectification(short, max_gap=max_gap) == [
            pytest.approx((10.0 * t, 20.0 * t)) for t in range(frames + 1)
        ]
        too_long = [(0.0, 0.0), *[None] * (max_gap + 1), (60.0, 60.0)]
        assert trajectory_rectification(too_long, max_gap=max_gap) == too_long

    def test_fills_a_gap_only_if_the_ball_need_not_move_faster_than_max_speed(self):
        """Two missing frames between detections 30 px apart (a 3-4-5 triangle):
        10 px per frame. A gap just under a limit of 10 px per frame is filled
        whole, one just over is not filled at all; with no limit both are."""

        def gap(scale: float) -> list[tuple[float, float] | None]:
            return [(0.0, 0.0), None, None, (18.0 * scale, 24.0 * scale)]

        def line(scale: float) -> list:
            return [pytest.approx((6.0 * scale * t, 8.0 * scale * t)) for t in range(4)]

        under, over = gap(0.999), gap(1.001)
        assert trajectory_rectification(under, max_speed=10.0) == line(0.999)
        assert trajectory_rectification(over, max_speed=10.0) == over
        assert trajectory_rectification(over, max_speed=None) == line(1.001)

    def test_fills_between_detections_accurately_on_a_parabola(self):
        """A ball arc (constant vertical acceleration a = 1 px per frame squared)
        with gaps of one, two and four frames. For detections L frames apart
        (L - 1 missing between them), the straight line misses the arc k frames
        after the first by a * k * (L - k) / 2: at most 0.5, 1 and 3 px here."""

        def ball(t: int) -> tuple[float, float]:
            return (100.0 + 12.0 * t, 500.0 - 30.0 * t + 0.5 * t * t)

        n = 16
        gaps = {3, 7, 8, 11, 12, 13, 14}
        detections = [None if t in gaps else ball(t) for t in range(n)]
        result = trajectory_rectification(detections)
        for t in range(n):
            pos = result[t]
            assert pos is not None
            if t not in gaps:
                assert pos == detections[t]
                continue
            x, y = pos
            assert x == pytest.approx(ball(t)[0])
            tolerance = 1.0 if t == 3 else 4.0
            assert abs(y - ball(t)[1]) < tolerance

    def test_fills_exactly_the_short_slow_gaps_on_the_line(self):
        """Random detection patterns with positions anywhere (as clutter gives).
        For every missing frame, take the nearest detection on each side in the
        input. A filled frame has both, at most ``max_gap`` missing frames
        between them, a speed between them of at most ``max_speed`` and lies on
        the straight line between them; a frame that meets all of that is
        filled. Detections are returned unchanged."""
        rng = np.random.default_rng(0)
        filled = too_fast = 0
        for _ in range(300):
            n = int(rng.integers(0, 40))
            max_gap = int(rng.integers(0, 6))
            max_speed = [None, 20.0, 100.0, 300.0][int(rng.integers(4))]
            p_detected = rng.uniform(0.2, 0.9)
            detections: list[tuple[float, float] | None] = [
                (float(rng.uniform(0, 1000)), float(rng.uniform(0, 1000)))
                if rng.random() < p_detected
                else None
                for _ in range(n)
            ]
            result = trajectory_rectification(
                detections, max_gap=max_gap, max_speed=max_speed
            )
            assert len(result) == n
            known = {j: d for j, d in enumerate(detections) if d is not None}
            for i, pos in enumerate(result):
                if i in known:
                    assert pos == known[i]
                    continue
                before = max((j for j in known if j < i), default=None)
                after = min((j for j in known if j > i), default=None)
                if before is None or after is None:
                    assert pos is None, f"frame {i} extrapolated"
                    continue
                x0, y0 = known[before]
                x1, y1 = known[after]
                frames = after - before
                speed = math.hypot(x1 - x0, y1 - y0) / frames
                short = frames - 1 <= max_gap
                slow = max_speed is None or speed <= max_speed
                too_fast += short and not slow
                if pos is None:
                    assert not (short and slow), f"frame {i} left unfilled"
                    continue
                filled += 1
                assert short, f"frame {i} filled in a gap of {frames - 1}"
                assert slow, f"frame {i} filled at {speed:.1f} px per frame"
                t = (i - before) / frames
                assert pos == pytest.approx((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        # The patterns do exercise filling and the speed limit.
        assert filled > 100
        assert too_fast > 100


class TestDrawBallOnFrame:
    def test_draws_circle_on_frame(self):
        """Ball circle is drawn; frame is modified."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = draw_ball_on_frame(frame, x=640, y=360, confidence=0.9)
        # The drawn region should not be all zeros
        assert result[360, 640].sum() > 0

    def test_high_confidence_is_green(self):
        """High confidence (>0.8) draws green circle."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = draw_ball_on_frame(frame, x=640, y=360, confidence=0.9)
        # Green channel should be dominant at center (BGR format)
        b, g, r = result[360, 640]
        assert g > b and g > r

    def test_medium_confidence_is_yellow(self):
        """Medium confidence (0.5-0.8) draws yellow circle."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = draw_ball_on_frame(frame, x=640, y=360, confidence=0.65)
        b, g, r = result[360, 640]
        # Yellow in BGR = (0, 255, 255)
        assert g > 0 and r > 0

    def test_low_confidence_is_red(self):
        """Low confidence (<0.5) draws red circle."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = draw_ball_on_frame(frame, x=640, y=360, confidence=0.3)
        b, g, r = result[360, 640]
        # Red in BGR = (0, 0, 255)
        assert r > g and r > b

    def test_does_not_modify_original(self):
        """Original frame is not modified; a copy is returned."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        original_sum = frame.sum()
        draw_ball_on_frame(frame, x=640, y=360, confidence=0.9)
        assert frame.sum() == original_sum


class TestKalmanBallTracker:
    def test_smooth_linear_trajectory(self):
        """Kalman filter smooths a noisy linear trajectory."""
        tracker = KalmanBallTracker()
        # True trajectory: x goes 100->200, y stays 150
        noisy_measurements = []
        rng = np.random.default_rng(42)
        for i in range(20):
            true_x = 100 + i * 5
            true_y = 150.0
            noisy_x = true_x + rng.normal(0, 3)
            noisy_y = true_y + rng.normal(0, 3)
            noisy_measurements.append((noisy_x, noisy_y))

        smoothed = []
        for mx, my in noisy_measurements:
            sx, sy = tracker.update(mx, my)
            smoothed.append((sx, sy))

        # Smoothed trajectory should be closer to the true line than raw measurements
        true_y_val = 150.0
        raw_y_errors = [abs(m[1] - true_y_val) for m in noisy_measurements]
        smooth_y_errors = [abs(s[1] - true_y_val) for s in smoothed[5:]]  # skip warmup
        assert np.mean(smooth_y_errors) < np.mean(raw_y_errors)

    def test_predict_without_update(self):
        """Tracker can predict next position without a measurement."""
        tracker = KalmanBallTracker()
        tracker.update(100.0, 150.0)
        tracker.update(110.0, 150.0)
        x, y = tracker.predict()
        # Should predict forward based on velocity
        assert x > 110.0

    def test_reset(self):
        """Reset clears tracker state."""
        tracker = KalmanBallTracker()
        tracker.update(100.0, 150.0)
        tracker.reset()
        # After reset, predict should return initial position (zeroed)
        x, y = tracker.predict()
        assert x == 0.0 and y == 0.0


class TestCLI:
    def test_help_shows_subcommands(self):
        """CLI --help should list train, evaluate, infer subcommands."""
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/kevinlee/workspace/playground/playground-track-net",
        )
        assert result.returncode == 0
        assert "train" in result.stdout
        assert "infer" in result.stdout

    def test_infer_requires_video(self):
        """Infer subcommand requires --video argument."""
        result = subprocess.run(
            [sys.executable, "main.py", "infer"],
            capture_output=True,
            text=True,
            cwd="/home/kevinlee/workspace/playground/playground-track-net",
        )
        assert result.returncode != 0
        assert "video" in result.stderr.lower() or "required" in result.stderr.lower()
