# TrackNet V2 Inference Pipeline

**Feature:** Inference pipeline -- video preprocessing, post-processing, Kalman tracking, visualization, and CLI entry point
**Goal:** End-to-end inference: video file in -> per-frame ball coordinates CSV out, with optional annotated video output
**Architecture:** Modular pipeline -- frame extraction -> model forward pass -> heatmap post-processing -> trajectory rectification -> Kalman smoothing -> CSV/visualization output
**Tech Stack:** Python 3.12, PyTorch, OpenCV, scipy, numpy, argparse (no filterpy — custom Kalman filter using NumPy)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` -- each task below is a self-contained TDD unit. Write failing test first, implement, verify, commit.

---

## File Map

| File | Purpose |
|------|---------|
| `inference/__init__.py` | Package init, re-exports key classes/functions |
| `inference/video_preprocess.py` | **UPDATE** -- Frame extraction, resize to 512x288, normalize to [0,1], sliding window of 3 frames with stride=3, boundary padding |
| `inference/postprocess.py` | `heatmap_to_coordinates()` (threshold -> connected components -> centroid -> scale to original resolution), `trajectory_rectification()` (spline interpolation over gaps) |
| `inference/tracker.py` | `KalmanBallTracker` using custom NumPy implementation -- state: [x, y, vx, vy], smooths noisy detections. No filterpy dependency (unmaintained since 2018, incompatible with Python 3.14). |
| `utils/__init__.py` | Package init |
| `utils/visualization.py` | `draw_ball_on_frame()` -- overlay detected ball position on video frames, color-coded by confidence |
| `main.py` | **UPDATE** -- CLI entry point with argparse subcommands: `train`, `evaluate`, `infer` |
| `tests/test_inference.py` | All inference pipeline tests (synthetic data only, no GPU required) |

---

## Task 1: Inference package init + video preprocessing rewrite

**Files:** `inference/__init__.py`, `inference/video_preprocess.py`, `tests/test_inference.py`

### Steps

- [ ] **1a. Create `inference/__init__.py`**

  ```python
  from inference.video_preprocess import extract_frames, create_sliding_windows
  from inference.postprocess import heatmap_to_coordinates, trajectory_rectification
  from inference.tracker import KalmanBallTracker
  ```

  This will initially fail on imports since the other modules don't exist yet. That is fine -- we build them incrementally. Create a minimal `__init__.py` for now with just the preprocess import, and expand it later in Task 7.

  Minimal version for now:

  ```python
  from inference.video_preprocess import extract_frames, create_sliding_windows, preprocess_frame
  ```

- [ ] **1b. Write failing tests for video preprocessing** in `tests/test_inference.py`

  ```python
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
  ```

  Run: `uv run pytest tests/test_inference.py -x -v` (expect failures)

- [ ] **1c. Implement `inference/video_preprocess.py`**

  Rewrite the existing file completely:

  ```python
  from pathlib import Path

  import cv2
  import numpy as np

  TARGET_WIDTH = 512
  TARGET_HEIGHT = 288


  def preprocess_frame(frame: np.ndarray) -> np.ndarray:
      """Resize frame to 512x288, normalize to [0,1], return as (C, H, W) float32."""
      resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
      normalized = resized.astype(np.float32) / 255.0
      return np.transpose(normalized, (2, 0, 1))  # HWC -> CHW


  def extract_frames(source: str) -> tuple[list[np.ndarray], dict]:
      """Extract and preprocess frames from a video file or image directory.

      Args:
          source: Path to a video file (mp4/avi) or directory of images.

      Returns:
          Tuple of (list of preprocessed frames as CHW float32 arrays, metadata dict).
          Metadata contains original_width, original_height, frame_count.
      """
      source_path = Path(source)
      if source_path.is_dir():
          return _extract_from_directory(source_path)
      else:
          return _extract_from_video(source_path)


  def _extract_from_directory(directory: Path) -> tuple[list[np.ndarray], dict]:
      extensions = {".png", ".jpg", ".jpeg", ".bmp"}
      image_files = sorted(
          f for f in directory.iterdir() if f.suffix.lower() in extensions
      )
      if not image_files:
          raise ValueError(f"No image files found in {directory}")

      frames = []
      metadata = {}
      for i, img_path in enumerate(image_files):
          frame = cv2.imread(str(img_path))
          if frame is None:
              raise ValueError(f"Failed to read image: {img_path}")
          if i == 0:
              h, w = frame.shape[:2]
              metadata = {
                  "original_width": w,
                  "original_height": h,
                  "frame_count": len(image_files),
              }
          frames.append(preprocess_frame(frame))
      return frames, metadata


  def _extract_from_video(video_path: Path) -> tuple[list[np.ndarray], dict]:
      cap = cv2.VideoCapture(str(video_path))
      if not cap.isOpened():
          raise ValueError(f"Failed to open video: {video_path}")

      w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      metadata = {
          "original_width": w,
          "original_height": h,
          "frame_count": frame_count,
      }

      frames = []
      while True:
          ret, frame = cap.read()
          if not ret:
              break
          frames.append(preprocess_frame(frame))
      cap.release()
      return frames, metadata


  def create_sliding_windows(
      frames: list[np.ndarray], window_size: int = 3, stride: int = 3
  ) -> list[np.ndarray]:
      """Create sliding windows of concatenated frames with boundary padding.

      Stride=3 (MIMO): windows are [0,1,2], [3,4,5], etc.
      At boundaries, pad by duplicating the first/last frame.

      Args:
          frames: List of preprocessed frames, each shape (C, H, W).
          window_size: Number of frames per window.
          stride: Step between window start indices.

      Returns:
          List of concatenated frame windows, each shape (C*window_size, H, W).
      """
      n = len(frames)
      if n == 0:
          return []

      windows = []
      for start in range(0, n, stride):
          window_frames = []
          for i in range(window_size):
              idx = start + i
              if idx < 0:
                  idx = 0
              elif idx >= n:
                  idx = n - 1
              window_frames.append(frames[idx])
          concatenated = np.concatenate(window_frames, axis=0)
          windows.append(concatenated)
      return windows
  ```

  Run: `uv run pytest tests/test_inference.py::TestPreprocessFrame tests/test_inference.py::TestCreateSlidingWindows tests/test_inference.py::TestExtractFrames -x -v`

- [ ] **1d. Commit:** `git add inference/__init__.py inference/video_preprocess.py tests/test_inference.py && git commit -m "feat(inference): rewrite video preprocessing with sliding window and frame extraction"`

---

## Task 2: Post-processing -- heatmap to coordinates

**Files:** `inference/postprocess.py`, `tests/test_inference.py`

### Steps

- [ ] **2a. Write failing tests for `heatmap_to_coordinates`**

  Append to `tests/test_inference.py`:

  ```python
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
  ```

  Run: `uv run pytest tests/test_inference.py::TestHeatmapToCoordinates -x -v` (expect failures)

- [ ] **2b. Implement `heatmap_to_coordinates` in `inference/postprocess.py`**

  ```python
  import numpy as np
  from scipy import ndimage


  def heatmap_to_coordinates(
      heatmap: np.ndarray,
      orig_width: int,
      orig_height: int,
      threshold: float = 0.5,
  ) -> tuple[float, float, float] | None:
      """Convert a sigmoid heatmap to ball coordinates in original resolution.

      Args:
          heatmap: 2D array shape (H, W) with values in [0, 1].
          orig_width: Original video frame width.
          orig_height: Original video frame height.
          threshold: Detection threshold.

      Returns:
          (x, y, confidence) in original resolution, or None if no ball detected.
      """
      h, w = heatmap.shape
      binary = (heatmap > threshold).astype(np.int32)

      labeled, num_features = ndimage.label(binary)
      if num_features == 0:
          return None

      # Find largest connected component
      component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
      largest_label = np.argmax(component_sizes) + 1

      # Compute centroid of largest component
      cy, cx = ndimage.center_of_mass(binary, labeled, largest_label)

      # Confidence: mean heatmap value within the component
      component_mask = labeled == largest_label
      confidence = float(np.mean(heatmap[component_mask]))

      # Scale to original resolution
      x_orig = cx * (orig_width / w)
      y_orig = cy * (orig_height / h)

      return (x_orig, y_orig, confidence)
  ```

  Run: `uv run pytest tests/test_inference.py::TestHeatmapToCoordinates -x -v`

- [ ] **2c. Commit:** `git add inference/postprocess.py tests/test_inference.py && git commit -m "feat(inference): add heatmap_to_coordinates post-processing"`

---

## Task 3: Post-processing -- trajectory rectification

**Files:** `inference/postprocess.py`, `tests/test_inference.py`

### Steps

- [ ] **3a. Write failing tests for `trajectory_rectification`**

  Append to `tests/test_inference.py`:

  ```python
  from inference.postprocess import trajectory_rectification


  class TestTrajectoryRectification:
      def test_fills_single_gap(self):
          """Interpolate a single missing frame in a linear trajectory."""
          detections = [
              (100.0, 100.0),  # frame 0
              (110.0, 105.0),  # frame 1
              None,             # frame 2 -- gap
              (130.0, 115.0),  # frame 3
              (140.0, 120.0),  # frame 4
          ]
          result = trajectory_rectification(detections, window=8)
          assert result[2] is not None
          x, y = result[2]
          # Should be approximately (120, 110) for a linear trajectory
          assert abs(x - 120.0) < 5.0
          assert abs(y - 110.0) < 5.0

      def test_does_not_fill_without_enough_context(self):
          """Don't interpolate if fewer than 3 known positions in window."""
          detections = [
              None,
              (100.0, 100.0),
              None,
              None,
              (120.0, 110.0),
              None,
              None,
              None,
          ]
          result = trajectory_rectification(detections, window=4)
          # Some gaps may not be filled if insufficient context
          # The key requirement: need >= 3 known positions in the window
          filled_count = sum(1 for d in result if d is not None)
          known_count = sum(1 for d in detections if d is not None)
          # Should fill some but not fabricate without evidence
          assert filled_count >= known_count

      def test_preserves_existing_detections(self):
          """Known detections must not be modified."""
          detections = [
              (100.0, 100.0),
              (110.0, 105.0),
              (120.0, 110.0),
          ]
          result = trajectory_rectification(detections, window=8)
          for i in range(3):
              assert result[i] == detections[i]

      def test_all_none_returns_all_none(self):
          """No detections at all -- nothing to interpolate."""
          detections = [None, None, None, None]
          result = trajectory_rectification(detections, window=8)
          assert all(d is None for d in result)

      def test_fills_multi_frame_gap(self):
          """Fill a 2-frame gap in a parabolic trajectory."""
          detections = [
              (100.0, 200.0),
              (110.0, 190.0),
              (120.0, 182.0),
              None,             # gap
              None,             # gap
              (150.0, 172.0),
              (160.0, 170.0),
              (170.0, 170.0),
          ]
          result = trajectory_rectification(detections, window=8)
          assert result[3] is not None
          assert result[4] is not None
  ```

  Run: `uv run pytest tests/test_inference.py::TestTrajectoryRectification -x -v` (expect failures)

- [ ] **3b. Implement `trajectory_rectification` in `inference/postprocess.py`**

  Add to `inference/postprocess.py`:

  ```python
  from scipy.interpolate import UnivariateSpline


  def trajectory_rectification(
      detections: list[tuple[float, float] | None],
      window: int = 8,
  ) -> list[tuple[float, float] | None]:
      """Interpolate missing ball positions using spline fitting on surrounding detections.

      Args:
          detections: List of (x, y) tuples or None for each frame.
          window: Number of surrounding frames to consider for fitting.

      Returns:
          New list with gaps filled where possible. Existing detections are preserved.
      """
      n = len(detections)
      result = list(detections)

      for i in range(n):
          if result[i] is not None:
              continue

          # Gather known positions within the window
          half_w = window // 2
          start = max(0, i - half_w)
          end = min(n, i + half_w + 1)

          known_indices = []
          known_x = []
          known_y = []
          for j in range(start, end):
              if detections[j] is not None:
                  known_indices.append(j)
                  known_x.append(detections[j][0])
                  known_y.append(detections[j][1])

          # Need at least 3 known positions for spline fitting
          if len(known_indices) < 3:
              continue

          k = min(3, len(known_indices) - 1)  # spline degree
          try:
              spline_x = UnivariateSpline(known_indices, known_x, k=k, s=0)
              spline_y = UnivariateSpline(known_indices, known_y, k=k, s=0)
              interp_x = float(spline_x(i))
              interp_y = float(spline_y(i))
              result[i] = (interp_x, interp_y)
          except Exception:
              # Spline fitting can fail with degenerate inputs -- skip
              continue

      return result
  ```

  Run: `uv run pytest tests/test_inference.py::TestTrajectoryRectification -x -v`

- [ ] **3c. Commit:** `git add inference/postprocess.py tests/test_inference.py && git commit -m "feat(inference): add trajectory rectification with spline interpolation"`

---

## Task 4: Kalman ball tracker

**Files:** `inference/tracker.py`, `tests/test_inference.py`

### Steps

- [ ] **4a. Write failing tests for `KalmanBallTracker`**

  Append to `tests/test_inference.py`:

  ```python
  from inference.tracker import KalmanBallTracker


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
  ```

  Run: `uv run pytest tests/test_inference.py::TestKalmanBallTracker -x -v` (expect failures)

- [ ] **4b. Implement `inference/tracker.py`**

  ```python
  import numpy as np


  class KalmanBallTracker:
      """Kalman filter for smoothing ball position detections.

      Pure NumPy implementation (no filterpy dependency).
      State: [x, y, vx, vy] -- position and velocity.
      Measurement: [x, y] -- detected ball position.
      Constant-velocity model with higher process noise for fast ball dynamics.
      """

      def __init__(self, process_noise: float = 50.0, measurement_noise: float = 5.0):
          # State vector: [x, y, vx, vy]
          self.x = np.zeros((4, 1), dtype=np.float64)

          # State transition: constant velocity model
          # x' = x + vx*dt, y' = y + vy*dt (dt=1 frame)
          self.F = np.array([
              [1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1],
          ], dtype=np.float64)

          # Measurement function: observe x, y only
          self.H = np.array([
              [1, 0, 0, 0],
              [0, 1, 0, 0],
          ], dtype=np.float64)

          # Process noise -- higher than typical for fast ball dynamics
          self.Q = np.eye(4, dtype=np.float64) * process_noise
          self.Q[2, 2] *= 2.0  # Extra noise on velocity components
          self.Q[3, 3] *= 2.0

          # Measurement noise
          self.R = np.eye(2, dtype=np.float64) * measurement_noise

          # Covariance matrix -- high initial uncertainty
          self.P = np.eye(4, dtype=np.float64) * 1000.0

          self._initialized = False

      def _predict(self):
          """Predict step: project state and covariance forward."""
          self.x = self.F @ self.x
          self.P = self.F @ self.P @ self.F.T + self.Q

      def _correct(self, z: np.ndarray):
          """Update step: incorporate measurement."""
          S = self.H @ self.P @ self.H.T + self.R
          K = self.P @ self.H.T @ np.linalg.inv(S)
          y = z.reshape(2, 1) - self.H @ self.x
          self.x = self.x + K @ y
          I = np.eye(4, dtype=np.float64)
          self.P = (I - K @ self.H) @ self.P

      def update(self, x: float, y: float) -> tuple[float, float]:
          """Update tracker with a new measurement and return smoothed position."""
          measurement = np.array([x, y], dtype=np.float64)

          if not self._initialized:
              self.x[:2] = measurement.reshape(2, 1)
              self._initialized = True
          else:
              self._predict()
              self._correct(measurement)

          return (float(self.x[0, 0]), float(self.x[1, 0]))

      def predict(self) -> tuple[float, float]:
          """Predict next position without a measurement."""
          self._predict()
          return (float(self.x[0, 0]), float(self.x[1, 0]))

      def reset(self):
          """Reset tracker to initial state."""
          self.x = np.zeros((4, 1), dtype=np.float64)
          self.P = np.eye(4, dtype=np.float64) * 1000.0
          self._initialized = False
  ```

  Run: `uv run pytest tests/test_inference.py::TestKalmanBallTracker -x -v`

- [ ] **4c. Commit:** `git add inference/tracker.py tests/test_inference.py && git commit -m "feat(inference): add Kalman ball tracker using custom NumPy implementation"`

---

## Task 5: Visualization utility

**Files:** `utils/__init__.py`, `utils/visualization.py`, `tests/test_inference.py`

### Steps

- [ ] **5a. Write failing tests for `draw_ball_on_frame`**

  Append to `tests/test_inference.py`:

  ```python
  from utils.visualization import draw_ball_on_frame


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
  ```

  Run: `uv run pytest tests/test_inference.py::TestDrawBallOnFrame -x -v` (expect failures)

- [ ] **5b. Create `utils/__init__.py`**

  ```python
  from utils.visualization import draw_ball_on_frame
  ```

- [ ] **5c. Implement `utils/visualization.py`**

  ```python
  import cv2
  import numpy as np


  def draw_ball_on_frame(
      frame: np.ndarray,
      x: float,
      y: float,
      confidence: float,
      radius: int = 10,
  ) -> np.ndarray:
      """Draw a circle at the detected ball position, color-coded by confidence.

      Args:
          frame: BGR image (H, W, 3) uint8.
          x: Ball x coordinate in frame resolution.
          y: Ball y coordinate in frame resolution.
          confidence: Detection confidence in [0, 1].
          radius: Circle radius in pixels.

      Returns:
          Copy of frame with circle drawn.
      """
      result = frame.copy()
      center = (int(round(x)), int(round(y)))

      # Color by confidence (BGR format)
      if confidence >= 0.8:
          color = (0, 255, 0)      # Green -- high confidence
      elif confidence >= 0.5:
          color = (0, 255, 255)    # Yellow -- medium confidence
      else:
          color = (0, 0, 255)      # Red -- low confidence

      cv2.circle(result, center, radius, color, thickness=-1)
      return result
  ```

  Run: `uv run pytest tests/test_inference.py::TestDrawBallOnFrame -x -v`

- [ ] **5d. Commit:** `git add utils/__init__.py utils/visualization.py tests/test_inference.py && git commit -m "feat(utils): add ball visualization with confidence-based coloring"`

---

## Task 6: CLI entry point (main.py)

**Files:** `main.py`, `tests/test_inference.py`

### Steps

- [ ] **6a. Write failing tests for CLI**

  Append to `tests/test_inference.py`:

  ```python
  import subprocess
  import sys


  class TestCLI:
      def test_help_shows_subcommands(self):
          """CLI --help should list train, evaluate, infer subcommands."""
          result = subprocess.run(
              [sys.executable, "main.py", "--help"],
              capture_output=True, text=True,
              cwd="/home/kevinlee/workspace/playground/playground-track-net",
          )
          assert result.returncode == 0
          assert "train" in result.stdout
          assert "infer" in result.stdout

      def test_infer_requires_video(self):
          """Infer subcommand requires --video argument."""
          result = subprocess.run(
              [sys.executable, "main.py", "infer"],
              capture_output=True, text=True,
              cwd="/home/kevinlee/workspace/playground/playground-track-net",
          )
          assert result.returncode != 0
          assert "video" in result.stderr.lower() or "required" in result.stderr.lower()
  ```

  Run: `uv run pytest tests/test_inference.py::TestCLI -x -v` (expect failures)

- [ ] **6b. Rewrite `main.py`**

  ```python
  import argparse
  import csv
  import sys
  from pathlib import Path

  import numpy as np
  import torch

  from inference.video_preprocess import extract_frames, create_sliding_windows
  from inference.postprocess import heatmap_to_coordinates, trajectory_rectification
  from inference.tracker import KalmanBallTracker


  def build_parser() -> argparse.ArgumentParser:
      parser = argparse.ArgumentParser(
          prog="tracknet",
          description="TrackNet V2 ball tracking system",
      )
      subparsers = parser.add_subparsers(dest="command", required=True)

      # Train subcommand (placeholder)
      train_parser = subparsers.add_parser("train", help="Train the model")
      train_parser.add_argument("--config", type=str, default="configs/default.yaml")

      # Evaluate subcommand (placeholder)
      evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
      evaluate_parser.add_argument("--config", type=str, default="configs/default.yaml")
      evaluate_parser.add_argument("--weights", type=str, required=True)

      # Infer subcommand
      infer_parser = subparsers.add_parser("infer", help="Run inference on a video")
      infer_parser.add_argument(
          "--video", type=str, required=True,
          help="Path to input video file or image directory",
      )
      infer_parser.add_argument(
          "--model", type=str, required=True,
          help="Path to model weights (.pt file)",
      )
      infer_parser.add_argument(
          "--output", type=str, required=True,
          help="Path to output CSV file",
      )
      infer_parser.add_argument(
          "--output-video", type=str, default=None,
          help="Path to output annotated video (optional)",
      )
      infer_parser.add_argument(
          "--threshold", type=float, default=0.5,
          help="Heatmap detection threshold (default: 0.5)",
      )

      return parser


  def run_inference(args: argparse.Namespace) -> None:
      """Run the full inference pipeline."""
      # 1. Extract and preprocess frames
      print(f"Extracting frames from {args.video}...")
      frames, metadata = extract_frames(args.video)
      orig_w = metadata["original_width"]
      orig_h = metadata["original_height"]
      print(f"Extracted {len(frames)} frames ({orig_w}x{orig_h})")

      # 2. Create sliding windows
      windows = create_sliding_windows(frames, window_size=3, stride=3)
      print(f"Created {len(windows)} sliding windows")

      # 3. Load model
      print(f"Loading model from {args.model}...")
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      # Model loading assumes models/tracknet.py exists and provides TrackNet class
      # This will be implemented by the model subsystem
      from models.tracknet import TrackNet

      model = TrackNet()
      model.load_state_dict(
          torch.load(args.model, map_location=device, weights_only=True)
      )
      model.to(device)

      # 4. Run inference on each window
      all_detections: list[tuple[float, float, float] | None] = []
      model.eval()
      with torch.no_grad():
          for window in windows:
              input_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
              heatmaps = model(input_tensor)  # (1, 3, H, W)
              heatmaps = heatmaps.squeeze(0).cpu().numpy()  # (3, H, W)

              for hm in heatmaps:
                  result = heatmap_to_coordinates(
                      hm, orig_width=orig_w, orig_height=orig_h,
                      threshold=args.threshold,
                  )
                  if result is not None:
                      all_detections.append(result)
                  else:
                      all_detections.append(None)

      # Trim to actual frame count (sliding windows may overshoot)
      all_detections = all_detections[:len(frames)]

      # 5. Trajectory rectification
      positions = [
          (d[0], d[1]) if d is not None else None
          for d in all_detections
      ]
      rectified = trajectory_rectification(positions, window=8)

      # 6. Kalman smoothing
      tracker = KalmanBallTracker()
      final_results = []
      for i, det in enumerate(all_detections):
          confidence = det[2] if det is not None else 0.0
          pos = rectified[i]
          if pos is not None:
              sx, sy = tracker.update(pos[0], pos[1])
              visibility = 1
          else:
              sx, sy = 0.0, 0.0
              visibility = 0
          final_results.append({
              "frame": i,
              "x": sx,
              "y": sy,
              "confidence": confidence,
              "visibility": visibility,
          })

      # 7. Write CSV output
      output_path = Path(args.output)
      output_path.parent.mkdir(parents=True, exist_ok=True)
      with open(output_path, "w", newline="") as f:
          writer = csv.DictWriter(
              f, fieldnames=["Frame", "X", "Y", "Confidence", "Visibility"]
          )
          writer.writeheader()
          for r in final_results:
              writer.writerow({
                  "Frame": r["frame"],
                  "X": f"{r['x']:.2f}",
                  "Y": f"{r['y']:.2f}",
                  "Confidence": f"{r['confidence']:.4f}",
                  "Visibility": r["visibility"],
              })
      print(f"Results written to {args.output}")

      # 8. Optional annotated video output
      if args.output_video:
          import cv2
          from utils.visualization import draw_ball_on_frame

          cap = cv2.VideoCapture(args.video)
          fourcc = cv2.VideoWriter_fourcc(*"mp4v")
          out = cv2.VideoWriter(args.output_video, fourcc, 30.0, (orig_w, orig_h))

          for r in final_results:
              ret, frame = cap.read()
              if not ret:
                  break
              if r["visibility"] == 1:
                  frame = draw_ball_on_frame(
                      frame, r["x"], r["y"], r["confidence"],
                  )
              out.write(frame)

          cap.release()
          out.release()
          print(f"Annotated video written to {args.output_video}")


  def main():
      parser = build_parser()
      args = parser.parse_args()

      if args.command == "infer":
          run_inference(args)
      elif args.command == "train":
          print("Training not yet implemented. See training/ subsystem.")
          sys.exit(1)
      elif args.command == "evaluate":
          print("Evaluation not yet implemented. See training/ subsystem.")
          sys.exit(1)


  if __name__ == "__main__":
      main()
  ```

  Run: `uv run pytest tests/test_inference.py::TestCLI -x -v`

- [ ] **6c. Commit:** `git add main.py tests/test_inference.py && git commit -m "feat(cli): add argparse CLI with train/evaluate/infer subcommands"`

---

## Task 7: Finalize inference package and update dependencies

**Files:** `inference/__init__.py`, `pyproject.toml`

### Steps

- [ ] **7a. Update `inference/__init__.py`** with all imports:

  ```python
  from inference.video_preprocess import extract_frames, create_sliding_windows, preprocess_frame
  from inference.postprocess import heatmap_to_coordinates, trajectory_rectification
  from inference.tracker import KalmanBallTracker

  __all__ = [
      "extract_frames",
      "create_sliding_windows",
      "preprocess_frame",
      "heatmap_to_coordinates",
      "trajectory_rectification",
      "KalmanBallTracker",
  ]
  ```

- [ ] **7b. Update `pyproject.toml` dependencies**

  Add `"scipy>=1.15.0"` to the `dependencies` list (needed for ndimage.label and UnivariateSpline). Remove `"filterpy>=1.4.5"` (unmaintained since 2018, incompatible with Python 3.14 — replaced by custom NumPy Kalman filter).

- [ ] **7c. Run full test suite**

  ```bash
  uv run pytest tests/test_inference.py -v
  ```

  All tests should pass.

- [ ] **7d. Commit:** `git add inference/__init__.py pyproject.toml uv.lock && git commit -m "chore: finalize inference package, add scipy, remove filterpy"`

---

## Task 8: Lint, format, and final verification

### Steps

- [ ] **8a. Run ruff and black**

  ```bash
  uv run ruff check inference/ utils/ main.py tests/test_inference.py --fix
  uv run black inference/ utils/ main.py tests/test_inference.py
  ```

- [ ] **8b. Run full test suite one final time**

  ```bash
  uv run pytest tests/test_inference.py -v --tb=short
  ```

- [ ] **8c. Fix any issues and commit**

  ```bash
  git add -u && git commit -m "style: format inference pipeline with ruff and black"
  ```

---

## Interface Notes

These are boundary points where the inference pipeline connects with other subsystems:

- **Model subsystem:** `run_inference()` in `main.py` imports `from models.tracknet import TrackNet`. This class must accept a 9-channel input tensor `(batch, 9, 288, 512)` and return `(batch, 3, 288, 512)` sigmoid heatmaps. The model subsystem must provide this interface.
- **Config subsystem:** Currently hardcoded values (threshold=0.5, window=8, etc.) will move to `configs/default.yaml` when the config subsystem is built.
- **Training subsystem:** The `train` and `evaluate` subcommands in `main.py` are placeholders that exit with an error. The training subsystem will implement these.
