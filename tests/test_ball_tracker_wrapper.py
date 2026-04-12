"""Tests for BallTrackerWrapper."""

import numpy as np
import pytest
import torch

from models.ball_tracker_wrapper import BallTrackerWrapper


def test_ball_tracker_stub_mode():
    """Test tracker works in stub mode (no model)."""
    tracker = BallTrackerWrapper(model_path=None)

    # Create 3 fake frames (720x1280x3 uint8 BGR)
    fake_frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]

    ball_pos = tracker.track_ball(fake_frames, orig_width=1280, orig_height=720)

    assert ball_pos is not None
    assert len(ball_pos) == 3  # (x, y, confidence)
    assert 0 <= ball_pos[0] <= 1280  # x within bounds
    assert 0 <= ball_pos[1] <= 720  # y within bounds
    assert 0 <= ball_pos[2] <= 1.0  # confidence in [0, 1]


def test_ball_tracker_requires_three_frames():
    """Test tracker raises error if not given 3 frames."""
    tracker = BallTrackerWrapper(model_path=None)

    # Try with 2 frames
    two_frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(2)]

    with pytest.raises(ValueError, match="Expected 3 frames, got 2"):
        tracker.track_ball(two_frames, orig_width=1280, orig_height=720)

    # Try with 4 frames
    four_frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(4)]

    with pytest.raises(ValueError, match="Expected 3 frames, got 4"):
        tracker.track_ball(four_frames, orig_width=1280, orig_height=720)


def test_ball_tracker_stub_returns_center():
    """Test stub mode returns center of frame."""
    tracker = BallTrackerWrapper(model_path=None)

    fake_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]

    ball_pos = tracker.track_ball(fake_frames, orig_width=640, orig_height=480)

    # Stub mode returns (orig_width/2, orig_height/2, 0.95)
    assert ball_pos[0] == 320.0  # x = 640/2
    assert ball_pos[1] == 240.0  # y = 480/2
    assert ball_pos[2] == 0.95  # confidence


def test_ball_tracker_preprocess_shape():
    """Test preprocessing creates correct tensor shape."""
    tracker = BallTrackerWrapper(model_path=None)

    # Create 3 fake frames
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]

    # Access private method for testing
    tensor = tracker._preprocess_frames(frames)

    # Should be (1, 9, 288, 512) for TrackNet V2
    assert tensor.shape == (1, 9, 288, 512)
    assert tensor.dtype == torch.float32


def test_ball_tracker_preprocess_normalization():
    """Test preprocessing normalizes values to [0, 1]."""
    tracker = BallTrackerWrapper(model_path=None)

    # Create frames with known values
    frames = [np.full((100, 100, 3), 255, dtype=np.uint8) for _ in range(3)]

    tensor = tracker._preprocess_frames(frames)

    # All values should be 1.0 (255/255)
    assert torch.all(tensor <= 1.0)
    assert torch.all(tensor >= 0.0)
    # Most values should be close to 1.0 (since input was all 255)
    assert torch.mean(tensor) > 0.9


def test_ball_tracker_preprocess_channel_order():
    """Test preprocessing stacks 3 frames along channel dimension."""
    tracker = BallTrackerWrapper(model_path=None)

    # Create 3 frames with different colors
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame1[:, :, 0] = 255  # Blue channel

    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2[:, :, 1] = 255  # Green channel

    frame3 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame3[:, :, 2] = 255  # Red channel

    frames = [frame1, frame2, frame3]
    tensor = tracker._preprocess_frames(frames)

    # Shape: (1, 9, H, W)
    # Channels 0-2: frame1 (BGR)
    # Channels 3-5: frame2 (BGR)
    # Channels 6-8: frame3 (BGR)

    assert tensor.shape == (1, 9, 288, 512)

    # Frame1 blue channel (BGR order, so index 0)
    assert torch.mean(tensor[0, 0]) > 0.9

    # Frame2 green channel (BGR order, so index 4)
    assert torch.mean(tensor[0, 4]) > 0.9

    # Frame3 red channel (BGR order, so index 8)
    assert torch.mean(tensor[0, 8]) > 0.9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ball_tracker_uses_cuda_when_available():
    """Test tracker uses CUDA when available."""
    tracker = BallTrackerWrapper(model_path=None, device="cuda")

    # Device should be set even in stub mode
    assert tracker.device == torch.device("cuda")


def test_ball_tracker_uses_cpu_when_specified(tmp_path):
    """Test tracker uses CPU when specified with a model."""
    from models.tracknet import TrackNet

    model = TrackNet()
    checkpoint_path = tmp_path / "mock.pt"
    torch.save(model.state_dict(), checkpoint_path)

    tracker = BallTrackerWrapper(model_path=str(checkpoint_path), device="cpu")

    assert tracker.device == torch.device("cpu")


def test_ball_tracker_stub_mode_has_no_device():
    """Test stub mode has no device set."""
    tracker = BallTrackerWrapper(model_path=None)

    assert tracker.device is None
    assert tracker.model is None


def test_ball_tracker_custom_threshold():
    """Test custom threshold is stored."""
    tracker = BallTrackerWrapper(model_path=None, threshold=0.7)

    assert tracker.threshold == 0.7


def test_ball_tracker_default_threshold():
    """Test default threshold is 0.5."""
    tracker = BallTrackerWrapper(model_path=None)

    assert tracker.threshold == 0.5


# Integration test with mock model
def test_ball_tracker_with_mock_model(tmp_path):
    """Test tracker works with a mock TrackNet model."""
    # Create a simple mock model
    from models.tracknet import TrackNet

    model = TrackNet()

    # Save mock checkpoint
    checkpoint_path = tmp_path / "mock_tracknet.pt"
    torch.save(model.state_dict(), checkpoint_path)

    # Load with tracker
    tracker = BallTrackerWrapper(model_path=str(checkpoint_path), device="cpu")

    assert tracker.model is not None
    assert tracker.device == torch.device("cpu")

    # Create test frames
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]

    # Should run without error (may return None since model is untrained)
    result = tracker.track_ball(frames, orig_width=1280, orig_height=720)

    # Result is either None or (x, y, conf)
    assert result is None or len(result) == 3


def test_ball_tracker_handles_checkpoint_dict(tmp_path):
    """Test tracker handles checkpoint with 'model_state_dict' key."""
    from models.tracknet import TrackNet

    model = TrackNet()

    # Save as checkpoint dict (common in training)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = {"model_state_dict": model.state_dict(), "epoch": 10}
    torch.save(checkpoint, checkpoint_path)

    # Load with tracker
    tracker = BallTrackerWrapper(model_path=str(checkpoint_path), device="cpu")

    assert tracker.model is not None
