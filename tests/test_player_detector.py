"""Tests for PlayerDetector."""

import numpy as np
import pytest

from models.player_detector import PlayerDetector


def test_player_detector_stub_mode():
    """Test detector works in stub mode (no YOLO)."""
    detector = PlayerDetector(stub=True)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    assert detections.shape == (2, 5)  # 2 players, 5 values each
    assert detections.dtype == np.float32


def test_player_detector_stub_returns_two_players():
    """Test stub mode always returns 2 players."""
    detector = PlayerDetector(stub=True)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    assert len(detections) == 2


def test_player_detector_stub_bbox_format():
    """Test stub detections have correct format [x1, y1, x2, y2, conf]."""
    detector = PlayerDetector(stub=True)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    for det in detections:
        x1, y1, x2, y2, conf = det

        # Bounding box should be valid
        assert x1 < x2
        assert y1 < y2

        # Should be within frame bounds
        assert 0 <= x1 < 1280
        assert 0 <= y1 < 720
        assert 0 < x2 <= 1280
        assert 0 < y2 <= 720

        # Confidence should be in [0, 1]
        assert 0 <= conf <= 1.0


def test_player_detector_stub_positions():
    """Test stub players are positioned on left and right."""
    detector = PlayerDetector(stub=True)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    # Player 1 should be on left side (x1 < width/2)
    assert detections[0, 0] < 640  # x1 < width/2

    # Player 2 should be on right side (x1 > width/2)
    assert detections[1, 0] > 640  # x1 > width/2


def test_player_detector_stub_scales_with_frame_size():
    """Test stub detections scale with frame size."""
    detector = PlayerDetector(stub=True)

    # Small frame
    small_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    small_dets = detector.detect(small_frame)

    # Large frame
    large_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    large_dets = detector.detect(large_frame)

    # Positions should scale proportionally
    # Player 1 x1: 0.1 * width
    assert abs(small_dets[0, 0] / 640 - large_dets[0, 0] / 1920) < 0.01

    # Player 2 x1: 0.7 * width
    assert abs(small_dets[1, 0] / 640 - large_dets[1, 0] / 1920) < 0.01


def test_player_detector_stub_high_confidence():
    """Test stub detections have high confidence."""
    detector = PlayerDetector(stub=True)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    # Both detections should have high confidence (>0.9)
    assert detections[0, 4] > 0.9
    assert detections[1, 4] > 0.9


def test_player_detector_detect_with_keypoints_stub():
    """Test detect_with_keypoints in stub mode."""
    detector = PlayerDetector(stub=True)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections, keypoints = detector.detect_with_keypoints(frame)

    assert detections.shape == (2, 5)
    assert keypoints is None  # Stub mode doesn't return keypoints


def test_player_detector_stub_model_name():
    """Test custom model name is stored."""
    detector = PlayerDetector(stub=True, model_name="yolov8n-pose.pt")

    assert detector.model_name == "yolov8n-pose.pt"
    assert detector.stub is True


def test_player_detector_multiple_frames():
    """Test detector can process multiple frames."""
    detector = PlayerDetector(stub=True)

    for i in range(10):
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert detections.shape == (2, 5)


def test_player_detector_consistent_stub_output():
    """Test stub mode returns consistent detections."""
    detector = PlayerDetector(stub=True)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Run detection multiple times
    det1 = detector.detect(frame)
    det2 = detector.detect(frame)
    det3 = detector.detect(frame)

    # Results should be identical
    np.testing.assert_array_equal(det1, det2)
    np.testing.assert_array_equal(det2, det3)


def test_player_detector_different_frame_sizes():
    """Test detector handles different frame sizes."""
    detector = PlayerDetector(stub=True)

    sizes = [(480, 640), (720, 1280), (1080, 1920), (360, 640)]

    for h, w in sizes:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert detections.shape == (2, 5)

        # Detections should be within bounds
        for det in detections:
            assert 0 <= det[0] < w  # x1
            assert 0 <= det[1] < h  # y1
            assert 0 < det[2] <= w  # x2
            assert 0 < det[3] <= h  # y2


@pytest.mark.skipif(True, reason="Requires ultralytics installation and model download")
def test_player_detector_real_mode():
    """Test detector with real YOLO model (integration test)."""
    detector = PlayerDetector(stub=False, model_name="yolov8n-pose.pt")

    # Create a dummy frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = detector.detect(frame)

    # Should return array (may be empty for blank frame)
    assert isinstance(detections, np.ndarray)
    assert detections.shape[1] == 5 if len(detections) > 0 else True
