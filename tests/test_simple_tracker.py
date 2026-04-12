"""Tests for SimpleTracker."""

import numpy as np
import pytest

from models.trackers.simple_tracker import SimpleTracker, compute_iou


def test_compute_iou_identical_boxes():
    """Test IoU of identical boxes is 1.0."""
    box1 = np.array([10, 20, 50, 80])
    box2 = np.array([10, 20, 50, 80])

    iou = compute_iou(box1, box2)

    assert iou == 1.0


def test_compute_iou_no_overlap():
    """Test IoU of non-overlapping boxes is 0.0."""
    box1 = np.array([10, 20, 50, 80])
    box2 = np.array([100, 200, 150, 280])

    iou = compute_iou(box1, box2)

    assert iou == 0.0


def test_compute_iou_partial_overlap():
    """Test IoU of partially overlapping boxes."""
    box1 = np.array([0, 0, 100, 100])
    box2 = np.array([50, 50, 150, 150])

    iou = compute_iou(box1, box2)

    # Intersection: 50x50 = 2500
    # Union: 10000 + 10000 - 2500 = 17500
    # IoU: 2500 / 17500 = 0.1428...
    assert abs(iou - 0.1428) < 0.01


def test_compute_iou_nested_boxes():
    """Test IoU when one box contains another."""
    box1 = np.array([0, 0, 100, 100])  # Larger
    box2 = np.array([25, 25, 75, 75])  # Smaller, nested

    iou = compute_iou(box1, box2)

    # Intersection: 50x50 = 2500
    # Union: 10000 (box1 area, since box2 is inside)
    # IoU: 2500 / 10000 = 0.25
    assert abs(iou - 0.25) < 0.01


def test_simple_tracker_initialization():
    """Test tracker initializes correctly."""
    tracker = SimpleTracker()

    assert len(tracker.tracks) == 0
    assert tracker.next_id == 0
    assert tracker.iou_threshold == 0.3
    assert tracker.max_age == 30


def test_simple_tracker_custom_params():
    """Test tracker with custom parameters."""
    tracker = SimpleTracker(iou_threshold=0.5, max_age=10)

    assert tracker.iou_threshold == 0.5
    assert tracker.max_age == 10


def test_simple_tracker_first_frame():
    """Test tracker creates tracks on first frame."""
    tracker = SimpleTracker()

    detections = np.array([
        [50, 100, 150, 400, 0.95],
        [400, 100, 500, 400, 0.93],
    ])

    tracks = tracker.update(detections)

    assert len(tracks) == 2
    assert tracks[0, 4] == 0  # Track ID 0
    assert tracks[1, 4] == 1  # Track ID 1


def test_simple_tracker_maintains_ids():
    """Test tracker maintains IDs across frames."""
    tracker = SimpleTracker()

    # Frame 1: 2 detections
    dets1 = np.array([
        [50, 100, 150, 400],
        [400, 100, 500, 400],
    ])
    tracks1 = tracker.update(dets1)

    # Frame 2: Same players, moved slightly (high IoU)
    dets2 = np.array([
        [55, 105, 155, 405],  # Player 1 moved
        [405, 105, 505, 405],  # Player 2 moved
    ])
    tracks2 = tracker.update(dets2)

    # IDs should be maintained
    assert tracks1[0, 4] == tracks2[0, 4]  # ID 0
    assert tracks1[1, 4] == tracks2[1, 4]  # ID 1


def test_simple_tracker_swapped_detections():
    """Test tracker handles swapped detection order."""
    tracker = SimpleTracker()

    # Frame 1: Player 1 left, Player 2 right
    dets1 = np.array([
        [50, 100, 150, 400],   # Player 1 (left)
        [400, 100, 500, 400],  # Player 2 (right)
    ])
    tracks1 = tracker.update(dets1)

    # Frame 2: Detections in swapped order (but same spatial positions)
    dets2 = np.array([
        [405, 105, 505, 405],  # Player 2 (right) - listed first
        [55, 105, 155, 405],   # Player 1 (left) - listed second
    ])
    tracks2 = tracker.update(dets2)

    # Tracker should match based on IoU, not order
    # Find which track corresponds to left player
    left_track = tracks2[np.argmin(tracks2[:, 0])]  # Smallest x1
    right_track = tracks2[np.argmax(tracks2[:, 0])]  # Largest x1

    # IDs should match original positions
    assert left_track[4] == 0  # Left player keeps ID 0
    assert right_track[4] == 1  # Right player keeps ID 1


def test_simple_tracker_missing_detection():
    """Test tracker handles missing detections."""
    tracker = SimpleTracker(max_age=5)

    # Frame 1: 2 detections
    dets1 = np.array([
        [50, 100, 150, 400],
        [400, 100, 500, 400],
    ])
    tracks1 = tracker.update(dets1)

    # Frame 2: Only 1 detection
    dets2 = np.array([
        [55, 105, 155, 405],  # Only player 1
    ])
    tracks2 = tracker.update(dets2)

    # Should still return only the matched detection
    assert len(tracks2) == 1
    assert tracks2[0, 4] == 0  # Player 1 ID


def test_simple_tracker_empty_frame():
    """Test tracker handles empty detections."""
    tracker = SimpleTracker()

    # Frame 1: 2 detections
    dets1 = np.array([
        [50, 100, 150, 400],
        [400, 100, 500, 400],
    ])
    tracker.update(dets1)

    # Frame 2: No detections
    dets2 = np.array([]).reshape(0, 4)
    tracks2 = tracker.update(dets2)

    # Tracks are aged but still returned (age=1, max_age=30)
    assert len(tracks2) == 2

    # Verify tracks are aged
    for track in tracker.tracks:
        assert track["age"] == 1


def test_simple_tracker_track_aging():
    """Test tracker ages tracks and removes old ones."""
    tracker = SimpleTracker(max_age=2)

    # Frame 1: 1 detection
    dets1 = np.array([[50, 100, 150, 400]])
    tracker.update(dets1)

    # Frames 2-3: No detections (age track)
    tracker.update(np.array([]).reshape(0, 4))  # Age = 1
    tracker.update(np.array([]).reshape(0, 4))  # Age = 2

    # Frame 4: Still no detections (should remove track)
    tracker.update(np.array([]).reshape(0, 4))  # Age = 3 > max_age

    # Track should be gone
    assert len(tracker.tracks) == 0


def test_simple_tracker_new_detection_after_disappearance():
    """Test tracker creates new ID for new detection."""
    tracker = SimpleTracker(iou_threshold=0.5, max_age=2)

    # Frame 1: 1 detection
    dets1 = np.array([[50, 100, 150, 400]])
    tracks1 = tracker.update(dets1)
    first_id = tracks1[0, 4]

    # Frames 2-4: No detections (track disappears)
    for _ in range(3):
        tracker.update(np.array([]).reshape(0, 4))

    # Frame 5: New detection at different location
    dets5 = np.array([[400, 100, 500, 400]])
    tracks5 = tracker.update(dets5)

    # Should get new ID
    assert tracks5[0, 4] != first_id


def test_simple_tracker_reset():
    """Test tracker reset clears all tracks."""
    tracker = SimpleTracker()

    # Create some tracks
    dets = np.array([
        [50, 100, 150, 400],
        [400, 100, 500, 400],
    ])
    tracker.update(dets)

    assert len(tracker.tracks) > 0
    assert tracker.next_id > 0

    # Reset
    tracker.reset()

    assert len(tracker.tracks) == 0
    assert tracker.next_id == 0


def test_simple_tracker_three_players():
    """Test tracker handles more than 2 detections."""
    tracker = SimpleTracker()

    # Frame 1: 3 detections
    dets1 = np.array([
        [50, 100, 150, 400],
        [250, 100, 350, 400],
        [450, 100, 550, 400],
    ])
    tracks1 = tracker.update(dets1)

    assert len(tracks1) == 3
    assert tracks1[0, 4] == 0
    assert tracks1[1, 4] == 1
    assert tracks1[2, 4] == 2


def test_simple_tracker_low_iou_creates_new_track():
    """Test tracker creates new track if IoU is below threshold."""
    tracker = SimpleTracker(iou_threshold=0.5)

    # Frame 1: 1 detection
    dets1 = np.array([[50, 100, 150, 400]])
    tracks1 = tracker.update(dets1)

    # Frame 2: Detection far away (low IoU)
    dets2 = np.array([[450, 100, 550, 400]])
    tracks2 = tracker.update(dets2)

    # Should create new track (no match due to low IoU)
    # But only active tracks (age=0) are returned
    assert len(tracks2) == 1
    assert tracks2[0, 4] == 1  # New ID


def test_simple_tracker_with_confidence():
    """Test tracker handles detections with confidence."""
    tracker = SimpleTracker()

    # Detections with confidence (5th column)
    dets = np.array([
        [50, 100, 150, 400, 0.95],
        [400, 100, 500, 400, 0.87],
    ])

    tracks = tracker.update(dets)

    # Should work (confidence ignored)
    assert len(tracks) == 2


def test_simple_tracker_bbox_update():
    """Test tracker updates bounding box positions."""
    tracker = SimpleTracker()

    # Frame 1
    dets1 = np.array([[50, 100, 150, 400]])
    tracks1 = tracker.update(dets1)

    # Frame 2: Same player, moved
    dets2 = np.array([[60, 110, 160, 410]])
    tracks2 = tracker.update(dets2)

    # Position should be updated
    assert tracks2[0, 0] == 60  # x1
    assert tracks2[0, 1] == 110  # y1
    assert tracks2[0, 4] == tracks1[0, 4]  # Same ID
