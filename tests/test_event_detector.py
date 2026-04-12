"""Tests for EventDetector."""

import numpy as np
import pytest

from inference.event_detector import EventDetector


def test_event_detector_initialization():
    """Test detector initializes correctly."""
    detector = EventDetector()

    assert detector.sport == "tennis"
    assert detector.rally_active is False
    assert detector.last_bounce_frame is None
    assert detector.last_hit_frame is None


def test_event_detector_custom_sport():
    """Test detector with custom sport."""
    detector = EventDetector(sport="badminton")

    assert detector.sport == "badminton"


def test_bounce_detection_simple():
    """Test bounce detection with simple falling and bouncing ball."""
    detector = EventDetector()

    # Simulate ball falling (y increasing) then bouncing (y decreasing)
    ball_trajectory = [
        (0, (100, 100)),   # Start
        (1, (100, 150)),   # Falling (+50)
        (2, (100, 200)),   # Falling (+50)
        (3, (100, 250)),   # Falling (+50)
        (4, (100, 280)),   # At ground (+30)
        (5, (100, 250)),   # Bouncing up (-30)
        (6, (100, 200)),   # Going up (-50)
    ]

    all_events = []
    for frame_id, ball_pos in ball_trajectory:
        events = detector.process_frame(frame_id, ball_pos, np.array([]))
        all_events.extend(events)

    # Should detect bounce around frame 5
    bounce_events = [e for e in all_events if e["type"] == "bounce"]
    assert len(bounce_events) >= 1


def test_bounce_detection_no_bounce():
    """Test no bounce detected for ball moving in one direction."""
    detector = EventDetector()

    # Ball only moving down (no bounce)
    ball_trajectory = [
        (0, (100, 100)),
        (1, (100, 150)),
        (2, (100, 200)),
        (3, (100, 250)),
        (4, (100, 300)),
    ]

    all_events = []
    for frame_id, ball_pos in ball_trajectory:
        events = detector.process_frame(frame_id, ball_pos, np.array([]))
        all_events.extend(events)

    # Should not detect bounce
    bounce_events = [e for e in all_events if e["type"] == "bounce"]
    assert len(bounce_events) == 0


def test_hit_detection_ball_inside_bbox():
    """Test hit detection when ball is inside player bbox."""
    detector = EventDetector()

    # Player bbox
    player_tracks = np.array([
        [50, 100, 150, 400, 0]  # Player 0
    ])

    # Ball inside player bbox
    ball_pos = (100, 200)

    events = detector.process_frame(0, ball_pos, player_tracks)

    # First hit is a serve
    assert len(events) == 1
    assert events[0]["type"] == "serve_start"
    assert events[0]["player_id"] == 0


def test_hit_detection_ball_outside_bbox():
    """Test no hit detected when ball is outside player bbox."""
    detector = EventDetector()

    # Player bbox
    player_tracks = np.array([
        [50, 100, 150, 400, 0]
    ])

    # Ball far from player
    ball_pos = (500, 200)

    events = detector.process_frame(0, ball_pos, player_tracks)

    # No hit
    assert len(events) == 0


def test_serve_detection_first_hit():
    """Test first hit of rally is detected as serve."""
    detector = EventDetector()

    player_tracks = np.array([
        [50, 100, 150, 400, 0]
    ])

    # First contact with player
    ball_pos = (100, 200)
    events = detector.process_frame(0, ball_pos, player_tracks)

    assert len(events) == 1
    assert events[0]["type"] == "serve_start"
    assert events[0]["player_id"] == 0
    assert detector.rally_active is True


def test_hit_detection_after_serve():
    """Test subsequent hits are detected as 'hit' not 'serve'."""
    detector = EventDetector()

    player_tracks_p0 = np.array([[50, 100, 150, 400, 0]])
    player_tracks_p1 = np.array([[400, 100, 500, 400, 1]])

    # Frame 0: Serve by player 0
    events = detector.process_frame(0, (100, 200), player_tracks_p0)
    assert events[0]["type"] == "serve_start"

    # Frame 1-2: Ball in flight
    detector.process_frame(1, (200, 200), np.array([]))
    detector.process_frame(2, (300, 200), np.array([]))

    # Frame 3: Hit by player 1
    events = detector.process_frame(3, (450, 200), player_tracks_p1)

    # Should be a hit, not serve
    hit_events = [e for e in events if e["type"] == "hit"]
    assert len(hit_events) == 1
    assert hit_events[0]["player_id"] == 1


def test_multiple_players_hit_detection():
    """Test hit detection with multiple players."""
    detector = EventDetector()

    player_tracks = np.array([
        [50, 100, 150, 400, 0],   # Player 0 (left)
        [400, 100, 500, 400, 1],  # Player 1 (right)
    ])

    # Ball near player 1
    ball_pos = (450, 200)
    events = detector.process_frame(0, ball_pos, player_tracks)

    assert len(events) == 1
    assert events[0]["player_id"] == 1


def test_no_consecutive_hits_same_player():
    """Test same player cannot hit ball consecutively."""
    detector = EventDetector()

    player_tracks = np.array([[50, 100, 150, 400, 0]])

    # Frame 0: First hit
    events = detector.process_frame(0, (100, 200), player_tracks)
    assert len(events) == 1

    # Frame 1: Ball still near same player (should not trigger new hit)
    events = detector.process_frame(1, (100, 200), player_tracks)
    assert len(events) == 0


def test_hit_distance_threshold():
    """Test hit detection uses distance threshold."""
    detector = EventDetector(hit_distance_threshold=30.0)

    player_tracks = np.array([
        [50, 100, 150, 400, 0]
    ])

    # Ball just outside bbox but within threshold
    ball_pos = (170, 200)  # x=170, bbox x2=150, threshold=30, so 170 < 180

    events = detector.process_frame(0, ball_pos, player_tracks)

    # Should detect hit due to threshold
    assert len(events) == 1


def test_process_frame_with_none_ball():
    """Test processing frame when ball is not detected."""
    detector = EventDetector()

    player_tracks = np.array([[50, 100, 150, 400, 0]])

    events = detector.process_frame(0, None, player_tracks)

    # No events when ball not visible
    assert len(events) == 0


def test_rally_end_event():
    """Test manually ending a rally."""
    detector = EventDetector()

    # Start a rally
    player_tracks = np.array([[50, 100, 150, 400, 0]])
    detector.process_frame(0, (100, 200), player_tracks)

    assert detector.rally_active is True

    # End rally
    event = detector.end_rally(frame_id=10, winner=0, reason="ace")

    assert event["type"] == "rally_end"
    assert event["winner"] == 0
    assert event["reason"] == "ace"
    assert detector.rally_active is False


def test_reset_clears_state():
    """Test reset clears all detector state."""
    detector = EventDetector()

    # Create some state
    player_tracks = np.array([[50, 100, 150, 400, 0]])
    detector.process_frame(0, (100, 200), player_tracks)
    detector.process_frame(1, (100, 250), np.array([]))

    # Reset
    detector.reset()

    assert len(detector.ball_history) == 0
    assert detector.last_bounce_frame is None
    assert detector.last_hit_frame is None
    assert detector.rally_active is False


def test_bounce_event_structure():
    """Test bounce event has correct structure."""
    detector = EventDetector()

    # Create bounce
    ball_trajectory = [
        (0, (100, 100)),
        (1, (100, 150)),
        (2, (100, 200)),
        (3, (100, 250)),
        (4, (100, 280)),
        (5, (100, 250)),
    ]

    all_events = []
    for frame_id, ball_pos in ball_trajectory:
        events = detector.process_frame(frame_id, ball_pos, np.array([]))
        all_events.extend(events)

    bounce_events = [e for e in all_events if e["type"] == "bounce"]

    if len(bounce_events) > 0:
        event = bounce_events[0]
        assert "frame" in event
        assert "data" in event
        assert "position" in event["data"]
        assert "valid" in event["data"]


def test_serve_event_structure():
    """Test serve event has correct structure."""
    detector = EventDetector()

    player_tracks = np.array([[50, 100, 150, 400, 0]])
    events = detector.process_frame(0, (100, 200), player_tracks)

    assert len(events) == 1
    event = events[0]

    assert event["type"] == "serve_start"
    assert "frame" in event
    assert "player_id" in event
    assert "data" in event
    assert "is_first_serve" in event["data"]


def test_hit_event_structure():
    """Test hit event has correct structure."""
    detector = EventDetector()

    player_tracks_p0 = np.array([[50, 100, 150, 400, 0]])
    player_tracks_p1 = np.array([[400, 100, 500, 400, 1]])

    # Serve first
    detector.process_frame(0, (100, 200), player_tracks_p0)
    detector.process_frame(1, (200, 200), np.array([]))
    detector.process_frame(2, (300, 200), np.array([]))

    # Hit
    events = detector.process_frame(3, (450, 200), player_tracks_p1)

    hit_events = [e for e in events if e["type"] == "hit"]
    assert len(hit_events) == 1

    event = hit_events[0]
    assert "frame" in event
    assert "player_id" in event
    assert "data" in event
    assert "shot_type" in event["data"]


def test_ball_history_max_length():
    """Test ball history doesn't exceed max length."""
    detector = EventDetector()

    # Process many frames
    for i in range(20):
        detector.process_frame(i, (100, 100 + i * 10), np.array([]))

    # History should be capped at max_history_length (10)
    assert len(detector.ball_history) == detector.max_history_length


def test_bounce_minimum_gap():
    """Test minimum gap between bounce detections."""
    detector = EventDetector()

    # Create conditions that might trigger multiple bounces
    ball_trajectory = [
        (0, (100, 100)),
        (1, (100, 150)),
        (2, (100, 200)),
        (3, (100, 250)),
        (4, (100, 280)),
        (5, (100, 250)),
        (6, (100, 280)),  # Might look like another bounce
        (7, (100, 250)),
    ]

    all_events = []
    for frame_id, ball_pos in ball_trajectory:
        events = detector.process_frame(frame_id, ball_pos, np.array([]))
        all_events.extend(events)

    bounce_events = [e for e in all_events if e["type"] == "bounce"]

    # Should not detect bounces too close together
    if len(bounce_events) >= 2:
        frame_gap = bounce_events[1]["frame"] - bounce_events[0]["frame"]
        assert frame_gap >= 5


def test_empty_player_tracks():
    """Test processing with empty player tracks."""
    detector = EventDetector()

    # No players on court
    events = detector.process_frame(0, (100, 200), np.array([]).reshape(0, 5))

    # Should not crash, no hit detection
    assert isinstance(events, list)
