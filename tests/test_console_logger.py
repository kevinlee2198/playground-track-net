"""Tests for console logger."""

import pytest
from utils.console_logger import ConsoleMatchLogger


def test_console_logger_initializes():
    """Test logger initializes without error."""
    logger = ConsoleMatchLogger(["Player 1", "Player 2"], "tennis")
    assert logger.player_names == ["Player 1", "Player 2"]
    assert logger.sport == "tennis"


def test_console_logger_logs_serve_event():
    """Test logger handles serve event."""
    logger = ConsoleMatchLogger(["Roger Federer", "Rafael Nadal"], "tennis")

    event = {"type": "serve_start", "player_id": 0, "data": {"is_first_serve": True}}

    # Should not crash
    logger.log_event(event, frame_id=100, timestamp=3.33)


def test_console_logger_logs_bounce_event():
    """Test logger handles bounce event."""
    logger = ConsoleMatchLogger(["P1", "P2"], "tennis")

    event = {
        "type": "bounce",
        "data": {"zone": "left_service_box", "valid": True},
    }

    # Should not crash
    logger.log_event(event, frame_id=150, timestamp=5.0)


def test_console_logger_logs_hit_event():
    """Test logger handles hit event."""
    logger = ConsoleMatchLogger(["P1", "P2"], "tennis")

    event = {
        "type": "hit",
        "player_id": 1,
        "data": {"shot_type": "forehand"},
    }

    # Should not crash
    logger.log_event(event, frame_id=200, timestamp=6.67)


def test_console_logger_logs_rally_end():
    """Test logger handles rally end event."""
    logger = ConsoleMatchLogger(["P1", "P2"], "tennis")

    event = {"type": "rally_end", "winner": 0, "reason": "ace"}

    # Should not crash
    logger.log_event(event, frame_id=250, timestamp=8.33)


def test_console_logger_logs_score():
    """Test logger logs score."""
    logger = ConsoleMatchLogger(["P1", "P2"], "tennis")

    # Should not crash
    logger.log_score("30-15 | Games: 2-1 | Sets: 1-0")


def test_console_logger_logs_stats():
    """Test logger logs player stats."""
    logger = ConsoleMatchLogger(["P1", "P2"], "tennis")

    stats = {
        "points_won": 10,
        "aces": 3,
        "winners": 5,
        "unforced_errors": 2,
        "first_serves_in": 8,
        "first_serves_total": 10,
    }

    # Should not crash
    logger.log_stats(player_id=0, stats=stats)


def test_console_logger_logs_final_summary():
    """Test logger logs final summary."""
    logger = ConsoleMatchLogger(["Roger Federer", "Rafael Nadal"], "tennis")

    player_stats = [
        {
            "points_won": 78,
            "aces": 12,
            "double_faults": 3,
            "winners": 45,
            "unforced_errors": 28,
            "first_serves_in": 48,
            "first_serves_total": 77,
            "service_points_total": 77,
            "total_shots": 450,
        },
        {
            "points_won": 65,
            "aces": 8,
            "double_faults": 5,
            "winners": 38,
            "unforced_errors": 35,
            "first_serves_in": 42,
            "first_serves_total": 72,
            "service_points_total": 72,
            "total_shots": 420,
        },
    ]

    # Should not crash
    logger.log_final_summary(
        final_score="6-4, 6-3",
        winner_id=0,
        player_stats=player_stats,
        duration_seconds=4980,
    )


def test_console_logger_handles_missing_data():
    """Test logger gracefully handles missing event data."""
    logger = ConsoleMatchLogger(["P1", "P2"], "tennis")

    # Event with minimal data
    event = {"type": "bounce"}

    # Should not crash even with missing data
    logger.log_event(event, frame_id=100, timestamp=3.33)


def test_console_logger_handles_empty_stats():
    """Test logger handles empty stats dict."""
    logger = ConsoleMatchLogger(["P1", "P2"], "tennis")

    # Empty stats
    stats = {}

    # Should not crash
    logger.log_stats(player_id=0, stats=stats)
