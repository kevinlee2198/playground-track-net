"""Tests for JSON exporter."""

import json
import tempfile
from pathlib import Path

import pytest
from utils.json_exporter import MatchJSONExporter


def test_export_minimal_creates_valid_json():
    """Test minimal export creates valid JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test_minimal.json")

        MatchJSONExporter.export_minimal(
            output_path=output_path,
            match_id="2026-04-10_123456",
            date="2026-04-10T12:34:56",
            sport="tennis",
            player_names=["Player 1", "Player 2"],
            winner_id=0,
            final_score="6-4, 6-3",
            player_stats=[
                {
                    "player_id": 0,
                    "points": 48,
                    "aces": 5,
                    "winners": 20,
                    "errors": 12,
                    "first_serve_pct": 65.0,
                },
                {
                    "player_id": 1,
                    "points": 35,
                    "aces": 2,
                    "winners": 15,
                    "errors": 18,
                    "first_serve_pct": 58.0,
                },
            ],
        )

        # Verify file exists
        assert Path(output_path).exists()

        # Verify valid JSON
        with open(output_path) as f:
            data = json.load(f)

        # Verify content
        assert data["match_id"] == "2026-04-10_123456"
        assert data["date"] == "2026-04-10T12:34:56"
        assert data["sport"] == "tennis"
        assert data["players"] == ["Player 1", "Player 2"]
        assert data["winner"] == 0
        assert data["final_score"] == "6-4, 6-3"
        assert len(data["player_stats"]) == 2
        assert data["player_stats"][0]["player_id"] == 0
        assert data["player_stats"][0]["points"] == 48


def test_export_full_creates_valid_json():
    """Test full export creates valid JSON file with all fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test_full.json")

        match_metadata = {
            "match_id": "2026-04-10_123456",
            "date": "2026-04-10T12:34:56",
            "sport": "tennis",
            "video_path": "/path/to/video.mp4",
        }

        players = [
            {"player_id": 0, "name": "Roger Federer"},
            {"player_id": 1, "name": "Rafael Nadal"},
        ]

        final_score = {"sets": [{"games": [6, 4]}, {"games": [6, 3]}], "winner_id": 0}

        statistics = {
            "player_0": {
                "points_won": 48,
                "aces": 5,
                "double_faults": 2,
                "winners": 20,
                "unforced_errors": 12,
            },
            "player_1": {
                "points_won": 35,
                "aces": 2,
                "double_faults": 4,
                "winners": 15,
                "unforced_errors": 18,
            },
        }

        events = [
            {"type": "serve_start", "frame": 10, "player_id": 0},
            {"type": "hit", "frame": 25, "player_id": 1},
            {"type": "rally_end", "frame": 50, "winner": 0},
        ]

        rally_log = [
            {
                "rally_id": 0,
                "start_frame": 10,
                "end_frame": 50,
                "server_id": 0,
                "winner_id": 0,
                "num_shots": 3,
            }
        ]

        processing_info = {
            "tracknet_model": "weights/tracknet_v2.pt",
            "yolo_model": "yolov8x-pose.pt",
            "fps": 30.0,
            "total_frames": 1800,
        }

        MatchJSONExporter.export_full(
            output_path=output_path,
            match_metadata=match_metadata,
            players=players,
            final_score=final_score,
            statistics=statistics,
            events=events,
            rally_log=rally_log,
            processing_info=processing_info,
        )

        # Verify file exists
        assert Path(output_path).exists()

        # Verify valid JSON
        with open(output_path) as f:
            data = json.load(f)

        # Verify all sections present
        assert "match_metadata" in data
        assert "players" in data
        assert "final_score" in data
        assert "statistics" in data
        assert "events" in data
        assert "rally_log" in data
        assert "processing_info" in data

        # Verify content
        assert data["match_metadata"]["match_id"] == "2026-04-10_123456"
        assert len(data["players"]) == 2
        assert data["players"][0]["name"] == "Roger Federer"
        assert data["final_score"]["winner_id"] == 0
        assert len(data["events"]) == 3
        assert len(data["rally_log"]) == 1


def test_export_minimal_creates_output_directory():
    """Test exporter creates output directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use nested path that doesn't exist yet
        output_path = str(Path(tmpdir) / "nested" / "dir" / "test.json")

        MatchJSONExporter.export_minimal(
            output_path=output_path,
            match_id="test",
            date="2026-04-10",
            sport="tennis",
            player_names=["P1", "P2"],
            winner_id=0,
            final_score="6-0",
            player_stats=[{"player_id": 0}, {"player_id": 1}],
        )

        # Verify directory was created
        assert Path(output_path).parent.exists()
        assert Path(output_path).exists()


def test_export_full_creates_output_directory():
    """Test full exporter creates output directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use nested path that doesn't exist yet
        output_path = str(Path(tmpdir) / "nested" / "dir" / "test_full.json")

        MatchJSONExporter.export_full(
            output_path=output_path,
            match_metadata={"match_id": "test"},
            players=[],
            final_score={},
            statistics={},
            events=[],
            rally_log=[],
            processing_info={},
        )

        # Verify directory was created
        assert Path(output_path).parent.exists()
        assert Path(output_path).exists()


def test_export_minimal_handles_empty_stats():
    """Test minimal export handles empty stats gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test_empty.json")

        MatchJSONExporter.export_minimal(
            output_path=output_path,
            match_id="test",
            date="2026-04-10",
            sport="tennis",
            player_names=["P1", "P2"],
            winner_id=0,
            final_score="0-0",
            player_stats=[{}, {}],  # Empty stats
        )

        # Verify file created and valid
        assert Path(output_path).exists()

        with open(output_path) as f:
            data = json.load(f)

        assert len(data["player_stats"]) == 2
        assert data["player_stats"][0] == {}


def test_export_minimal_json_formatting():
    """Test JSON is properly formatted with indentation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test_format.json")

        MatchJSONExporter.export_minimal(
            output_path=output_path,
            match_id="test",
            date="2026-04-10",
            sport="tennis",
            player_names=["P1", "P2"],
            winner_id=0,
            final_score="6-0",
            player_stats=[{"points": 24}, {"points": 10}],
        )

        # Read raw file content
        with open(output_path) as f:
            content = f.read()

        # Verify it's indented (has newlines and spaces)
        assert "\n" in content
        assert "  " in content

        # Verify it's valid JSON
        json.loads(content)


def test_export_full_json_formatting():
    """Test full export JSON is properly formatted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test_full_format.json")

        MatchJSONExporter.export_full(
            output_path=output_path,
            match_metadata={"match_id": "test"},
            players=[{"name": "P1"}],
            final_score={"winner_id": 0},
            statistics={"player_0": {"points": 24}},
            events=[],
            rally_log=[],
            processing_info={"fps": 30.0},
        )

        # Read raw file content
        with open(output_path) as f:
            content = f.read()

        # Verify it's indented
        assert "\n" in content
        assert "  " in content

        # Verify it's valid JSON
        json.loads(content)


def test_export_minimal_overwrites_existing_file():
    """Test exporter overwrites existing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "test_overwrite.json")

        # Write first file
        MatchJSONExporter.export_minimal(
            output_path=output_path,
            match_id="first",
            date="2026-04-10",
            sport="tennis",
            player_names=["P1", "P2"],
            winner_id=0,
            final_score="6-0",
            player_stats=[{}, {}],
        )

        # Overwrite with second file
        MatchJSONExporter.export_minimal(
            output_path=output_path,
            match_id="second",
            date="2026-04-11",
            sport="badminton",
            player_names=["P3", "P4"],
            winner_id=1,
            final_score="21-19",
            player_stats=[{}, {}],
        )

        # Verify second data is present
        with open(output_path) as f:
            data = json.load(f)

        assert data["match_id"] == "second"
        assert data["sport"] == "badminton"
        assert data["winner"] == 1
