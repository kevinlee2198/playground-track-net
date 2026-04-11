# MVP Implementation Guide: Console Output First

**Date:** 2026-04-10
**Status:** Implementation Ready
**Goal:** Get multi-object scoring working with simple console/JSON output, defer backend integration

---

## 1. MVP Scope: What We're Building First

### Phase 1: Console Output (This Implementation) ✅

**Input:** Video file
**Output:**
- Real-time console output with score and stats
- Final JSON file with complete match data
- Annotated video

**Components:**
- Ball tracking (existing TrackNet)
- Player detection (YOLOv8-pose)
- Event detection (heuristics)
- Scoring engine (tennis rules)
- Stats aggregation (in-memory)
- **Console logger** (pretty print)
- **JSON export** (for backend integration)

**No Database, No API, No Dashboard** - Keep it simple!

### Phase 2: Backend Integration (Later)

Your existing backend can:
1. Accept the JSON output via API call
2. Store in your database
3. Serve to your frontend
4. Handle user management, authentication, etc.

We'll design the JSON format to be backend-agnostic.

---

## 2. Architecture (Simplified)

```
Video Input
  ↓
TrackNet (ball) + YOLOv8 (players)
  ↓
Event Detector (bounce, hit, serve)
  ↓
Scoring Engine (points, games, sets)
  ↓
Stats Aggregator (aces, winners, errors)
  ↓
┌─────────────────────┬───────────────────────┐
│   Console Logger    │   JSON Exporter       │
│   (real-time)       │   (final output)      │
│                     │                       │
│ Frame 1234          │ {                     │
│ Score: 30-15        │   "match_id": "...",  │
│ [ACE] Player 1!     │   "final_score": ..., │
│                     │   "events": [...],    │
│ Player 1 Stats:     │   "stats": {...}      │
│   Aces: 5          │ }                     │
│   Winners: 12      │                       │
└─────────────────────┴───────────────────────┘
                ↓
        Annotated Video
```

**Later:** POST JSON to your backend API

---

## 3. Console Output Format

### 3.1 Real-Time Match Output

```
╔══════════════════════════════════════════════════════════════╗
║              TrackNet Match Analysis                          ║
║  Tennis Singles - 2024-04-10                                  ║
║  Player 1 vs Player 2                                         ║
╚══════════════════════════════════════════════════════════════╝

Frame 0000 | 00:00.0 | Warm-up
Frame 0120 | 00:04.0 | Match Start

─────────────────────────────────────────────────────────────
  SERVE - Player 1
─────────────────────────────────────────────────────────────
Frame 0145 | 00:04.8 | [BOUNCE] In service box ✓
Frame 0148 | 00:04.9 | [HIT] Player 2 - Forehand return
Frame 0165 | 00:05.5 | [HIT] Player 1 - Backhand
Frame 0182 | 00:06.1 | [BOUNCE] Out ✗

  🎾 Point: Player 1 (unforced error by Player 2)

  Score: 15-0 | Games: 0-0 | Sets: 0-0

─────────────────────────────────────────────────────────────
  SERVE - Player 1
─────────────────────────────────────────────────────────────
Frame 0210 | 00:07.0 | [BOUNCE] Ace! ⚡

  🎾 Point: Player 1 (ace)

  Score: 30-0 | Games: 0-0 | Sets: 0-0

  Player 1 Stats:
    Points Won: 2
    Aces: 1
    Winners: 0
    Errors: 0
    First Serve: 2/2 (100%)

  Player 2 Stats:
    Points Won: 0
    Aces: 0
    Winners: 0
    Errors: 1
    First Serve: 0/0 (—)

─────────────────────────────────────────────────────────────

[Continue for entire match...]

╔══════════════════════════════════════════════════════════════╗
║                    MATCH COMPLETE                             ║
╚══════════════════════════════════════════════════════════════╝

  Final Score: 6-4, 6-3
  Winner: Player 1
  Duration: 1h 23m

  Player 1 Final Stats:
    Points Won: 78
    Aces: 12
    Double Faults: 3
    Winners: 45
    Unforced Errors: 28
    First Serve: 62% (48/77)
    Ace %: 15.6%
    Winner Rate: 23.1%

  Player 2 Final Stats:
    Points Won: 65
    Aces: 8
    Double Faults: 5
    Winners: 38
    Unforced Errors: 35
    First Serve: 58% (42/72)
    Ace %: 11.1%
    Winner Rate: 19.5%

Match data saved to: results/match_2024-04-10_001.json
Annotated video saved to: results/match_2024-04-10_001_annotated.mp4
```

### 3.2 Verbose Mode (Optional)

```bash
# Run with verbose flag for more detail
uv run python main.py score \
  --video match.mp4 \
  --verbose

# Output includes:
# - Ball position every frame
# - Player bounding boxes
# - Court homography info
# - Confidence scores
# - Debug info
```

---

## 4. JSON Output Format

### 4.1 Match Summary JSON

```json
{
  "match_metadata": {
    "match_id": "2024-04-10_001",
    "date": "2024-04-10T14:30:00Z",
    "sport": "tennis",
    "match_type": "singles",
    "location": "Local Tennis Club",
    "video_path": "videos/match.mp4",
    "duration_seconds": 4980,
    "processed_at": "2024-04-10T16:15:00Z"
  },

  "players": [
    {
      "player_id": 0,
      "name": "Player 1",
      "team": 1
    },
    {
      "player_id": 1,
      "name": "Player 2",
      "team": 2
    }
  ],

  "final_score": {
    "winner": 0,
    "score_string": "6-4, 6-3",
    "sets": [
      {"player_0": 6, "player_1": 4},
      {"player_0": 6, "player_1": 3}
    ],
    "total_games": {"player_0": 12, "player_1": 7},
    "total_points": {"player_0": 78, "player_1": 65}
  },

  "statistics": {
    "player_0": {
      "points_won": 78,
      "points_lost": 65,
      "games_won": 12,
      "games_lost": 7,
      "sets_won": 2,
      "sets_lost": 0,

      "serve": {
        "aces": 12,
        "double_faults": 3,
        "first_serves_in": 48,
        "first_serves_total": 77,
        "first_serve_pct": 62.3,
        "second_serves_in": 26,
        "second_serves_total": 29,
        "second_serve_pct": 89.7,
        "service_points_won": 52,
        "service_points_total": 77,
        "service_points_won_pct": 67.5,
        "ace_pct": 15.6
      },

      "return": {
        "return_points_won": 26,
        "return_points_total": 66,
        "return_points_won_pct": 39.4,
        "break_points_converted": 4,
        "break_points_total": 7,
        "break_point_conversion_pct": 57.1
      },

      "shots": {
        "winners": 45,
        "unforced_errors": 28,
        "forced_errors": 12,
        "forehand_winners": 24,
        "backhand_winners": 15,
        "volley_winners": 4,
        "smash_winners": 2,
        "total_shots": 456,
        "winner_rate": 9.9,
        "error_rate": 6.1
      },

      "rally": {
        "total_rallies": 143,
        "avg_rally_length": 4.2,
        "longest_rally": 18,
        "net_approaches": 23,
        "net_points_won": 15,
        "net_success_pct": 65.2
      },

      "derived_metrics": {
        "consistency_score": 3.8,
        "aggression_index": 62.5,
        "clutch_performance": 57.1
      }
    },

    "player_1": {
      // Same structure as player_0
      "points_won": 65,
      "serve": {...},
      "return": {...},
      "shots": {...},
      "rally": {...},
      "derived_metrics": {...}
    }
  },

  "events": [
    {
      "event_id": 0,
      "frame": 120,
      "timestamp": 4.0,
      "type": "serve_start",
      "player_id": 0,
      "data": {
        "is_first_serve": true,
        "position": [850, 300]
      }
    },
    {
      "event_id": 1,
      "frame": 145,
      "timestamp": 4.8,
      "type": "bounce",
      "player_id": null,
      "data": {
        "position_pixel": [920, 650],
        "position_court": [18.5, 3.2],
        "zone": "left_service_box",
        "valid": true,
        "confidence": 0.97
      }
    },
    {
      "event_id": 2,
      "frame": 148,
      "timestamp": 4.9,
      "type": "hit",
      "player_id": 1,
      "data": {
        "shot_type": "forehand",
        "position": [950, 680]
      }
    },
    // ... all events for entire match
  ],

  "rally_log": [
    {
      "rally_id": 0,
      "start_frame": 120,
      "end_frame": 182,
      "duration_seconds": 2.1,
      "shot_count": 4,
      "server": 0,
      "winner": 0,
      "reason": "unforced_error",
      "final_score": "15-0"
    },
    // ... all rallies
  ],

  "processing_info": {
    "tracknet_version": "v5",
    "yolo_version": "yolov8x-pose",
    "processing_fps": 24.5,
    "total_frames": 14940,
    "ball_detection_rate": 96.8,
    "player_detection_rate": 99.2
  }
}
```

### 4.2 Minimal JSON (for Quick Backend Integration)

If you want a lighter format:

```json
{
  "match_id": "2024-04-10_001",
  "date": "2024-04-10T14:30:00Z",
  "sport": "tennis",
  "players": ["Player 1", "Player 2"],
  "winner": 0,
  "final_score": "6-4, 6-3",
  "player_stats": [
    {
      "player_id": 0,
      "points": 78,
      "aces": 12,
      "winners": 45,
      "errors": 28,
      "first_serve_pct": 62.3
    },
    {
      "player_id": 1,
      "points": 65,
      "aces": 8,
      "winners": 38,
      "errors": 35,
      "first_serve_pct": 58.0
    }
  ]
}
```

---

## 5. Implementation Code

### 5.1 Console Logger

```python
# utils/console_logger.py

from datetime import datetime
from typing import Dict, List
import json

class ConsoleMatchLogger:
    """Pretty console output for match progress"""

    def __init__(self, player_names: List[str], sport: str = 'tennis'):
        self.player_names = player_names
        self.sport = sport
        self.start_time = datetime.now()
        self._print_header()

    def _print_header(self):
        """Print match header"""
        print("\n" + "═" * 70)
        print("              TrackNet Match Analysis".center(70))
        print(f"  {self.sport.title()} - {datetime.now().strftime('%Y-%m-%d')}".center(70))
        print(f"  {self.player_names[0]} vs {self.player_names[1]}".center(70))
        print("═" * 70 + "\n")

    def log_event(self, event: Dict, frame_id: int, timestamp: float):
        """Log single event"""
        time_str = f"{timestamp:.1f}s"
        frame_str = f"Frame {frame_id:04d}"

        if event['type'] == 'serve_start':
            print(f"\n{'─' * 70}")
            print(f"  SERVE - {self.player_names[event['player_id']]}")
            print("─" * 70)

        elif event['type'] == 'bounce':
            zone = event['data']['zone']
            valid = "✓" if event['data']['valid'] else "✗"
            print(f"{frame_str} | {time_str} | [BOUNCE] {zone} {valid}")

        elif event['type'] == 'hit':
            player = self.player_names[event['player_id']]
            shot = event['data'].get('shot_type', 'unknown')
            print(f"{frame_str} | {time_str} | [HIT] {player} - {shot}")

        elif event['type'] == 'rally_end':
            winner = self.player_names[event['winner']]
            reason = event['reason']
            print(f"\n  🎾 Point: {winner} ({reason})\n")

    def log_score(self, score_string: str):
        """Log current score"""
        print(f"  Score: {score_string}\n")

    def log_stats(self, player_id: int, stats: Dict):
        """Log player stats"""
        player = self.player_names[player_id]
        print(f"  {player} Stats:")

        # Core stats
        print(f"    Points Won: {stats['points_won']}")
        print(f"    Aces: {stats['aces']}")
        print(f"    Winners: {stats['winners']}")
        print(f"    Errors: {stats['unforced_errors']}")

        # Serve stats
        if stats['first_serves_total'] > 0:
            first_pct = stats['first_serves_in'] / stats['first_serves_total'] * 100
            print(f"    First Serve: {stats['first_serves_in']}/{stats['first_serves_total']} ({first_pct:.0f}%)")
        else:
            print(f"    First Serve: 0/0 (—)")

        print()

    def log_final_summary(self, final_score: str, winner_id: int,
                         player_stats: List[Dict], duration_seconds: int):
        """Print final match summary"""
        print("\n" + "═" * 70)
        print("                    MATCH COMPLETE".center(70))
        print("═" * 70 + "\n")

        winner = self.player_names[winner_id]
        print(f"  Final Score: {final_score}")
        print(f"  Winner: {winner}")

        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        if hours > 0:
            print(f"  Duration: {hours}h {minutes}m\n")
        else:
            print(f"  Duration: {minutes}m\n")

        # Print detailed stats for each player
        for player_id, stats in enumerate(player_stats):
            self._print_player_final_stats(player_id, stats)

    def _print_player_final_stats(self, player_id: int, stats: Dict):
        """Print detailed final stats for one player"""
        player = self.player_names[player_id]
        print(f"  {player} Final Stats:")
        print(f"    Points Won: {stats['points_won']}")
        print(f"    Aces: {stats['aces']}")
        print(f"    Double Faults: {stats['double_faults']}")
        print(f"    Winners: {stats['winners']}")
        print(f"    Unforced Errors: {stats['unforced_errors']}")

        if stats['first_serves_total'] > 0:
            first_pct = stats['first_serves_in'] / stats['first_serves_total'] * 100
            print(f"    First Serve: {first_pct:.0f}% ({stats['first_serves_in']}/{stats['first_serves_total']})")

        if stats.get('service_points_total', 0) > 0:
            ace_pct = stats['aces'] / stats['service_points_total'] * 100
            print(f"    Ace %: {ace_pct:.1f}%")

        if stats.get('total_shots', 0) > 0:
            winner_rate = stats['winners'] / stats['total_shots'] * 100
            print(f"    Winner Rate: {winner_rate:.1f}%")

        print()
```

### 5.2 JSON Exporter

```python
# utils/json_exporter.py

import json
from datetime import datetime
from typing import Dict, List
from pathlib import Path

class MatchJSONExporter:
    """Export match data to JSON"""

    @staticmethod
    def export_full(
        output_path: str,
        match_metadata: Dict,
        players: List[Dict],
        final_score: Dict,
        statistics: Dict,
        events: List[Dict],
        rally_log: List[Dict],
        processing_info: Dict
    ):
        """Export complete match data"""

        data = {
            "match_metadata": match_metadata,
            "players": players,
            "final_score": final_score,
            "statistics": statistics,
            "events": events,
            "rally_log": rally_log,
            "processing_info": processing_info
        }

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nMatch data saved to: {output_path}")

    @staticmethod
    def export_minimal(
        output_path: str,
        match_id: str,
        date: str,
        sport: str,
        player_names: List[str],
        winner_id: int,
        final_score: str,
        player_stats: List[Dict]
    ):
        """Export minimal match data (lighter format)"""

        data = {
            "match_id": match_id,
            "date": date,
            "sport": sport,
            "players": player_names,
            "winner": winner_id,
            "final_score": final_score,
            "player_stats": player_stats
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Minimal match data saved to: {output_path}")
```

### 5.3 Updated Main Pipeline

```python
# main.py - score command (simplified)

import cv2
from datetime import datetime
from utils.console_logger import ConsoleMatchLogger
from utils.json_exporter import MatchJSONExporter
from inference.stats_aggregator import StatsAggregator
# ... other imports

def score_match(
    video_path: str,
    output_dir: str = 'results',
    player1_name: str = 'Player 1',
    player2_name: str = 'Player 2',
    sport: str = 'tennis',
    verbose: bool = False
):
    """
    Score match with console output and JSON export

    Args:
        video_path: Input video
        output_dir: Where to save results
        player1_name: Name of player 1
        player2_name: Name of player 2
        sport: 'tennis', 'badminton', or 'pickleball'
        verbose: Enable detailed logging
    """

    # Generate match ID
    match_id = datetime.now().strftime('%Y-%m-%d_%H%M%S')

    # Initialize console logger
    logger = ConsoleMatchLogger([player1_name, player2_name], sport)

    # Initialize stats aggregator
    stats_agg = StatsAggregator()
    stats_agg.start_match({
        'match_id': match_id,
        'players': [0, 1],  # Player IDs
        'sport': sport,
        'match_type': 'singles'
    })

    # Initialize scoring engine
    scorer = TennisScorer(player1_name, player2_name)

    # Initialize models (existing code)
    ball_tracker = tracknet_v5()
    scene_detector = YOLO('yolov8x-pose.pt')
    # ... court detector, event detector, etc.

    # Video I/O
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # ... video writer setup

    # Process video
    event_log = []
    rally_log = []
    frame_id = 0
    start_time = datetime.now()

    while cap.isOpened():
        # ... (existing processing pipeline)
        # Ball tracking, player detection, event detection

        # Process events
        for event in frame_events:
            # Log to console
            logger.log_event(event, frame_id, frame_id / fps)

            # Update stats
            stats_agg.process_event(event)

            # Update scoring
            if event['type'] == 'rally_end':
                scorer.award_point(event['winner'], event['reason'])

                # Log score
                logger.log_score(scorer.get_score_string())

                # Log current stats (every 5 points in verbose mode)
                if verbose and scorer.total_points() % 5 == 0:
                    match_stats = stats_agg.get_match_stats()
                    for player_id in [0, 1]:
                        logger.log_stats(player_id, match_stats[player_id])

                # Save rally
                rally_log.append({
                    'rally_id': len(rally_log),
                    'start_frame': event.get('rally_start_frame'),
                    'end_frame': frame_id,
                    'duration_seconds': (frame_id - event.get('rally_start_frame', frame_id)) / fps,
                    'shot_count': event.get('shot_count', 0),
                    'server': event.get('server'),
                    'winner': event['winner'],
                    'reason': event['reason'],
                    'final_score': scorer.get_score_string()
                })

            # Add to event log
            event_log.append(event)

        frame_id += 1

    # Clean up
    cap.release()
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Get final stats
    final_match_stats = stats_agg.get_match_stats()
    winner_id = scorer.get_winner()
    final_score_string = scorer.get_score_string()

    # Print final summary
    logger.log_final_summary(
        final_score_string,
        winner_id,
        [final_match_stats[0], final_match_stats[1]],
        int(duration)
    )

    # Export JSON
    json_path = f"{output_dir}/{match_id}.json"
    MatchJSONExporter.export_full(
        output_path=json_path,
        match_metadata={
            "match_id": match_id,
            "date": start_time.isoformat(),
            "sport": sport,
            "match_type": "singles",
            "video_path": video_path,
            "duration_seconds": int(duration),
            "processed_at": datetime.now().isoformat()
        },
        players=[
            {"player_id": 0, "name": player1_name, "team": 1},
            {"player_id": 1, "name": player2_name, "team": 2}
        ],
        final_score=scorer.get_score_dict(),
        statistics={
            "player_0": final_match_stats[0],
            "player_1": final_match_stats[1]
        },
        events=event_log,
        rally_log=rally_log,
        processing_info={
            "tracknet_version": "v5",
            "yolo_version": "yolov8x-pose",
            "processing_fps": fps,
            "total_frames": frame_id
        }
    )

    # Also export minimal version (for quick backend integration)
    minimal_path = f"{output_dir}/{match_id}_minimal.json"
    MatchJSONExporter.export_minimal(
        output_path=minimal_path,
        match_id=match_id,
        date=start_time.isoformat(),
        sport=sport,
        player_names=[player1_name, player2_name],
        winner_id=winner_id,
        final_score=final_score_string,
        player_stats=[
            {
                "player_id": 0,
                "points": final_match_stats[0]['points_won'],
                "aces": final_match_stats[0]['aces'],
                "winners": final_match_stats[0]['winners'],
                "errors": final_match_stats[0]['unforced_errors'],
                "first_serve_pct": final_match_stats[0].get('first_serve_pct', 0)
            },
            {
                "player_id": 1,
                "points": final_match_stats[1]['points_won'],
                "aces": final_match_stats[1]['aces'],
                "winners": final_match_stats[1]['winners'],
                "errors": final_match_stats[1]['unforced_errors'],
                "first_serve_pct": final_match_stats[1].get('first_serve_pct', 0)
            }
        ]
    )

    print(f"\n{'═' * 70}")
    print("Processing complete!")
    print(f"Full JSON: {json_path}")
    print(f"Minimal JSON: {minimal_path}")
    print(f"Annotated video: {output_dir}/{match_id}_annotated.mp4")
    print("═" * 70 + "\n")

    return match_id, final_match_stats
```

---

## 6. CLI Usage

### Basic Usage

```bash
# Score a match (simplest)
uv run python main.py score \
  --video tennis_match.mp4 \
  --player1 "Roger Federer" \
  --player2 "Rafael Nadal"

# Output:
# - Console: Real-time score and stats
# - results/2024-04-10_143000.json (full data)
# - results/2024-04-10_143000_minimal.json (light version)
# - results/2024-04-10_143000_annotated.mp4 (video)
```

### With Options

```bash
# Verbose mode (shows stats every 5 points)
uv run python main.py score \
  --video match.mp4 \
  --player1 "Player 1" \
  --player2 "Player 2" \
  --sport badminton \
  --output-dir my_results \
  --verbose

# Quiet mode (minimal console output, just final summary)
uv run python main.py score \
  --video match.mp4 \
  --quiet

# Export only (no video annotation to save time)
uv run python main.py score \
  --video match.mp4 \
  --no-video-output
```

---

## 7. Backend Integration (Later)

When you're ready to connect to your backend:

### Option 1: POST JSON After Processing

```python
# Add to end of score_match()

import requests

def upload_to_backend(json_path: str, backend_url: str):
    """Upload match data to your backend"""

    with open(json_path, 'r') as f:
        match_data = json.load(f)

    response = requests.post(
        f"{backend_url}/api/matches",
        json=match_data,
        headers={"Authorization": f"Bearer {API_TOKEN}"}
    )

    if response.status_code == 200:
        print(f"✓ Match uploaded to backend: {response.json()['match_id']}")
    else:
        print(f"✗ Upload failed: {response.text}")

# Usage
if args.upload_to_backend:
    upload_to_backend(json_path, args.backend_url)
```

### Option 2: Watch Directory

```python
# scripts/watch_and_upload.py

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class MatchUploader(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith('.json') and not event.src_path.endswith('_minimal.json'):
            print(f"New match detected: {event.src_path}")
            upload_to_backend(event.src_path, BACKEND_URL)

observer = Observer()
observer.schedule(MatchUploader(), path='results/', recursive=False)
observer.start()

print("Watching results/ directory for new matches...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
```

Run in background:
```bash
uv run python scripts/watch_and_upload.py &
```

---

## 8. What Gets Removed from Specs

Since we're doing console output first, these components are **deferred**:

- ❌ Database schema (SQLite/PostgreSQL)
- ❌ StatsDatabase class
- ❌ FastAPI REST API
- ❌ Streamlit dashboard
- ❌ Aggregate statistics computation
- ❌ Head-to-head tracking
- ❌ Leaderboards

Your backend will handle all of this!

---

## 9. Implementation Checklist

### Week 1-2: Core Tracking (from multi-object spec)
- [ ] Implement BoT-SORT tracker
- [ ] Implement multi-tracker wrapper
- [ ] Test ball + player detection on sample video

### Week 3: Event Detection (from multi-object spec)
- [ ] Train court detector (or skip if overhead camera)
- [ ] Implement event detector
- [ ] Test bounce/hit detection

### Week 4: Scoring + Stats (this spec)
- [ ] Implement scoring engine (tennis)
- [ ] Implement stats aggregator
- [ ] **Implement console logger** ← NEW
- [ ] **Implement JSON exporter** ← NEW
- [ ] End-to-end test

### Week 5: Polish
- [ ] Improve console output formatting
- [ ] Add progress bar during processing
- [ ] Add error handling
- [ ] Test on multiple matches

### Later: Backend Integration
- [ ] Design API contract with your backend
- [ ] Implement POST to backend
- [ ] Add authentication
- [ ] Handle errors and retries

---

## 10. Example Output Files

Your `results/` directory will look like:

```
results/
  2024-04-10_143000.json              # Full match data (for backend)
  2024-04-10_143000_minimal.json      # Quick summary (for apps)
  2024-04-10_143000_annotated.mp4     # Video with overlays
```

That's it! Clean, simple, and ready to integrate with your backend when you're ready.

---

**End of MVP Guide**
