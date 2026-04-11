# Player Statistics Tracking and Database System — Design Specification

**Date:** 2026-04-10
**Status:** Draft
**Related:** `2026-04-10-multi-object-scoring-system.md`
**Goal:** Track player/pair statistics across matches with persistent storage and analytics dashboard

---

## 1. Problem Statement

Extend the automated scoring system with comprehensive player statistics tracking:

- **Player Profiles:** Manage individual players and doubles pairs
- **Match Statistics:** Track detailed stats per match (aces, winners, errors, shot types, etc.)
- **Aggregate Statistics:** Career stats, win/loss records, rankings
- **Sport-Specific Stats:** Tennis, badminton, pickleball with different metrics
- **Historical Analysis:** Track improvement over time, head-to-head records
- **Dashboard:** Visualize stats with charts and leaderboards

### Use Cases

1. **Personal Training:** Player reviews own stats to identify weaknesses
2. **Coaching:** Coach analyzes player performance trends
3. **League Management:** Track rankings and tournament results
4. **Social Platform:** Players share stats, compare with friends
5. **AI Insights:** "Your backhand winner rate improved 15% this month"

---

## 2. System Architecture

### 2.1 Data Flow

```
Match Video
  ↓
Scoring System (from existing spec)
  ↓
Events: [serve, hit, bounce, rally_end, ...]
  ↓
┌──────────────────────────────────┐
│   Statistics Aggregator          │
│   - Compute match stats          │
│   - Attribute events to players  │
│   - Calculate derived metrics    │
└──────────────────────────────────┘
  ↓
┌──────────────────────────────────┐
│   Player Database                │
│   - Player profiles              │
│   - Match records                │
│   - Aggregate statistics         │
└──────────────────────────────────┘
  ↓
┌──────────────────────────────────┐
│   Analytics Engine               │
│   - Trends over time             │
│   - Head-to-head comparisons     │
│   - AI-powered insights          │
└──────────────────────────────────┘
  ↓
┌──────────────────────────────────┐
│   Dashboard / API                │
│   - Web UI for stats viewing     │
│   - REST API for integrations    │
│   - Export to CSV/PDF            │
└──────────────────────────────────┘
```

### 2.2 Component Overview

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| **Statistics Aggregator** | Extract stats from events, compute metrics | Python (inference/stats.py) |
| **Player Database** | Store players, matches, stats | SQLite / PostgreSQL |
| **Analytics Engine** | Trend analysis, insights | Pandas, NumPy |
| **REST API** | Query stats, update profiles | FastAPI |
| **Dashboard** | Web UI for visualization | React + Recharts (or Streamlit) |

---

## 3. Database Schema

### 3.1 Entity Relationship Diagram

```
┌──────────────┐         ┌──────────────┐
│   Players    │         │    Pairs     │
│──────────────│         │──────────────│
│ player_id PK │◄────┐   │ pair_id PK   │
│ name         │     │   │ player1_id FK│
│ email        │     └───│ player2_id FK│
│ sport        │         │ created_at   │
│ created_at   │         └──────────────┘
└──────────────┘                │
       │                        │
       │         ┌──────────────┴───────────────┐
       │         │                              │
       ▼         ▼                              ▼
┌─────────────────────────────┐      ┌─────────────────────┐
│        Matches              │      │   MatchParticipants │
│─────────────────────────────│      │─────────────────────│
│ match_id PK                 │      │ match_id FK         │
│ sport                       │◄─────│ entity_id FK        │
│ match_type (singles/doubles)│      │ entity_type (player/pair)│
│ date                        │      │ team (1 or 2)       │
│ location                    │      │ result (win/loss)   │
│ video_path                  │      └─────────────────────┘
│ final_score                 │
│ duration_seconds            │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│      MatchStatistics        │
│─────────────────────────────│
│ stat_id PK                  │
│ match_id FK                 │
│ entity_id FK                │
│ entity_type                 │
│ points_won                  │
│ aces                        │
│ double_faults               │
│ winners                     │
│ unforced_errors             │
│ first_serve_pct             │
│ ... (50+ stat fields)       │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│         Events              │
│─────────────────────────────│
│ event_id PK                 │
│ match_id FK                 │
│ frame_number                │
│ timestamp                   │
│ event_type                  │
│ player_id FK (optional)     │
│ data (JSON)                 │
└─────────────────────────────┘
```

### 3.2 SQL Schema

```sql
-- Players table
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    sport VARCHAR(20) NOT NULL,  -- 'tennis', 'badminton', 'pickleball'
    skill_level VARCHAR(20),     -- 'beginner', 'intermediate', 'advanced', 'pro'
    date_of_birth DATE,
    country VARCHAR(3),           -- ISO 3166-1 alpha-3 code
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON                 -- Custom fields
);

-- Doubles pairs
CREATE TABLE pairs (
    pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player1_id INTEGER NOT NULL REFERENCES players(player_id),
    player2_id INTEGER NOT NULL REFERENCES players(player_id),
    pair_name VARCHAR(100),       -- Optional team name
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player1_id, player2_id)
);

-- Matches
CREATE TABLE matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport VARCHAR(20) NOT NULL,
    match_type VARCHAR(20) NOT NULL,  -- 'singles', 'doubles'
    date DATE NOT NULL,
    location VARCHAR(100),
    court_type VARCHAR(20),       -- 'hard', 'clay', 'grass', 'indoor', 'outdoor'
    video_path VARCHAR(500),
    final_score VARCHAR(50),      -- "6-4, 6-3" or "21-18, 21-19"
    duration_seconds INTEGER,
    weather VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

-- Match participants (links players/pairs to matches)
CREATE TABLE match_participants (
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    entity_id INTEGER NOT NULL,   -- player_id or pair_id
    entity_type VARCHAR(10) NOT NULL,  -- 'player' or 'pair'
    team INTEGER NOT NULL,        -- 1 or 2
    result VARCHAR(10) NOT NULL,  -- 'win', 'loss', 'draw'
    PRIMARY KEY (match_id, entity_id, entity_type)
);

-- Match statistics (per player/pair per match)
CREATE TABLE match_statistics (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    entity_id INTEGER NOT NULL,
    entity_type VARCHAR(10) NOT NULL,

    -- Scoring
    points_won INTEGER DEFAULT 0,
    points_lost INTEGER DEFAULT 0,
    games_won INTEGER DEFAULT 0,
    games_lost INTEGER DEFAULT 0,
    sets_won INTEGER DEFAULT 0,
    sets_lost INTEGER DEFAULT 0,

    -- Serve stats
    aces INTEGER DEFAULT 0,
    double_faults INTEGER DEFAULT 0,
    first_serves_in INTEGER DEFAULT 0,
    first_serves_total INTEGER DEFAULT 0,
    second_serves_in INTEGER DEFAULT 0,
    second_serves_total INTEGER DEFAULT 0,
    service_points_won INTEGER DEFAULT 0,
    service_points_total INTEGER DEFAULT 0,

    -- Return stats
    return_points_won INTEGER DEFAULT 0,
    return_points_total INTEGER DEFAULT 0,
    break_points_converted INTEGER DEFAULT 0,
    break_points_total INTEGER DEFAULT 0,

    -- Shot stats
    winners INTEGER DEFAULT 0,
    unforced_errors INTEGER DEFAULT 0,
    forced_errors INTEGER DEFAULT 0,
    forehand_winners INTEGER DEFAULT 0,
    backhand_winners INTEGER DEFAULT 0,
    volley_winners INTEGER DEFAULT 0,
    smash_winners INTEGER DEFAULT 0,

    -- Rally stats
    avg_rally_length REAL DEFAULT 0.0,
    longest_rally INTEGER DEFAULT 0,
    net_approaches INTEGER DEFAULT 0,
    net_points_won INTEGER DEFAULT 0,

    -- Computed metrics (stored for performance)
    first_serve_pct REAL GENERATED ALWAYS AS (
        CASE WHEN first_serves_total > 0
        THEN (first_serves_in * 100.0 / first_serves_total)
        ELSE 0 END
    ) STORED,

    ace_pct REAL GENERATED ALWAYS AS (
        CASE WHEN service_points_total > 0
        THEN (aces * 100.0 / service_points_total)
        ELSE 0 END
    ) STORED,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Events (detailed frame-by-frame)
CREATE TABLE events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(match_id),
    frame_number INTEGER NOT NULL,
    timestamp REAL NOT NULL,      -- Seconds from start
    event_type VARCHAR(50) NOT NULL,  -- 'serve', 'hit', 'bounce', 'rally_end', etc.
    player_id INTEGER REFERENCES players(player_id),
    data JSON NOT NULL,           -- Event-specific data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_match_frame (match_id, frame_number)
);

-- Aggregate statistics (career totals)
CREATE TABLE aggregate_statistics (
    entity_id INTEGER NOT NULL,
    entity_type VARCHAR(10) NOT NULL,
    sport VARCHAR(20) NOT NULL,
    time_period VARCHAR(20) NOT NULL,  -- 'all_time', 'year_2024', 'month_2024_04', etc.

    -- Summary
    matches_played INTEGER DEFAULT 0,
    matches_won INTEGER DEFAULT 0,
    matches_lost INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0.0,

    -- Aggregated match stats (sums)
    total_points_won INTEGER DEFAULT 0,
    total_aces INTEGER DEFAULT 0,
    total_winners INTEGER DEFAULT 0,
    total_unforced_errors INTEGER DEFAULT 0,

    -- Averages
    avg_first_serve_pct REAL DEFAULT 0.0,
    avg_ace_pct REAL DEFAULT 0.0,
    avg_winner_rate REAL DEFAULT 0.0,

    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_id, entity_type, sport, time_period)
);

-- Head-to-head records
CREATE TABLE head_to_head (
    entity1_id INTEGER NOT NULL,
    entity1_type VARCHAR(10) NOT NULL,
    entity2_id INTEGER NOT NULL,
    entity2_type VARCHAR(10) NOT NULL,
    sport VARCHAR(20) NOT NULL,

    entity1_wins INTEGER DEFAULT 0,
    entity2_wins INTEGER DEFAULT 0,
    total_matches INTEGER DEFAULT 0,
    last_match_date DATE,
    last_match_id INTEGER REFERENCES matches(match_id),

    PRIMARY KEY (entity1_id, entity1_type, entity2_id, entity2_type, sport)
);
```

---

## 4. Statistics Definitions

### 4.1 Sport-Specific Stats

#### Tennis Statistics

| Category | Stat | Formula |
|----------|------|---------|
| **Serve** | First Serve % | `first_serves_in / first_serves_total * 100` |
| | Ace % | `aces / service_points_total * 100` |
| | Double Fault % | `double_faults / service_points_total * 100` |
| | Service Points Won % | `service_points_won / service_points_total * 100` |
| **Return** | Return Points Won % | `return_points_won / return_points_total * 100` |
| | Break Point Conversion % | `break_points_converted / break_points_total * 100` |
| **Shots** | Winner Rate | `winners / total_shots * 100` |
| | Unforced Error Rate | `unforced_errors / total_shots * 100` |
| | Forehand Winner % | `forehand_winners / total_winners * 100` |
| **Rally** | Avg Rally Length | `sum(rally_lengths) / total_rallies` |
| | Net Success % | `net_points_won / net_approaches * 100` |

#### Badminton Statistics

| Category | Stat | Formula |
|----------|------|---------|
| **Serve** | Short Serve % | `short_serves / total_serves * 100` |
| | Long Serve % | `long_serves / total_serves * 100` |
| | Service Aces | Count of unreturned serves |
| **Shots** | Smash % | `smashes / total_shots * 100` |
| | Drop Shot % | `drop_shots / total_shots * 100` |
| | Clear % | `clears / total_shots * 100` |
| | Net Kill % | `net_kills / total_shots * 100` |
| **Rally** | Avg Rally Length | `sum(shot_counts) / total_rallies` |
| | Longest Rally | `max(shot_counts)` |

#### Pickleball Statistics

| Category | Stat | Formula |
|----------|------|---------|
| **Serve** | Ace % | `aces / total_serves * 100` |
| | Fault % | `faults / total_serves * 100` |
| **Kitchen** | Dink Success % | `successful_dinks / total_dinks * 100` |
| | Kitchen Violation % | `kitchen_violations / net_approaches * 100` |
| **Shots** | Third Shot Drop % | `third_shot_drops / third_shots * 100` |
| | Volley Winner % | `volley_winners / total_volleys * 100` |

### 4.2 Derived Metrics

**Momentum Score:**
```python
momentum = (recent_5_points_won / 5) * 100  # Percentage of last 5 points won
```

**Clutch Performance:**
```python
clutch_score = (break_points_won / break_points_faced) * 100
```

**Consistency Score:**
```python
consistency = (winners - unforced_errors) / total_shots * 100
```

**Aggression Index:**
```python
aggression = (winners + forced_errors) / (winners + all_errors) * 100
```

---

## 5. Statistics Aggregator Implementation

### 5.1 Event Processing

```python
# inference/stats_aggregator.py

from typing import Dict, List
import json

class StatsAggregator:
    """
    Extract statistics from event stream
    """

    def __init__(self):
        self.current_match = None
        self.player_stats = {}
        self.rally_state = {
            'server': None,
            'receiver': None,
            'shot_count': 0,
            'ball_positions': []
        }

    def start_match(self, match_info: Dict):
        """
        Initialize match tracking

        Args:
            match_info: {
                'match_id': int,
                'players': [player1_id, player2_id],
                'sport': 'tennis',
                'match_type': 'singles'
            }
        """
        self.current_match = match_info

        # Initialize stats for each player
        for player_id in match_info['players']:
            self.player_stats[player_id] = self._init_player_stats()

    def process_event(self, event: Dict):
        """
        Process single event and update stats

        Args:
            event: {
                'type': 'serve' | 'hit' | 'bounce' | 'rally_end',
                'player_id': int,
                'data': {...}
            }
        """
        event_type = event['type']
        player_id = event.get('player_id')

        # Route to appropriate handler
        if event_type == 'serve_start':
            self._handle_serve_start(player_id, event)
        elif event_type == 'serve_in':
            self._handle_serve_in(player_id, event)
        elif event_type == 'fault':
            self._handle_fault(player_id, event)
        elif event_type == 'hit':
            self._handle_hit(player_id, event)
        elif event_type == 'bounce':
            self._handle_bounce(event)
        elif event_type == 'rally_end':
            self._handle_rally_end(event)

    def _handle_serve_start(self, player_id: int, event: Dict):
        """Track serve attempt"""
        self.rally_state['server'] = player_id
        self.rally_state['receiver'] = self._get_opponent(player_id)
        self.rally_state['shot_count'] = 0

        # Check if first or second serve
        if event['data'].get('is_first_serve'):
            self.player_stats[player_id]['first_serves_total'] += 1
        else:
            self.player_stats[player_id]['second_serves_total'] += 1

    def _handle_serve_in(self, player_id: int, event: Dict):
        """Track successful serve"""
        if event['data'].get('is_first_serve'):
            self.player_stats[player_id]['first_serves_in'] += 1
        else:
            self.player_stats[player_id]['second_serves_in'] += 1

        # Check if ace (unreturned serve)
        if event['data'].get('is_ace'):
            self.player_stats[player_id]['aces'] += 1

    def _handle_fault(self, player_id: int, event: Dict):
        """Track fault"""
        fault_type = event['data'].get('fault_type')

        if fault_type == 'double_fault':
            self.player_stats[player_id]['double_faults'] += 1

    def _handle_hit(self, player_id: int, event: Dict):
        """Track shot"""
        self.rally_state['shot_count'] += 1
        shot_type = event['data'].get('shot_type')

        # Count shot types
        if shot_type == 'forehand':
            self.player_stats[player_id]['forehand_count'] += 1
        elif shot_type == 'backhand':
            self.player_stats[player_id]['backhand_count'] += 1
        elif shot_type == 'volley':
            self.player_stats[player_id]['volley_count'] += 1
        elif shot_type == 'smash':
            self.player_stats[player_id]['smash_count'] += 1

    def _handle_rally_end(self, event: Dict):
        """Track rally outcome"""
        winner_id = event['winner']
        loser_id = self._get_opponent(winner_id)
        reason = event['reason']

        # Update points
        self.player_stats[winner_id]['points_won'] += 1
        self.player_stats[loser_id]['points_lost'] += 1

        # Update rally stats
        rally_length = self.rally_state['shot_count']
        self.player_stats[winner_id]['total_rally_length'] += rally_length
        self.player_stats[winner_id]['total_rallies'] += 1

        if rally_length > self.player_stats[winner_id]['longest_rally']:
            self.player_stats[winner_id]['longest_rally'] = rally_length

        # Categorize outcome
        if reason == 'ace':
            # Already counted in _handle_serve_in
            pass
        elif reason == 'winner':
            self.player_stats[winner_id]['winners'] += 1
            # Determine shot type of winner
            last_shot = event['data'].get('last_shot_type')
            if last_shot == 'forehand':
                self.player_stats[winner_id]['forehand_winners'] += 1
            elif last_shot == 'backhand':
                self.player_stats[winner_id]['backhand_winners'] += 1
            elif last_shot == 'volley':
                self.player_stats[winner_id]['volley_winners'] += 1
            elif last_shot == 'smash':
                self.player_stats[winner_id]['smash_winners'] += 1
        elif reason == 'out' or reason == 'net':
            self.player_stats[loser_id]['unforced_errors'] += 1
        elif reason == 'forced_error':
            self.player_stats[loser_id]['forced_errors'] += 1

        # Service point tracking
        server = self.rally_state['server']
        if winner_id == server:
            self.player_stats[server]['service_points_won'] += 1
        self.player_stats[server]['service_points_total'] += 1

        # Return point tracking
        receiver = self.rally_state['receiver']
        if winner_id == receiver:
            self.player_stats[receiver]['return_points_won'] += 1
        self.player_stats[receiver]['return_points_total'] += 1

        # Reset rally state
        self.rally_state = {
            'server': None,
            'receiver': None,
            'shot_count': 0,
            'ball_positions': []
        }

    def get_match_stats(self) -> Dict:
        """
        Return final match statistics

        Returns:
            {
                player1_id: {...stats...},
                player2_id: {...stats...}
            }
        """
        # Compute derived metrics
        for player_id, stats in self.player_stats.items():
            stats['avg_rally_length'] = (
                stats['total_rally_length'] / stats['total_rallies']
                if stats['total_rallies'] > 0 else 0
            )

            stats['first_serve_pct'] = (
                stats['first_serves_in'] / stats['first_serves_total'] * 100
                if stats['first_serves_total'] > 0 else 0
            )

            stats['ace_pct'] = (
                stats['aces'] / stats['service_points_total'] * 100
                if stats['service_points_total'] > 0 else 0
            )

            # ... compute other derived metrics

        return self.player_stats

    def _init_player_stats(self) -> Dict:
        """Initialize empty stats dict"""
        return {
            'points_won': 0,
            'points_lost': 0,
            'games_won': 0,
            'games_lost': 0,
            'sets_won': 0,
            'sets_lost': 0,
            'aces': 0,
            'double_faults': 0,
            'first_serves_in': 0,
            'first_serves_total': 0,
            'second_serves_in': 0,
            'second_serves_total': 0,
            'service_points_won': 0,
            'service_points_total': 0,
            'return_points_won': 0,
            'return_points_total': 0,
            'winners': 0,
            'unforced_errors': 0,
            'forced_errors': 0,
            'forehand_count': 0,
            'backhand_count': 0,
            'volley_count': 0,
            'smash_count': 0,
            'forehand_winners': 0,
            'backhand_winners': 0,
            'volley_winners': 0,
            'smash_winners': 0,
            'total_rally_length': 0,
            'total_rallies': 0,
            'longest_rally': 0
        }

    def _get_opponent(self, player_id: int) -> int:
        """Get opponent player ID"""
        players = self.current_match['players']
        return players[0] if player_id == players[1] else players[1]
```

### 5.2 Database Integration

```python
# database/stats_db.py

import sqlite3
from typing import Dict, List, Optional
from datetime import datetime
import json

class StatsDatabase:
    """Manage player statistics database"""

    def __init__(self, db_path='data/stats.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create database schema"""
        with open('database/schema.sql', 'r') as f:
            schema = f.read()
        self.conn.executescript(schema)
        self.conn.commit()

    def create_player(self, name: str, email: str, sport: str, **kwargs) -> int:
        """
        Create new player profile

        Returns:
            player_id
        """
        cursor = self.conn.execute('''
            INSERT INTO players (name, email, sport, skill_level, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, email, sport, kwargs.get('skill_level'), json.dumps(kwargs)))

        self.conn.commit()
        return cursor.lastrowid

    def create_match(self, sport: str, match_type: str, date: str,
                    participants: List[Dict], **kwargs) -> int:
        """
        Create match record

        Args:
            participants: [
                {'entity_id': player_id, 'entity_type': 'player', 'team': 1},
                {'entity_id': player_id, 'entity_type': 'player', 'team': 2}
            ]

        Returns:
            match_id
        """
        # Insert match
        cursor = self.conn.execute('''
            INSERT INTO matches (sport, match_type, date, location, video_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (sport, match_type, date, kwargs.get('location'),
              kwargs.get('video_path'), json.dumps(kwargs)))

        match_id = cursor.lastrowid

        # Insert participants
        for participant in participants:
            self.conn.execute('''
                INSERT INTO match_participants (match_id, entity_id, entity_type, team, result)
                VALUES (?, ?, ?, ?, ?)
            ''', (match_id, participant['entity_id'], participant['entity_type'],
                  participant['team'], 'pending'))  # Result updated later

        self.conn.commit()
        return match_id

    def save_match_stats(self, match_id: int, entity_id: int,
                        entity_type: str, stats: Dict):
        """Save match statistics for player/pair"""

        # Build INSERT query dynamically from stats dict
        columns = list(stats.keys())
        placeholders = ', '.join(['?'] * len(columns))
        column_str = ', '.join(columns)

        query = f'''
            INSERT INTO match_statistics
            (match_id, entity_id, entity_type, {column_str})
            VALUES (?, ?, ?, {placeholders})
        '''

        values = [match_id, entity_id, entity_type] + [stats[col] for col in columns]
        self.conn.execute(query, values)
        self.conn.commit()

    def update_match_result(self, match_id: int, winner_entity_id: int,
                           winner_entity_type: str, final_score: str):
        """Update match result"""

        # Update winner
        self.conn.execute('''
            UPDATE match_participants
            SET result = 'win'
            WHERE match_id = ? AND entity_id = ? AND entity_type = ?
        ''', (match_id, winner_entity_id, winner_entity_type))

        # Update loser
        self.conn.execute('''
            UPDATE match_participants
            SET result = 'loss'
            WHERE match_id = ? AND NOT (entity_id = ? AND entity_type = ?)
        ''', (match_id, winner_entity_id, winner_entity_type))

        # Update match
        self.conn.execute('''
            UPDATE matches
            SET final_score = ?
            WHERE match_id = ?
        ''', (final_score, match_id))

        self.conn.commit()

        # Update aggregate stats
        self._update_aggregates(match_id)

    def _update_aggregates(self, match_id: int):
        """Recompute aggregate statistics after match"""

        # Get match participants
        participants = self.conn.execute('''
            SELECT entity_id, entity_type, result
            FROM match_participants
            WHERE match_id = ?
        ''', (match_id,)).fetchall()

        for participant in participants:
            entity_id = participant['entity_id']
            entity_type = participant['entity_type']

            # Get all matches for this entity
            all_stats = self.conn.execute('''
                SELECT ms.*
                FROM match_statistics ms
                JOIN match_participants mp ON ms.match_id = mp.match_id
                WHERE mp.entity_id = ? AND mp.entity_type = ?
            ''', (entity_id, entity_type)).fetchall()

            # Aggregate
            totals = self._aggregate_stats(all_stats)

            # Get sport from first match
            sport = self.conn.execute('''
                SELECT sport FROM matches
                JOIN match_statistics ON matches.match_id = match_statistics.match_id
                WHERE entity_id = ? AND entity_type = ?
                LIMIT 1
            ''', (entity_id, entity_type)).fetchone()['sport']

            # Upsert aggregate stats
            self.conn.execute('''
                INSERT INTO aggregate_statistics
                (entity_id, entity_type, sport, time_period,
                 matches_played, matches_won, total_points_won, total_aces, ...)
                VALUES (?, ?, ?, 'all_time', ?, ?, ?, ?, ...)
                ON CONFLICT (entity_id, entity_type, sport, time_period)
                DO UPDATE SET
                    matches_played = excluded.matches_played,
                    matches_won = excluded.matches_won,
                    ...
            ''', (entity_id, entity_type, sport, totals['matches_played'],
                  totals['matches_won'], ...))

        self.conn.commit()

    def get_player_stats(self, player_id: int, sport: str,
                        time_period='all_time') -> Dict:
        """Get aggregate statistics for player"""

        row = self.conn.execute('''
            SELECT * FROM aggregate_statistics
            WHERE entity_id = ? AND entity_type = 'player'
              AND sport = ? AND time_period = ?
        ''', (player_id, sport, time_period)).fetchone()

        if row:
            return dict(row)
        else:
            return {}

    def get_head_to_head(self, player1_id: int, player2_id: int, sport: str) -> Dict:
        """Get head-to-head record between two players"""

        row = self.conn.execute('''
            SELECT * FROM head_to_head
            WHERE ((entity1_id = ? AND entity2_id = ?) OR
                   (entity1_id = ? AND entity2_id = ?))
              AND entity1_type = 'player' AND entity2_type = 'player'
              AND sport = ?
        ''', (player1_id, player2_id, player2_id, player1_id, sport)).fetchone()

        if row:
            return dict(row)
        else:
            return {
                'entity1_wins': 0,
                'entity2_wins': 0,
                'total_matches': 0
            }

    def get_leaderboard(self, sport: str, metric='win_rate', limit=10) -> List[Dict]:
        """
        Get leaderboard for sport

        Args:
            metric: 'win_rate', 'total_aces', 'avg_first_serve_pct', etc.
        """
        rows = self.conn.execute(f'''
            SELECT p.player_id, p.name, a.*
            FROM aggregate_statistics a
            JOIN players p ON a.entity_id = p.player_id
            WHERE a.sport = ? AND a.entity_type = 'player'
              AND a.time_period = 'all_time'
            ORDER BY a.{metric} DESC
            LIMIT ?
        ''', (sport, limit)).fetchall()

        return [dict(row) for row in rows]
```

---

## 6. REST API

### 6.1 API Endpoints

```python
# api/stats_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from database.stats_db import StatsDatabase

app = FastAPI(title="TrackNet Stats API")
db = StatsDatabase()

# === MODELS ===

class PlayerCreate(BaseModel):
    name: str
    email: str
    sport: str
    skill_level: Optional[str] = None

class MatchCreate(BaseModel):
    sport: str
    match_type: str
    date: str
    location: Optional[str] = None
    participants: List[dict]

class StatsQuery(BaseModel):
    player_id: int
    sport: str
    time_period: str = 'all_time'

# === ENDPOINTS ===

@app.post("/players", response_model=dict)
def create_player(player: PlayerCreate):
    """Create new player profile"""
    player_id = db.create_player(**player.dict())
    return {"player_id": player_id, "message": "Player created successfully"}

@app.get("/players/{player_id}")
def get_player(player_id: int):
    """Get player profile"""
    player = db.conn.execute(
        'SELECT * FROM players WHERE player_id = ?', (player_id,)
    ).fetchone()

    if not player:
        raise HTTPException(status_code=404, detail="Player not found")

    return dict(player)

@app.get("/players/{player_id}/stats")
def get_player_stats(player_id: int, sport: str, time_period: str = 'all_time'):
    """Get player statistics"""
    stats = db.get_player_stats(player_id, sport, time_period)

    if not stats:
        raise HTTPException(status_code=404, detail="No stats found")

    return stats

@app.get("/players/{player_id}/matches")
def get_player_matches(player_id: int, sport: Optional[str] = None, limit: int = 20):
    """Get player's match history"""
    query = '''
        SELECT m.*, mp.result
        FROM matches m
        JOIN match_participants mp ON m.match_id = mp.match_id
        WHERE mp.entity_id = ? AND mp.entity_type = 'player'
    '''
    params = [player_id]

    if sport:
        query += ' AND m.sport = ?'
        params.append(sport)

    query += ' ORDER BY m.date DESC LIMIT ?'
    params.append(limit)

    rows = db.conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]

@app.get("/head-to-head")
def get_head_to_head(player1_id: int, player2_id: int, sport: str):
    """Get head-to-head record"""
    h2h = db.get_head_to_head(player1_id, player2_id, sport)
    return h2h

@app.get("/leaderboard/{sport}")
def get_leaderboard(sport: str, metric: str = 'win_rate', limit: int = 10):
    """Get leaderboard"""
    leaderboard = db.get_leaderboard(sport, metric, limit)
    return leaderboard

@app.post("/matches")
def create_match(match: MatchCreate):
    """Create new match record"""
    match_id = db.create_match(**match.dict())
    return {"match_id": match_id, "message": "Match created successfully"}

@app.post("/matches/{match_id}/stats")
def save_match_stats(match_id: int, entity_id: int, entity_type: str, stats: dict):
    """Save match statistics"""
    db.save_match_stats(match_id, entity_id, entity_type, stats)
    return {"message": "Stats saved successfully"}

@app.post("/matches/{match_id}/result")
def update_match_result(match_id: int, winner_entity_id: int,
                       winner_entity_type: str, final_score: str):
    """Update match result and final score"""
    db.update_match_result(match_id, winner_entity_id, winner_entity_type, final_score)
    return {"message": "Match result updated successfully"}

# Run with: uvicorn api.stats_api:app --reload
```

### 6.2 API Usage Examples

```bash
# Create player
curl -X POST http://localhost:8000/players \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Roger Federer",
    "email": "roger@example.com",
    "sport": "tennis",
    "skill_level": "pro"
  }'

# Get player stats
curl http://localhost:8000/players/1/stats?sport=tennis&time_period=all_time

# Get leaderboard
curl http://localhost:8000/leaderboard/tennis?metric=win_rate&limit=10

# Get head-to-head
curl http://localhost:8000/head-to-head?player1_id=1&player2_id=2&sport=tennis
```

---

## 7. Dashboard (Optional)

### 7.1 Streamlit Dashboard

```python
# dashboard/app.py

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_BASE = "http://localhost:8000"

st.title("TrackNet Player Statistics Dashboard")

# Sidebar: Player selection
player_id = st.sidebar.number_input("Player ID", min_value=1, value=1)
sport = st.sidebar.selectbox("Sport", ["tennis", "badminton", "pickleball"])

# Fetch player info
response = requests.get(f"{API_BASE}/players/{player_id}")
if response.status_code == 200:
    player = response.json()
    st.header(f"{player['name']} - {sport.title()} Stats")
else:
    st.error("Player not found")
    st.stop()

# Fetch stats
stats_response = requests.get(
    f"{API_BASE}/players/{player_id}/stats",
    params={"sport": sport, "time_period": "all_time"}
)

if stats_response.status_code == 200:
    stats = stats_response.json()

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matches Played", stats['matches_played'])
    col2.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    col3.metric("Total Aces", stats['total_aces'])
    col4.metric("Winner Rate", f"{stats['avg_winner_rate']:.1f}%")

    # Serve stats
    st.subheader("Serve Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("First Serve %", f"{stats['avg_first_serve_pct']:.1f}%")
    col2.metric("Ace %", f"{stats['avg_ace_pct']:.1f}%")
    col3.metric("Double Fault %", f"{stats.get('avg_double_fault_pct', 0):.1f}%")

    # Match history
    st.subheader("Recent Matches")
    matches_response = requests.get(
        f"{API_BASE}/players/{player_id}/matches",
        params={"sport": sport, "limit": 10}
    )

    if matches_response.status_code == 200:
        matches = pd.DataFrame(matches_response.json())
        st.dataframe(matches[['date', 'final_score', 'result', 'location']])

    # Win rate trend (placeholder - would need time-series data)
    st.subheader("Performance Trend")
    # This would fetch monthly stats and plot trend

else:
    st.info("No statistics available yet. Play some matches!")

# Leaderboard
st.subheader(f"{sport.title()} Leaderboard")
leaderboard_response = requests.get(
    f"{API_BASE}/leaderboard/{sport}",
    params={"metric": "win_rate", "limit": 10}
)

if leaderboard_response.status_code == 200:
    leaderboard = pd.DataFrame(leaderboard_response.json())
    st.dataframe(leaderboard[['name', 'matches_played', 'matches_won', 'win_rate']])
```

Run with: `streamlit run dashboard/app.py`

---

## 8. Integration with Scoring System

### 8.1 End-to-End Workflow

```python
# main.py - Extended score command

from database.stats_db import StatsDatabase
from inference.stats_aggregator import StatsAggregator

def score_match_with_stats(video_path, player1_id, player2_id, sport='tennis', **kwargs):
    """
    Score match and save statistics to database

    Args:
        video_path: Video file
        player1_id: Database player ID for player 1
        player2_id: Database player ID for player 2
        sport: 'tennis', 'badminton', 'pickleball'
    """

    # Initialize database
    db = StatsDatabase()

    # Create match record
    match_id = db.create_match(
        sport=sport,
        match_type='singles',
        date=datetime.now().strftime('%Y-%m-%d'),
        participants=[
            {'entity_id': player1_id, 'entity_type': 'player', 'team': 1},
            {'entity_id': player2_id, 'entity_type': 'player', 'team': 2}
        ],
        video_path=video_path,
        **kwargs
    )

    # Initialize stats aggregator
    stats_agg = StatsAggregator()
    stats_agg.start_match({
        'match_id': match_id,
        'players': [player1_id, player2_id],
        'sport': sport,
        'match_type': 'singles'
    })

    # === RUN SCORING PIPELINE (from existing spec) ===
    # (TrackNet + YOLO + Event Detection + Scoring)

    # ... (existing scoring pipeline code) ...

    # Process each event through stats aggregator
    for event in event_log:
        stats_agg.process_event(event)

        # Save event to database
        db.conn.execute('''
            INSERT INTO events (match_id, frame_number, timestamp, event_type, player_id, data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (match_id, event['frame'], event.get('timestamp', 0),
              event['type'], event.get('player_id'), json.dumps(event)))

    # Get final match stats
    match_stats = stats_agg.get_match_stats()

    # Save to database
    for player_id, stats in match_stats.items():
        db.save_match_stats(match_id, player_id, 'player', stats)

    # Update match result
    winner_id = scorer.get_winner()  # From scoring engine
    final_score = scorer.get_score_string()
    db.update_match_result(match_id, winner_id, 'player', final_score)

    # Print summary
    print(f"\n=== MATCH COMPLETE ===")
    print(f"Match ID: {match_id}")
    print(f"Final Score: {final_score}")
    print(f"Winner: Player {winner_id}")
    print(f"\nStatistics saved to database")
    print(f"View stats: http://localhost:8501 (Streamlit dashboard)")

    return match_id, match_stats
```

### 8.2 CLI Usage

```bash
# Score match and save stats
uv run python main.py score \
  --video tennis_match.mp4 \
  --player1-id 1 \
  --player2-id 2 \
  --sport tennis \
  --location "Wimbledon Centre Court" \
  --save-stats

# View player stats
uv run python main.py stats --player-id 1 --sport tennis

# Export stats to CSV
uv run python main.py export-stats --player-id 1 --output player1_stats.csv
```

---

## 9. Future Enhancements

### Advanced Analytics
- **Shot placement heatmaps:** Visualize where players hit the ball
- **Movement tracking:** Court coverage analysis from player bounding boxes
- **Serve speed estimation:** From ball trajectory
- **Spin detection:** From high-FPS footage
- **Fatigue analysis:** Performance degradation over match duration

### AI Insights
- **Weakness detection:** "Your backhand error rate is 15% above average"
- **Improvement suggestions:** "Focus on first serve percentage"
- **Match predictions:** Based on head-to-head and form
- **Training recommendations:** Personalized drills

### Social Features
- **Player profiles:** Public profiles with stats
- **Challenges:** Players challenge each other
- **Tournaments:** League management
- **Live streaming:** Real-time stats overlay

---

## 10. Project Structure Updates

```
playground-track-net/
  database/
    __init__.py
    schema.sql               # NEW: Database schema
    stats_db.py              # NEW: Database interface

  inference/
    stats_aggregator.py      # NEW: Extract stats from events

  api/
    __init__.py
    stats_api.py             # NEW: FastAPI REST API

  dashboard/
    app.py                   # NEW: Streamlit dashboard
    requirements.txt         # NEW: Dashboard dependencies

  scripts/
    init_database.py         # NEW: Initialize database
    export_stats.py          # NEW: Export to CSV/PDF

  data/
    stats.db                 # SQLite database (gitignored)

  main.py                    # EXTEND: Add --save-stats flag
```

---

**End of Specification**
