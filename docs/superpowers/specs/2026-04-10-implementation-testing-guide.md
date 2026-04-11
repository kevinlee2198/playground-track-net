# Implementation and Testing Guide: How to Build This System

**Date:** 2026-04-10
**Status:** Practical Guide
**Goal:** Step-by-step approach to implement and test the multi-object scoring system

---

## 1. Implementation Strategy: Incremental Development

### The Right Approach ✅

**Build in vertical slices, test each piece independently, then integrate.**

```
Week 1: Stub system (end-to-end skeleton)
  ↓
Week 2: Real ball tracking
  ↓
Week 3: Real player tracking
  ↓
Week 4: Real event detection
  ↓
Week 5: Real stats + output
```

**Not this:** Build all models first, integrate at the end ❌

### Why This Works

1. **See progress immediately** - Working demo after Week 1
2. **Test continuously** - Catch bugs early
3. **Easy to debug** - Only one new component at a time
4. **Flexible** - Can ship earlier if needed

---

## 2. Week-by-Week Implementation Plan

### Week 1: End-to-End Stub (Skeleton)

**Goal:** Get the full pipeline working with fake data

#### What to Build

```python
# main.py - Stub version

def score_match_stub(video_path):
    """Stub: Fake everything, test pipeline flow"""

    # Fake video reading
    print("Reading video...")
    num_frames = 1000

    # Fake ball tracking
    print("Tracking ball...")
    ball_positions = [(100 + i, 200 + i*0.5) for i in range(num_frames)]

    # Fake player tracking
    print("Tracking players...")
    player_tracks = {
        'player_0': [(50, 100), (51, 101), ...],
        'player_1': [(400, 100), (401, 101), ...]
    }

    # Fake event detection
    print("Detecting events...")
    events = [
        {'frame': 10, 'type': 'serve_start', 'player_id': 0},
        {'frame': 25, 'type': 'bounce', 'valid': True},
        {'frame': 30, 'type': 'rally_end', 'winner': 0, 'reason': 'winner'}
    ]

    # Real console logger (implement this!)
    logger = ConsoleMatchLogger(['Player 1', 'Player 2'], 'tennis')
    for event in events:
        logger.log_event(event, event['frame'], event['frame'] / 30.0)

    # Real JSON exporter (implement this!)
    exporter = MatchJSONExporter()
    exporter.export_minimal(...)

    print("✓ Stub pipeline complete!")

# Run it
score_match_stub('fake_video.mp4')
```

#### Tests to Write

```python
# tests/test_console_logger.py

def test_console_logger_outputs_without_error():
    """Test logger doesn't crash"""
    logger = ConsoleMatchLogger(['P1', 'P2'], 'tennis')

    event = {'type': 'serve_start', 'player_id': 0}
    logger.log_event(event, 100, 3.33)  # Should not crash

    assert True  # If we get here, it worked


def test_json_exporter_creates_file():
    """Test JSON export works"""
    exporter = MatchJSONExporter()

    exporter.export_minimal(
        output_path='test_output.json',
        match_id='test_001',
        date='2024-01-01',
        sport='tennis',
        player_names=['P1', 'P2'],
        winner_id=0,
        final_score='6-0',
        player_stats=[{'points': 24}, {'points': 0}]
    )

    assert os.path.exists('test_output.json')

    with open('test_output.json') as f:
        data = json.load(f)
        assert data['winner'] == 0
```

**Deliverable:**
- Console logger works ✓
- JSON exporter works ✓
- Can run end-to-end with fake data ✓

---

### Week 2: Real Ball Tracking

**Goal:** Replace fake ball tracking with TrackNet

#### What to Build

```python
# models/ball_tracker_wrapper.py

class BallTrackerWrapper:
    """Wrapper around TrackNet for easy testing"""

    def __init__(self, model_path=None):
        if model_path:
            self.model = tracknet_v5()
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval().cuda()
        else:
            self.model = None  # Stub mode

    def track_ball(self, frames):
        """
        Args:
            frames: List of 3 consecutive frames [np.array, ...]
        Returns:
            ball_position: (x, y) or None
        """
        if self.model is None:
            # Stub mode (for testing without model)
            return (100, 200)

        # Real tracking
        frames_tensor = self._preprocess(frames)
        with torch.no_grad():
            heatmaps = self.model(frames_tensor)

        ball_pos = self._heatmap_to_position(heatmaps[0, 1])  # Middle frame
        return ball_pos

    def _preprocess(self, frames):
        # Resize, normalize, stack
        ...

    def _heatmap_to_position(self, heatmap):
        # Threshold, connected components, centroid
        ...
```

#### Tests

```python
# tests/test_ball_tracker.py

def test_ball_tracker_stub_mode():
    """Test tracker works in stub mode (no model)"""
    tracker = BallTrackerWrapper(model_path=None)

    fake_frames = [np.zeros((288, 512, 3)) for _ in range(3)]
    ball_pos = tracker.track_ball(fake_frames)

    assert ball_pos is not None
    assert len(ball_pos) == 2  # (x, y)


@pytest.mark.slow
def test_ball_tracker_with_model():
    """Test tracker with real model (requires checkpoint)"""
    if not os.path.exists('checkpoints/tracknet_v5.pt'):
        pytest.skip("Model checkpoint not found")

    tracker = BallTrackerWrapper('checkpoints/tracknet_v5.pt')

    # Use synthetic test frames (see Section 3)
    frames = generate_synthetic_frames_with_ball(ball_position=(100, 200))
    ball_pos = tracker.track_ball(frames)

    # Should detect ball near (100, 200)
    assert ball_pos is not None
    assert abs(ball_pos[0] - 100) < 20
    assert abs(ball_pos[1] - 200) < 20


def test_ball_tracker_integration():
    """Test ball tracker in pipeline"""
    tracker = BallTrackerWrapper(model_path=None)  # Stub

    # Replace stub in main.py
    ball_positions = []
    for frame_id in range(100):
        frames = get_three_frames(frame_id)  # Stub
        ball_pos = tracker.track_ball(frames)
        ball_positions.append(ball_pos)

    assert len(ball_positions) == 100
```

**Deliverable:**
- Ball tracker works in isolation ✓
- Integrated into pipeline ✓
- Can toggle stub/real mode ✓

---

### Week 3: Real Player Tracking

**Goal:** Add YOLOv8-pose + BoT-SORT

#### What to Build (Step by Step)

**Step 3.1: YOLO Detection Only**

```python
# models/player_detector.py

class PlayerDetector:
    """YOLO person detection"""

    def __init__(self, stub=False):
        self.stub = stub
        if not stub:
            from ultralytics import YOLO
            self.model = YOLO('yolov8x-pose.pt')

    def detect(self, frame):
        """Returns: List of [x1, y1, x2, y2, confidence]"""
        if self.stub:
            # Return 2 fake players
            return np.array([
                [50, 100, 150, 400, 0.95],   # Player 1
                [400, 100, 500, 400, 0.93]   # Player 2
            ])

        results = self.model(frame, verbose=False)
        persons = results[0].boxes[results[0].boxes.cls == 0]

        # Take top 2 by confidence
        if len(persons) > 2:
            top_indices = persons.conf.argsort(descending=True)[:2]
            persons = persons[top_indices]

        return persons.xyxy.cpu().numpy()
```

**Test:**
```python
def test_player_detector_stub():
    detector = PlayerDetector(stub=True)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert len(detections) == 2  # 2 players
    assert detections.shape[1] == 5  # x1, y1, x2, y2, conf
```

**Step 3.2: Add Tracking (BoT-SORT)**

```python
# models/trackers/bot_sort.py (simplified)

class SimpleTracker:
    """Simplified tracker using IoU matching"""

    def __init__(self):
        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        """
        Args:
            detections: [N, 4] boxes (x1, y1, x2, y2)
        Returns:
            tracks: [M, 5] (x1, y1, x2, y2, track_id)
        """
        if len(self.tracks) == 0:
            # Initialize tracks
            for det in detections:
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': det[:4],
                    'age': 0
                })
                self.next_id += 1
        else:
            # Match detections to existing tracks (IoU)
            matched = self._match(detections, self.tracks)

            # Update matched tracks
            for det_idx, track_idx in matched:
                self.tracks[track_idx]['bbox'] = detections[det_idx][:4]
                self.tracks[track_idx]['age'] = 0

        # Return as array
        return np.array([
            [*track['bbox'], track['id']]
            for track in self.tracks
        ])

    def _match(self, detections, tracks):
        """Simple IoU matching"""
        from scipy.optimize import linear_sum_assignment

        # Compute IoU matrix
        ious = np.zeros((len(detections), len(tracks)))
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                ious[i, j] = compute_iou(det[:4], track['bbox'])

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-ious)  # Maximize

        # Filter low IoU matches
        matched = []
        for r, c in zip(row_ind, col_ind):
            if ious[r, c] > 0.3:
                matched.append((r, c))

        return matched


def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-6)
```

**Test:**
```python
def test_simple_tracker():
    tracker = SimpleTracker()

    # Frame 1: 2 detections
    dets1 = np.array([
        [50, 100, 150, 400],
        [400, 100, 500, 400]
    ])
    tracks1 = tracker.update(dets1)

    assert len(tracks1) == 2
    assert tracks1[0, 4] == 0  # ID 0
    assert tracks1[1, 4] == 1  # ID 1

    # Frame 2: Same players, moved slightly
    dets2 = np.array([
        [55, 105, 155, 405],   # Player 1 moved
        [405, 105, 505, 405]   # Player 2 moved
    ])
    tracks2 = tracker.update(dets2)

    # IDs should be consistent
    assert tracks2[0, 4] == 0
    assert tracks2[1, 4] == 1
```

**Deliverable:**
- Player detection works ✓
- Tracker assigns consistent IDs ✓
- Integrated into pipeline ✓

---

### Week 4: Event Detection

**Goal:** Detect bounce, hit, serve

#### What to Build

```python
# inference/event_detector.py (simplified)

class EventDetector:
    """Detect game events from ball + player data"""

    def __init__(self, sport='tennis'):
        self.sport = sport
        self.ball_history = []  # Last N ball positions
        self.last_bounce_frame = None

    def process_frame(self, frame_id, ball_pos, player_tracks):
        """
        Args:
            frame_id: Current frame number
            ball_pos: (x, y) or None
            player_tracks: [N, 5] array

        Returns:
            events: List of event dicts
        """
        events = []

        # Update history
        self.ball_history.append((frame_id, ball_pos))
        if len(self.ball_history) > 10:
            self.ball_history.pop(0)

        # Detect bounce
        if self._detect_bounce():
            events.append({
                'type': 'bounce',
                'frame': frame_id,
                'position': ball_pos,
                'valid': True  # Simplified (no court detection yet)
            })
            self.last_bounce_frame = frame_id

        # Detect hit
        hit_player = self._detect_hit(ball_pos, player_tracks)
        if hit_player is not None:
            events.append({
                'type': 'hit',
                'frame': frame_id,
                'player_id': hit_player,
                'shot_type': 'groundstroke'  # Simplified
            })

        return events

    def _detect_bounce(self):
        """Detect bounce from y-velocity change"""
        if len(self.ball_history) < 5:
            return False

        # Get recent y positions
        recent_y = [pos[1] for _, pos in self.ball_history[-5:] if pos is not None]

        if len(recent_y) < 5:
            return False

        # Compute velocities
        vy = np.diff(recent_y)

        # Bounce = velocity reversal (down -> up)
        if len(vy) >= 2:
            return vy[-2] > 5 and vy[-1] < -5

        return False

    def _detect_hit(self, ball_pos, player_tracks):
        """Detect if ball is near any player"""
        if ball_pos is None or len(player_tracks) == 0:
            return None

        for i, track in enumerate(player_tracks):
            x1, y1, x2, y2, track_id = track

            # Check if ball is inside player bbox
            if x1 <= ball_pos[0] <= x2 and y1 <= ball_pos[1] <= y2:
                return int(track_id)

        return None
```

**Test:**
```python
def test_bounce_detection():
    detector = EventDetector()

    # Simulate ball falling and bouncing
    ball_trajectory = [
        (0, (100, 100)),  # Start
        (1, (100, 150)),  # Falling
        (2, (100, 200)),  # Falling
        (3, (100, 250)),  # Falling
        (4, (100, 280)),  # At ground
        (5, (100, 250)),  # Bouncing up
        (6, (100, 200)),  # Going up
    ]

    events_all = []
    for frame_id, ball_pos in ball_trajectory:
        events = detector.process_frame(frame_id, ball_pos, np.array([]))
        events_all.extend(events)

    # Should detect bounce around frame 4-5
    bounce_events = [e for e in events_all if e['type'] == 'bounce']
    assert len(bounce_events) >= 1


def test_hit_detection():
    detector = EventDetector()

    # Player bbox
    player_tracks = np.array([
        [50, 100, 150, 400, 0]  # Player 0
    ])

    # Ball inside player bbox
    ball_pos = (100, 200)

    events = detector.process_frame(0, ball_pos, player_tracks)

    hit_events = [e for e in events if e['type'] == 'hit']
    assert len(hit_events) == 1
    assert hit_events[0]['player_id'] == 0
```

**Deliverable:**
- Bounce detection works ✓
- Hit detection works ✓
- Events feed into scoring ✓

---

### Week 5: Stats + Output

**Goal:** Complete scoring engine and output formatting

Already covered in MVP guide - just implement and test!

---

## 3. Synthetic Test Data Generation

**Critical for fast testing without real videos**

### Generate Synthetic Frames with Ball

```python
# tests/fixtures/synthetic_data.py

import numpy as np
import cv2

def generate_synthetic_frames_with_ball(
    num_frames=100,
    ball_trajectory=None,
    image_size=(288, 512),
    ball_radius=5
):
    """
    Generate synthetic frames with a moving ball

    Args:
        ball_trajectory: List of (x, y) positions or None for parabolic

    Returns:
        frames: List of np.array (H, W, 3)
        ball_positions: List of (x, y)
    """
    if ball_trajectory is None:
        # Generate parabolic trajectory
        ball_trajectory = []
        for i in range(num_frames):
            x = 50 + i * 4
            y = 100 + i * 2 - (i ** 2) * 0.01  # Parabola
            ball_trajectory.append((x, y))

    frames = []
    for i in range(num_frames):
        # Create gray court
        frame = np.ones((*image_size, 3), dtype=np.uint8) * 100

        # Draw ball
        x, y = ball_trajectory[i]
        cv2.circle(frame, (int(x), int(y)), ball_radius, (255, 255, 0), -1)

        frames.append(frame)

    return frames, ball_trajectory


def generate_synthetic_frames_with_players(
    num_frames=100,
    player1_trajectory=None,
    player2_trajectory=None,
    image_size=(720, 1280)
):
    """Generate frames with two players"""

    if player1_trajectory is None:
        player1_trajectory = [(100, 300) for _ in range(num_frames)]
    if player2_trajectory is None:
        player2_trajectory = [(1000, 300) for _ in range(num_frames)]

    frames = []
    for i in range(num_frames):
        frame = np.ones((*image_size, 3), dtype=np.uint8) * 100

        # Draw player 1
        x1, y1 = player1_trajectory[i]
        cv2.rectangle(frame, (x1-25, y1-100), (x1+25, y1+100), (255, 0, 0), -1)

        # Draw player 2
        x2, y2 = player2_trajectory[i]
        cv2.rectangle(frame, (x2-25, y2-100), (x2+25, y2+100), (0, 0, 255), -1)

        frames.append(frame)

    return frames


def generate_complete_rally():
    """Generate a complete rally with ball and players"""
    num_frames = 200

    # Ball trajectory (serve -> bounce -> hit -> bounce -> out)
    ball_traj = []

    # Serve (frames 0-30)
    for i in range(30):
        x = 100 + i * 15
        y = 200 - i * 5 + (i ** 2) * 0.1  # Serve trajectory
        ball_traj.append((x, y))

    # Bounce (frame 30-35)
    for i in range(5):
        x = 550
        y = 500 + i * 10 - (i ** 2) * 3  # Bounce
        ball_traj.append((x, y))

    # Return (frames 35-65)
    for i in range(30):
        x = 550 - i * 15
        y = 400 - i * 3
        ball_traj.append((x, y))

    # ... continue for full rally

    frames, _ = generate_synthetic_frames_with_ball(
        num_frames=len(ball_traj),
        ball_trajectory=ball_traj
    )

    return frames, ball_traj
```

**Usage in Tests:**

```python
def test_full_pipeline_with_synthetic_data():
    """Integration test with synthetic rally"""

    frames, ball_positions = generate_complete_rally()

    # Run pipeline
    tracker = BallTrackerWrapper(model_path=None)  # Stub
    detector = EventDetector()
    scorer = TennisScorer()

    events = []
    for frame_id in range(0, len(frames)-2):
        # Track ball
        ball_pos = tracker.track_ball(frames[frame_id:frame_id+3])

        # Detect events
        frame_events = detector.process_frame(frame_id, ball_pos, [])
        events.extend(frame_events)

        # Update score
        for event in frame_events:
            if event['type'] == 'rally_end':
                scorer.award_point(event['winner'])

    # Verify events detected
    assert len([e for e in events if e['type'] == 'bounce']) >= 2
    assert scorer.score[0]['points'] > 0 or scorer.score[1]['points'] > 0
```

---

## 4. Testing Strategy

### Unit Tests (Fast, Isolated)

```python
# tests/test_stats_aggregator.py

def test_stats_aggregator_tracks_aces():
    agg = StatsAggregator()
    agg.start_match({'players': [0, 1], 'sport': 'tennis'})

    # Simulate ace
    agg.process_event({
        'type': 'serve_start',
        'player_id': 0,
        'data': {'is_first_serve': True}
    })

    agg.process_event({
        'type': 'serve_in',
        'player_id': 0,
        'data': {'is_ace': True}
    })

    agg.process_event({
        'type': 'rally_end',
        'winner': 0,
        'reason': 'ace'
    })

    stats = agg.get_match_stats()
    assert stats[0]['aces'] == 1
```

### Integration Tests (Medium, Multiple Components)

```python
# tests/test_integration.py

def test_ball_tracker_feeds_event_detector():
    """Test ball tracker -> event detector pipeline"""

    tracker = BallTrackerWrapper(stub=True)
    detector = EventDetector()

    frames = generate_synthetic_frames_with_ball()

    for i in range(len(frames)-2):
        ball_pos = tracker.track_ball(frames[i:i+3])
        events = detector.process_frame(i, ball_pos, [])

        # Should not crash
        assert isinstance(events, list)
```

### End-to-End Tests (Slow, Full Pipeline)

```python
# tests/test_e2e.py

@pytest.mark.slow
def test_complete_match_processing():
    """Test full pipeline on synthetic match"""

    # Generate 5-minute synthetic match
    frames = generate_synthetic_match(duration_seconds=300, fps=30)

    # Run full pipeline
    result = score_match_stub(frames)

    # Verify output
    assert os.path.exists(result['json_path'])
    assert result['final_score'] is not None
    assert len(result['events']) > 0
```

### Testing Pyramid

```
         /\
        /  \  E2E Tests (5-10 tests, slow)
       /────\
      /      \ Integration Tests (20-30 tests, medium)
     /────────\
    /          \ Unit Tests (100+ tests, fast)
   /────────────\
```

**Run fast tests often:**
```bash
# During development (fast, ~5 seconds)
uv run pytest tests/ -k "not slow"

# Before commit (all tests, ~2 minutes)
uv run pytest tests/ -v

# CI/CD (all tests + coverage)
uv run pytest tests/ --cov=. --cov-report=html
```

---

## 5. Development Workflow

### Daily Development Loop

```bash
# 1. Write a test (that fails)
# tests/test_new_feature.py
def test_new_feature():
    result = new_feature()
    assert result == expected

# 2. Run test (watch it fail)
uv run pytest tests/test_new_feature.py -v
# FAILED: AssertionError

# 3. Implement feature
# my_module/new_feature.py
def new_feature():
    return expected

# 4. Run test (watch it pass)
uv run pytest tests/test_new_feature.py -v
# PASSED ✓

# 5. Run all tests
uv run pytest tests/ -k "not slow"

# 6. Commit
git add .
git commit -m "feat: add new feature"
```

### Testing Tools

```bash
# Install testing tools
uv add pytest
uv add pytest-cov  # Coverage reporting
uv add pytest-xdist  # Parallel testing
uv add pytest-watch  # Auto-run tests on file change

# Run tests in parallel (4 CPUs)
uv run pytest tests/ -n 4

# Watch mode (auto-run on save)
uv run ptw tests/

# Generate coverage report
uv run pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

### Debugging Failed Tests

```python
# Use pytest fixtures for debugging
@pytest.fixture
def debug_mode():
    import pdb; pdb.set_trace()

def test_with_debugger(debug_mode):
    result = my_function()
    # Execution pauses here, can inspect variables
    assert result == expected

# Or use breakpoint() (Python 3.7+)
def test_with_breakpoint():
    result = my_function()
    breakpoint()  # Debugger opens here
    assert result == expected
```

---

## 6. Validation Checklist

### Week 1 Checkpoint ✓
- [ ] Console logger prints events correctly
- [ ] JSON exporter creates valid files
- [ ] Can run stub pipeline end-to-end
- [ ] All unit tests pass

### Week 2 Checkpoint ✓
- [ ] Ball tracker detects ball in synthetic frames
- [ ] Ball tracker integrates with pipeline
- [ ] Position accuracy within 20px on test data
- [ ] Tests pass with both stub and real model

### Week 3 Checkpoint ✓
- [ ] YOLO detects 2 players
- [ ] Tracker assigns consistent IDs across frames
- [ ] Player positions look reasonable in visualization
- [ ] Tests pass

### Week 4 Checkpoint ✓
- [ ] Bounce detection works on synthetic rallies
- [ ] Hit detection works on synthetic rallies
- [ ] Events feed into scoring engine
- [ ] Tests pass

### Week 5 Checkpoint ✓
- [ ] Scoring engine counts points correctly
- [ ] Stats aggregator tracks all metrics
- [ ] Console output looks good
- [ ] JSON output is valid
- [ ] Can process full synthetic match
- [ ] All tests pass

### Final Validation ✓
- [ ] Process real match video (5-10 minutes)
- [ ] Manual verification: score is correct
- [ ] Stats look reasonable
- [ ] No crashes
- [ ] Performance acceptable (>10 FPS)

---

## 7. Common Pitfalls and Solutions

### Pitfall 1: Trying to Run Everything at Once

**Problem:** Building all components, then integrating → hard to debug

**Solution:** Incremental integration with stubs

### Pitfall 2: No Synthetic Test Data

**Problem:** Testing only on real videos → slow iteration

**Solution:** Generate synthetic data for fast testing

### Pitfall 3: Not Testing Edge Cases

**Problem:** Works on happy path, crashes on real data

**Solution:** Test edge cases explicitly

```python
def test_ball_tracker_handles_no_ball():
    tracker = BallTrackerWrapper()
    frames = [np.zeros((288, 512, 3)) for _ in range(3)]  # No ball

    ball_pos = tracker.track_ball(frames)
    assert ball_pos is None  # Should not crash


def test_event_detector_handles_empty_history():
    detector = EventDetector()
    events = detector.process_frame(0, None, [])

    assert events == []  # Should not crash
```

### Pitfall 4: Over-Engineering Early

**Problem:** Building full BoT-SORT with ReID, Kalman filter, etc. → takes forever

**Solution:** Start with SimpleTracker (IoU matching only), upgrade later

### Pitfall 5: Testing on GPU Only

**Problem:** Tests fail on CI/CD without GPU

**Solution:** Make GPU optional, test on CPU

```python
def test_ball_tracker_works_on_cpu():
    tracker = BallTrackerWrapper(model_path=None, device='cpu')
    # ... test logic
```

---

## 8. Example Test Suite Structure

```
tests/
├── conftest.py              # Shared fixtures
├── fixtures/
│   ├── synthetic_data.py    # Synthetic frame generation
│   └── sample_events.py     # Sample event data
├── unit/
│   ├── test_ball_tracker.py
│   ├── test_player_detector.py
│   ├── test_event_detector.py
│   ├── test_stats_aggregator.py
│   ├── test_scoring_engine.py
│   ├── test_console_logger.py
│   └── test_json_exporter.py
├── integration/
│   ├── test_tracking_pipeline.py
│   ├── test_event_to_scoring.py
│   └── test_stats_to_output.py
└── e2e/
    ├── test_stub_pipeline.py
    ├── test_synthetic_match.py
    └── test_real_video.py (slow)
```

---

## 9. Performance Benchmarking

```python
# tests/test_performance.py

import time

def test_ball_tracking_performance():
    """Ball tracker should process 30 FPS"""
    tracker = BallTrackerWrapper('checkpoints/tracknet_v5.pt')

    frames = [np.random.rand(288, 512, 3) for _ in range(90)]  # 3 seconds

    start = time.time()
    for i in range(0, len(frames)-2):
        tracker.track_ball(frames[i:i+3])
    elapsed = time.time() - start

    fps = len(frames) / elapsed
    assert fps >= 20  # Should be real-time on RTX 3070


def test_end_to_end_performance():
    """Full pipeline should process >10 FPS"""
    # Test full pipeline speed
    ...
```

---

## 10. Quick Reference: Test Commands

```bash
# Run all fast tests (during development)
uv run pytest tests/ -k "not slow" -v

# Run specific test file
uv run pytest tests/test_ball_tracker.py -v

# Run specific test
uv run pytest tests/test_ball_tracker.py::test_stub_mode -v

# Run with coverage
uv run pytest tests/ --cov=models --cov=inference

# Run in watch mode (auto-run on file change)
uv run ptw tests/ --runner "pytest -k 'not slow'"

# Run in parallel (fast!)
uv run pytest tests/ -n 4

# Run with debugger on failure
uv run pytest tests/ --pdb

# Run only failed tests from last run
uv run pytest tests/ --lf

# Generate HTML coverage report
uv run pytest tests/ --cov=. --cov-report=html
```

---

## 11. Summary: The Right Way

✅ **DO:**
- Build incrementally (week by week)
- Write tests BEFORE implementation
- Use synthetic data for fast iteration
- Test components in isolation first
- Run tests continuously
- Start simple, add complexity later

❌ **DON'T:**
- Build everything then integrate
- Test only on real videos
- Skip edge case testing
- Over-engineer early
- Ignore failing tests

**Result:** Working system in 5 weeks with confidence it actually works!

---

**End of Guide**
