# Multi-Object Tracking and Automated Scoring System — Design Specification

**Date:** 2026-04-10
**Status:** Draft
**Goal:** Extend TrackNet ball tracking with player detection, court mapping, event detection, and automated scoring for tennis, badminton, and pickleball

---

## 1. Problem Statement

Build on the existing TrackNet ball tracking system (97.5% accuracy) to create a complete match analysis platform that tracks:
- **Ball position** (already solved via TrackNet V2/V5)
- **Player positions and poses** (2 players with persistent IDs)
- **Court geometry** (keypoint detection + homography mapping)
- **Game events** (serves, hits, bounces, faults, winners)
- **Automated scoring** (point/game/set tracking with match statistics)

This transforms TrackNet from a pure ball-tracking system into a SwingVision-like sports analysis platform.

### Why Hybrid Architecture (TrackNet + YOLO)

- **TrackNet for ball**: 97.5% accuracy with sub-pixel precision vs YOLO's 53-90%
- **YOLO for players/court**: COCO pre-trained weights give instant 95%+ person detection
- **Complementary strengths**: Heatmaps excel at tiny objects, bounding boxes excel at large objects
- **No retraining needed**: YOLO works out-of-the-box; only court detector needs sport-specific training

### Scope

**In Scope (this spec):**
- Multi-object tracking (ball + 2 players)
- Court line detection and homography
- Event detection (bounce, hit, serve, rally end)
- Automated scoring engine (tennis rules)
- Real-time visualization overlay

**Out of Scope (future phases):**
- Advanced shot classification ML models (use heuristics initially)
- Player identification (use jersey color/court side initially)
- Multi-camera fusion
- 3D ball trajectory reconstruction
- Mobile/edge deployment optimization

---

## 2. System Architecture

### High-Level Pipeline

```
Input: Video frame(s) @ 30fps (1920×1080)
  ↓
┌─────────────────────────────────┬─────────────────────────────────┐
│     Ball Tracking Branch        │    Scene Understanding Branch   │
│      (TrackNet V5)              │         (YOLOv8-Pose)          │
├─────────────────────────────────┼─────────────────────────────────┤
│ • Input: 3 frames (512×288×9ch) │ • Input: 1 frame (1920×1080)    │
│ • Model: U-Net + MDD + R-STR    │ • Model: YOLOv8x-pose           │
│ • Output: Heatmap (512×288)     │ • Output: Person boxes + poses  │
│ • Post: Centroid extraction     │   Court keypoints (if trained)  │
│ • Result: Ball (x, y, conf)     │ • Result: 2 players + court     │
└─────────────────────────────────┴─────────────────────────────────┘
  ↓                                   ↓
  └──────────────> Fusion Module <─────────┘
                  - Temporal tracker (BoT-SORT)
                  - Court homography
                  - Event detector
                  - Scoring engine
  ↓
Output:
  • Ball trajectory (x, y, conf per frame)
  • Player tracks (ID, bbox, keypoints, court position)
  • Events (serve, hit, bounce, fault, winner)
  • Score (points, games, sets)
  • Annotated video + JSON event log
```

### Module Breakdown

| Module | Responsibility | Input | Output | Complexity |
|--------|---------------|-------|--------|------------|
| **TrackNet** | Ball detection | 3 RGB frames | Ball heatmap | ✅ Existing (97.5% acc) |
| **YOLOv8-Pose** | Player detection | 1 RGB frame | Person boxes + 17 keypoints | ✅ Pre-trained COCO |
| **BoT-SORT** | Player tracking | Detections per frame | Persistent track IDs | 🆕 ~300 LOC |
| **Court Detector** | Court keypoints | 1 RGB frame | 4 corner points | 🆕 Train on ~500 frames |
| **Homography** | Pixel→Court mapping | Ball pos + court points | Court coordinates | 🆕 ~50 LOC (OpenCV) |
| **Event Detector** | Game events | Ball + players + court | Event stream | 🆕 ~500 LOC (heuristics) |
| **Scoring Engine** | Score tracking | Event stream | Match score | 🆕 ~300 LOC (rules) |
| **Visualizer** | Video overlay | All above | Annotated video | 🆕 ~200 LOC |

---

## 3. Ball Tracking (Existing - No Changes)

Use existing TrackNet V2/V5 as-is:
- Input: 3 consecutive frames (512×288 RGB)
- Output: 3 heatmaps → centroid → ball (x, y, confidence)
- Accuracy: 97.5% on tennis/badminton
- Speed: ~30-50 FPS on RTX 3070

**Reference:** `docs/superpowers/specs/2026-03-21-tracknet-ball-tracking-design.md`

**Integration point:** Ball position feeds into event detector.

---

## 4. Player Detection and Tracking

### 4.1 Player Detection (YOLOv8-Pose)

**Model:** Ultralytics YOLOv8x-pose (pre-trained on COCO)

**Detections per frame:**
- Bounding box: `[x1, y1, x2, y2, confidence, class_id]`
- Keypoints: 17 COCO keypoints `[[x, y, confidence], ...]`
  - 0: nose, 5-6: shoulders, 11-12: hips, 13-14: knees, 15-16: ankles
- Filter: `class_id == 0` (person) and `confidence > 0.5`
- Limit: Top 2 detections by confidence (assume singles match)

**Why YOLOv8-pose:**
- COCO pre-trained → 95%+ person detection out-of-the-box
- Keypoints enable shot classification (racket arm, body orientation)
- Fast: 60+ FPS on RTX 3070 at 1920×1080
- Mature ecosystem (Ultralytics, ONNX export, TensorRT support)

### 4.2 Multi-Object Tracking (BoT-SORT)

**Algorithm:** BoT-SORT (Robust Associations Multi-Pedestrian Tracking)
- Paper: https://arxiv.org/abs/2206.14651
- Combines Kalman filter (motion) + IoU matching (spatial) + ReID (appearance)

**State:**
- Track state: `[x, y, w, h, vx, vy, vw, vh]` (position + velocity, 8D)
- Measurement: Bounding box `[x, y, w, h]` (4D)
- Kalman filter: Constant velocity motion model

**Association:**
1. Predict track positions (Kalman prediction step)
2. Compute cost matrix: IoU distance between predictions and detections
3. Hungarian algorithm for optimal assignment
4. Match threshold: IoU > 0.7 for active tracks, IoU > 0.5 for lost tracks
5. Unmatched detections → new tracks (assign new ID)
6. Unmatched tracks → mark lost (keep for 30 frames / 1 second)

**Track management:**
- Track buffer: 30 frames (allow 1-second occlusion before deleting track)
- Track ID: Persistent integer (0, 1 for Player 1, Player 2)
- Track state: `'tracked'`, `'lost'`, `'removed'`

**Optional: ReID appearance features**
- Extract from player bbox using lightweight ReID model (FastReID or OSNet)
- Use for re-identification after long occlusion
- Not required for Phase 1 (IoU matching sufficient for singles tennis)

**Output:**
- Tracks: `[[x1, y1, x2, y2, track_id, confidence], ...]`
- Track IDs are persistent across frames

### 4.3 Player Identification

**Phase 1 (heuristic):**
- Assign Player 1 = left side of court, Player 2 = right side (based on initial position)
- Use court homography to map player bbox center to court coordinates
- Validate with serve detection (server starts from baseline)

**Phase 2 (ML-based, future):**
- Train jersey color classifier (team A vs team B)
- Use facial recognition for player identification (requires dataset)

---

## 5. Court Detection and Homography

### 5.1 Court Line Detector

**Goal:** Detect 4 corner keypoints of the court for homography transformation.

**Architecture:**
- Backbone: ResNet-18 (ImageNet pre-trained)
- Head: Fully connected layer → 8 outputs (4 keypoints × 2 coords)
- Input: Single RGB frame (1920×1080 or 1280×720)
- Output: `[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]`
  - Order: top-left, top-right, bottom-right, bottom-left

**Training:**
- Dataset: ~500-1000 frames manually annotated with court corners
- Use CVAT with 4-point polygon annotation
- Data augmentation: Random crop, color jitter, rotation (±5°)
- Loss: MSE on keypoint coordinates
- Epochs: 20-30 (converges quickly with pre-trained backbone)

**Alternative (if training infeasible):**
- Use classical computer vision: Hough line detection + intersection finding
- Less robust to lighting/shadows but zero training required

**Model file:**
```python
# models/court_detector.py
class CourtLineDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Linear(512, 8)  # 4 corners × (x, y)

    def forward(self, frame):
        """
        Args:
            frame: [B, 3, H, W]
        Returns:
            keypoints: [B, 4, 2] - (x, y) for each corner
        """
        out = self.backbone(frame)
        return out.view(-1, 4, 2)
```

### 5.2 Homography Transformation

**Goal:** Map pixel coordinates → real-world court coordinates (meters).

**Template court (tennis singles):**
```
Court dimensions (ITF standard):
- Length: 23.77 m
- Width (singles): 8.23 m
- Service box length: 6.40 m
- Net position: 11.885 m from baseline

Court template coordinates (origin = top-left corner):
  top_left:     (0,      0)
  top_right:    (23.77,  0)
  bottom_right: (23.77,  8.23)
  bottom_left:  (0,      8.23)
```

**Homography computation:**
```python
import cv2
import numpy as np

# Detected corners in pixel space
pixel_corners = court_detector(frame)  # [4, 2]

# Court template in meters
court_template = np.array([
    [0, 0],           # top-left
    [23.77, 0],       # top-right
    [23.77, 8.23],    # bottom-right
    [0, 8.23]         # bottom-left
], dtype=np.float32)

# Compute homography matrix
H = cv2.getPerspectiveTransform(
    pixel_corners.astype(np.float32),
    court_template
)

# Transform ball position
ball_pixel = np.array([[ball_x, ball_y]], dtype=np.float32)
ball_court = cv2.perspectiveTransform(ball_pixel[None], H)[0][0]
# ball_court is now (x, y) in meters on court
```

**Court zones (tennis):**
```python
def classify_court_zone(x, y):
    """
    Args:
        x, y: Court coordinates in meters
    Returns:
        zone: str (e.g., 'left_service_box', 'out')
    """
    # Out of bounds
    if not (0 <= x <= 23.77 and 0 <= y <= 8.23):
        # Check if slightly out (within 0.5m) for close calls
        if not (-0.5 <= x <= 24.27 and -0.5 <= y <= 8.73):
            return 'out'
        return 'close_to_line'

    # Service boxes (between net and service line)
    if 5.485 <= x <= 11.885:  # Service line to net
        if y < 4.115:
            return 'left_service_box'
        else:
            return 'right_service_box'

    # Backcourt (behind service line)
    if x < 11.885:  # Near-side backcourt
        return 'near_backcourt_left' if y < 4.115 else 'near_backcourt_right'
    else:  # Far-side backcourt
        return 'far_backcourt_left' if y < 4.115 else 'far_backcourt_right'
```

**Accuracy considerations:**
- Homography assumes planar court (valid for hard courts, not clay with texture)
- Camera distortion: Apply lens undistortion before homography if needed
- Dynamic calibration: Re-compute H every N frames to handle camera shake

---

## 6. Event Detection

### 6.1 Event Types

| Event | Trigger | Data | Importance |
|-------|---------|------|------------|
| **serve_start** | Player in service position + ball tossed | Server ID, position | Required for scoring |
| **serve_in** | Ball bounces in service box | Bounce location | Required for scoring |
| **fault** | Ball bounces outside service box | Bounce location, reason | Required for scoring |
| **hit** | Ball near player racket zone + velocity change | Player ID, shot type | Analytics |
| **bounce** | Ball vertical velocity reversal | Bounce location, court zone | Required for scoring |
| **net** | Ball trajectory crosses net line | Net crossing count | Analytics |
| **rally_end** | Out-of-bounds bounce or net violation | Winner, reason | Required for scoring |
| **let** | Serve touches net and lands in | - | Edge case |

### 6.2 Bounce Detection

**Algorithm:**
1. Track ball y-coordinate over last 5 frames
2. Fit quadratic to trajectory: `y = at² + bt + c`
3. Detect inflection point: `dy/dt` changes from positive (down) to negative (up)
4. Threshold: `|dy/dt_before - dy/dt_after| > 10 pixels` (tunable)

**Alternative (physics-based):**
- Compute ball acceleration from trajectory
- Bounce = sudden acceleration spike (impact force)

**Validation:**
- Check ball is near court surface (not mid-air detection)
- Filter spurious bounces (require minimum inter-bounce interval: 10 frames)

**Output:**
```python
{
    'type': 'bounce',
    'frame': 1234,
    'position_pixel': (850, 600),
    'position_court': (15.2, 3.8),  # meters
    'zone': 'far_backcourt_left',
    'valid': True,  # inside court bounds
    'confidence': 0.95
}
```

### 6.3 Hit Detection

**Algorithm:**
1. Check if ball is within player's racket zone (upper 40% of bbox)
2. Validate with ball velocity change:
   - Compute `|v_before - v_after|` over 3-frame window
   - Threshold: `|Δv| > 50 pixels/frame` (tunable)
3. Assign hit to nearest player

**Racket zone approximation:**
- Use shoulder keypoints (YOLO keypoint 5, 6) if available
- Otherwise: upper 40% of player bbox + margin

**Validation:**
- Require ball to be approaching player (not moving away)
- Filter duplicate hits (enforce 5-frame minimum between hits)

**Shot type classification (heuristic):**
```python
def classify_shot(ball_trajectory, player_keypoints, player_bbox):
    # Ball height relative to player
    ball_y = ball_trajectory[-1][1]
    player_center_y = (player_bbox[1] + player_bbox[3]) / 2

    # Trajectory history
    recent_frames = len(ball_trajectory)

    # Rules
    if recent_frames <= 3:
        return 'serve'  # First hit in rally
    elif ball_y < player_center_y - 50:
        return 'volley'  # Hit above player
    elif ball_y > player_center_y + 100:
        return 'low_slice'  # Hit below waist
    else:
        # Use shoulder orientation for forehand/backhand (requires keypoints)
        shoulder_left = player_keypoints[5]
        shoulder_right = player_keypoints[6]
        arm_side = 'left' if ball_x < player_center_x else 'right'
        # (Simplified - needs more logic)
        return 'groundstroke'
```

**Future ML-based classifier:**
- Input: 10-frame ball trajectory + player pose sequence
- LSTM encoder for trajectory, pose encoder for keypoints
- Output: `{serve, forehand, backhand, volley, smash, slice}` (6 classes)
- Train on ~2000 manually labeled shots

### 6.4 Rally State Machine

**States:**
- `idle`: Waiting for serve
- `serving`: Serve toss detected, waiting for first bounce
- `in_play`: Rally active, tracking ball exchanges
- `rally_end`: Point concluded, updating score

**Transitions:**
```
idle → serving:
  Trigger: Player in service position + ball detected above head height

serving → in_play:
  Trigger: Ball bounces in service box (valid serve)

serving → idle:
  Trigger: Ball bounces outside service box (fault) or net violation

in_play → rally_end:
  Trigger: Ball bounces out-of-bounds OR net violation OR double bounce

rally_end → idle:
  Trigger: Score updated, wait 2 seconds
```

**Implementation:**
```python
class RallyStateMachine:
    def __init__(self):
        self.state = 'idle'
        self.serve_count = 0  # 0 or 1 (fault count)
        self.last_hitter = None
        self.rally_start_frame = None

    def process_events(self, events, frame_id):
        if self.state == 'idle':
            if any(e['type'] == 'serve_start' for e in events):
                self.state = 'serving'
                self.rally_start_frame = frame_id
                self.serve_count = 0

        elif self.state == 'serving':
            bounces = [e for e in events if e['type'] == 'bounce']
            if bounces:
                bounce = bounces[0]
                if bounce['zone'] in ['left_service_box', 'right_service_box']:
                    self.state = 'in_play'
                    return [{'type': 'serve_in', 'frame': frame_id}]
                else:
                    self.serve_count += 1
                    if self.serve_count >= 2:
                        self.state = 'rally_end'
                        return [{'type': 'double_fault', 'frame': frame_id}]
                    else:
                        self.state = 'idle'
                        return [{'type': 'fault', 'frame': frame_id}]

        elif self.state == 'in_play':
            # Check for rally end conditions
            bounces = [e for e in events if e['type'] == 'bounce']
            if bounces and not bounces[0]['valid']:
                self.state = 'rally_end'
                winner = self._determine_winner(bounces[0])
                return [{'type': 'rally_end', 'winner': winner, 'reason': 'out'}]

            # Check for double bounce (same player hits twice)
            hits = [e for e in events if e['type'] == 'hit']
            if hits and hits[0]['player_id'] == self.last_hitter:
                self.state = 'rally_end'
                return [{'type': 'rally_end', 'winner': 1 - hits[0]['player_id'],
                         'reason': 'double_hit'}]

            if hits:
                self.last_hitter = hits[0]['player_id']

        elif self.state == 'rally_end':
            # Wait for score update, then reset
            if frame_id - self.rally_start_frame > 60:  # 2 seconds
                self.state = 'idle'

        return []
```

---

## 7. Scoring Engine

### 7.1 Tennis Scoring Rules

**Point scoring:**
- 0 points = "0" (or "love")
- 1 point = "15"
- 2 points = "30"
- 3 points = "40"
- 4+ points = "game" (if leading by 2+)
- Deuce: Both at 40 → require 2-point lead (advantage)

**Game scoring:**
- 6 games = 1 set (if leading by 2+)
- 7 games = 1 set (if 6-6, play tiebreak or continue to 8-6)
- Tiebreak: First to 7 points (with 2-point lead)

**Match scoring:**
- Best of 3 sets (standard) or best of 5 (Grand Slams)

### 7.2 Scoring Engine Implementation

```python
# inference/scoring.py

class TennisScorer:
    def __init__(self, player1_name='Player 1', player2_name='Player 2',
                 best_of=3):
        self.player_names = [player1_name, player2_name]
        self.best_of = best_of  # 3 or 5
        self.reset_match()

    def reset_match(self):
        self.score = {
            0: {'sets': 0, 'games': 0, 'points': 0},
            1: {'sets': 0, 'games': 0, 'points': 0}
        }
        self.server = 0  # 0 = player 0 serves, 1 = player 1 serves
        self.tiebreak = False
        self.match_over = False
        self.rally_history = []

    def award_point(self, winner_id, reason=''):
        """Award point to winner (0 or 1)"""
        if self.match_over:
            return

        loser_id = 1 - winner_id

        # Tiebreak scoring
        if self.tiebreak:
            self.score[winner_id]['points'] += 1
            # Switch server every 2 points in tiebreak
            total_points = sum(self.score[p]['points'] for p in [0, 1])
            if total_points % 2 == 1:
                self.server = 1 - self.server

            # Tiebreak win: first to 7 with 2-point lead
            if (self.score[winner_id]['points'] >= 7 and
                self.score[winner_id]['points'] - self.score[loser_id]['points'] >= 2):
                self._award_game(winner_id)
                self.tiebreak = False
                self.score[0]['points'] = 0
                self.score[1]['points'] = 0
            return

        # Regular point scoring
        self.score[winner_id]['points'] += 1

        # Check for game win
        p_win = self.score[winner_id]['points']
        p_lose = self.score[loser_id]['points']

        # Win game: 4+ points with 2-point lead
        if p_win >= 4 and p_win - p_lose >= 2:
            self._award_game(winner_id)
            self.score[0]['points'] = 0
            self.score[1]['points'] = 0
            self.server = 1 - self.server  # Switch server after game

        # Record rally
        self.rally_history.append({
            'winner': winner_id,
            'reason': reason,
            'score': self.get_score_string()
        })

    def _award_game(self, winner_id):
        """Award game to winner"""
        loser_id = 1 - winner_id
        self.score[winner_id]['games'] += 1

        g_win = self.score[winner_id]['games']
        g_lose = self.score[loser_id]['games']

        # Check for set win
        # Standard: 6 games with 2-game lead
        if g_win >= 6 and g_win - g_lose >= 2:
            self._award_set(winner_id)
        # Tiebreak at 6-6
        elif g_win == 6 and g_lose == 6:
            self.tiebreak = True
        # Long set: first to 8 with 2-game lead (alternative to tiebreak)
        elif g_win >= 8 and g_win - g_lose >= 2:
            self._award_set(winner_id)

    def _award_set(self, winner_id):
        """Award set to winner"""
        loser_id = 1 - winner_id
        self.score[winner_id]['sets'] += 1

        # Reset games
        self.score[0]['games'] = 0
        self.score[1]['games'] = 0

        # Check for match win
        if self.score[winner_id]['sets'] >= (self.best_of + 1) // 2:
            self.match_over = True

    def get_score_string(self):
        """Return human-readable score"""
        if self.tiebreak:
            # Tiebreak score
            return f"Tiebreak: {self.score[0]['points']}-{self.score[1]['points']}"

        # Point score
        point_map = {0: '0', 1: '15', 2: '30', 3: '40'}
        p0 = self.score[0]['points']
        p1 = self.score[1]['points']

        # Deuce/Advantage
        if p0 >= 3 and p1 >= 3:
            if p0 == p1:
                point_str = 'Deuce'
            elif p0 > p1:
                point_str = f'Adv {self.player_names[0]}'
            else:
                point_str = f'Adv {self.player_names[1]}'
        else:
            point_str = f"{point_map.get(p0, '40')}-{point_map.get(p1, '40')}"

        # Game score
        game_str = f"{self.score[0]['games']}-{self.score[1]['games']}"

        # Set score
        set_str = f"Sets: {self.score[0]['sets']}-{self.score[1]['sets']}"

        return f"{point_str} | Games: {game_str} | {set_str}"

    def get_stats(self):
        """Return match statistics"""
        return {
            'rallies': len(self.rally_history),
            'points_won': {
                0: sum(1 for r in self.rally_history if r['winner'] == 0),
                1: sum(1 for r in self.rally_history if r['winner'] == 1)
            },
            'match_duration_rallies': len(self.rally_history),
            'current_score': self.get_score_string(),
            'match_over': self.match_over
        }
```

### 7.3 Badminton/Pickleball Adaptations

**Badminton:**
- Rally point system: Every rally awards a point (no "side-out")
- First to 21 points wins game (with 2-point lead)
- Best of 3 games
- Service alternates after each rally (different from tennis)

**Pickleball:**
- Rally point system (as of 2024 rules update)
- First to 11 points wins game (with 2-point lead), or 15/21 in tournaments
- Win by 2
- No second serve (unlike tennis)

**Implementation:**
Create `BadmintonScorer` and `PickleballScorer` subclasses with sport-specific rules.

---

## 8. Integration and Pipeline

### 8.1 Main Processing Loop

```python
# main.py - score command

def score_match(video_path, output_path, sport='tennis', config=None):
    """
    End-to-end match scoring pipeline

    Args:
        video_path: Input video file
        output_path: Output annotated video + JSON
        sport: 'tennis', 'badminton', or 'pickleball'
        config: Config dict with model paths
    """
    # Initialize models
    ball_tracker = tracknet_v5()
    ball_tracker.load_state_dict(torch.load(config['ball_model']))
    ball_tracker.eval().cuda()

    scene_detector = YOLO(config.get('yolo_model', 'yolov8x-pose.pt'))

    court_detector = CourtLineDetector()
    if config.get('court_model'):
        court_detector.load_state_dict(torch.load(config['court_model']))
    court_detector.eval().cuda()

    # Initialize trackers and scorers
    player_tracker = BoTSORT(track_buffer=30, match_thresh=0.7)
    event_detector = SportsEventDetector(sport=sport)
    scorer = TennisScorer() if sport == 'tennis' else BadmintonScorer()

    # Video I/O
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Processing state
    frame_buffer = []
    frame_id = 0
    event_log = []

    print(f"Processing {total_frames} frames at {fps} fps...")

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # === BUILD 3-FRAME BUFFER ===
            frame_buffer.append(frame)
            if len(frame_buffer) < 3:
                continue
            if len(frame_buffer) > 3:
                frame_buffer.pop(0)

            current_frame = frame_buffer[1]  # Middle frame

            # === BALL TRACKING ===
            # Resize and stack frames for TrackNet
            frames_resized = [
                cv2.resize(f, (512, 288)) for f in frame_buffer
            ]
            frames_tensor = torch.stack([
                torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                for f in frames_resized
            ]).cuda()

            # TrackNet forward pass
            heatmaps = ball_tracker(frames_tensor.unsqueeze(0))  # [1, 3, 288, 512]
            ball_pos = heatmap_to_position(
                heatmaps[0, 1].cpu().numpy(),  # Middle frame heatmap
                threshold=0.5,
                scale_x=width / 512,
                scale_y=height / 288
            )

            # === PLAYER DETECTION ===
            yolo_results = scene_detector(current_frame, verbose=False)

            # Extract person detections
            persons = yolo_results[0].boxes[yolo_results[0].boxes.cls == 0]
            if len(persons) > 2:
                # Keep top 2 by confidence
                top_indices = persons.conf.argsort(descending=True)[:2]
                persons = persons[top_indices]

            player_boxes = persons.xyxy.cpu().numpy() if len(persons) > 0 else np.array([])
            player_keypoints = (yolo_results[0].keypoints.xy.cpu().numpy()
                              if hasattr(yolo_results[0], 'keypoints') else None)

            # === PLAYER TRACKING ===
            player_tracks = player_tracker.update(
                player_boxes,
                features=None  # Add ReID features if needed
            )

            # === COURT DETECTION ===
            frame_tensor = torch.from_numpy(current_frame).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
            court_keypoints = court_detector(frame_tensor)[0].cpu().numpy()  # [4, 2]

            # === EVENT DETECTION ===
            events = event_detector.process_frame(
                frame_id,
                ball_pos,
                player_tracks,
                court_keypoints
            )

            # === SCORING ===
            for event in events:
                if event['type'] == 'rally_end':
                    scorer.award_point(event['winner'], reason=event['reason'])
                    print(f"[Frame {frame_id}] {scorer.get_score_string()}")

                event_log.append(event)

            # === VISUALIZATION ===
            vis_frame = visualize_frame(
                current_frame.copy(),
                ball_pos=ball_pos,
                player_tracks=player_tracks,
                player_keypoints=player_keypoints,
                court_keypoints=court_keypoints,
                events=events,
                score_string=scorer.get_score_string()
            )

            out_video.write(vis_frame)

            # Progress
            if frame_id % 100 == 0:
                print(f"Progress: {frame_id}/{total_frames} ({100*frame_id/total_frames:.1f}%)")

            frame_id += 1

    cap.release()
    out_video.write()

    # === EXPORT RESULTS ===
    results = {
        'video': video_path,
        'sport': sport,
        'total_frames': frame_id,
        'fps': fps,
        'final_score': scorer.get_score_string(),
        'stats': scorer.get_stats(),
        'events': event_log,
        'rally_history': scorer.rally_history
    }

    json_path = output_path.replace('.mp4', '_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== MATCH COMPLETE ===")
    print(f"Final Score: {scorer.get_score_string()}")
    print(f"Total Rallies: {len(scorer.rally_history)}")
    print(f"Annotated video: {output_path}")
    print(f"Results JSON: {json_path}")
```

### 8.2 Visualization

```python
# utils/visualization.py - Extended for multi-object

def visualize_frame(frame, ball_pos, player_tracks, player_keypoints,
                   court_keypoints, events, score_string):
    """
    Draw overlay on frame with all tracking info

    Args:
        frame: RGB numpy array (H, W, 3)
        ball_pos: (x, y) or None
        player_tracks: [N, 6] array
        player_keypoints: [N, 17, 2] array or None
        court_keypoints: [4, 2] array
        events: List of event dicts
        score_string: str

    Returns:
        frame: Annotated frame
    """
    vis = frame.copy()

    # === COURT OVERLAY ===
    if court_keypoints is not None and len(court_keypoints) == 4:
        pts = court_keypoints.astype(np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        for i, pt in enumerate(pts):
            cv2.circle(vis, tuple(pt), 5, (0, 255, 255), -1)

    # === PLAYER OVERLAY ===
    colors = [(255, 0, 0), (0, 0, 255)]  # Red for P1, Blue for P2
    for track in player_tracks:
        x1, y1, x2, y2, track_id, conf = track
        track_id = int(track_id)
        color = colors[track_id % 2]

        # Bounding box
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Label
        label = f"Player {track_id} ({conf:.2f})"
        cv2.putText(vis, label, (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # === KEYPOINTS OVERLAY ===
    if player_keypoints is not None:
        skeleton = [
            (5, 7), (7, 9),   # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 11), (6, 12), # Torso
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16)   # Right leg
        ]
        for i, kpts in enumerate(player_keypoints):
            color = colors[i % 2]
            # Draw skeleton
            for a, b in skeleton:
                if kpts[a, 0] > 0 and kpts[b, 0] > 0:  # Confidence > 0
                    pt_a = (int(kpts[a, 0]), int(kpts[a, 1]))
                    pt_b = (int(kpts[b, 0]), int(kpts[b, 1]))
                    cv2.line(vis, pt_a, pt_b, color, 2)
            # Draw keypoints
            for kpt in kpts:
                if kpt[0] > 0:
                    cv2.circle(vis, (int(kpt[0]), int(kpt[1])), 3, color, -1)

    # === BALL OVERLAY ===
    if ball_pos is not None:
        x, y = int(ball_pos[0]), int(ball_pos[1])
        cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)  # Green ball
        cv2.circle(vis, (x, y), 12, (255, 255, 255), 2)  # White outline

    # === EVENT ANNOTATIONS ===
    for event in events:
        if event['type'] == 'bounce':
            pos = event['position']
            color = (0, 255, 0) if event['valid'] else (0, 0, 255)
            cv2.circle(vis, (int(pos[0]), int(pos[1])), 15, color, 3)
            cv2.putText(vis, event['zone'], (int(pos[0]) + 20, int(pos[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        elif event['type'] == 'hit':
            # Flash effect or trail
            pass

    # === SCORE OVERLAY ===
    # Semi-transparent background
    overlay = vis.copy()
    cv2.rectangle(overlay, (10, 10), (600, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

    # Score text
    cv2.putText(vis, score_string, (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return vis
```

---

## 9. Project Structure

### New Files

```
playground-track-net/
  models/
    multi_tracker.py         # NEW: Hybrid TrackNet + YOLO wrapper
    court_detector.py        # NEW: Court keypoint detection
    trackers/
      __init__.py
      bot_sort.py            # NEW: BoT-SORT multi-object tracker
      track.py               # NEW: Single track with Kalman filter

  inference/
    event_detector.py        # NEW: Game event detection
    scoring.py               # NEW: Scoring engines (Tennis, Badminton, Pickleball)
    homography.py            # NEW: Court coordinate mapping

  utils/
    visualization.py         # EXTEND: Add player/court overlays

  configs/
    scoring_tennis.yaml      # NEW: Config for tennis scoring
    scoring_badminton.yaml   # NEW: Config for badminton scoring

  tests/
    test_multi_tracker.py    # NEW: Tests for player tracking
    test_event_detector.py   # NEW: Tests for event detection
    test_scoring.py          # NEW: Tests for scoring logic

  docs/
    superpowers/specs/
      2026-04-10-multi-object-scoring-system.md  # THIS SPEC
```

---

## 10. Data Requirements

### 10.1 Court Line Annotations

**Need:** ~500-1000 frames with 4 court corners labeled

**Annotation tool:** CVAT
- Use "Polygon" shape with 4 points
- Export as CVAT XML → convert to CSV: `frame_id, x1, y1, x2, y2, x3, y3, x4, y4`

**Dataset creation:**
1. Sample 1 frame per 30 frames from 10-20 different matches (diverse cameras/lighting)
2. Ensure varied court angles (baseline view, side view, elevated)
3. Annotate all 4 corners (top-left, top-right, bottom-right, bottom-left)
4. Split: 70% train, 15% val, 15% test

**Estimate:** ~2-3 hours of annotation work (30 seconds per frame with CVAT polygon tool)

### 10.2 Shot Classification Labels (Optional, Phase 2)

**Need:** ~2000 shot instances labeled

**Labels:** `{serve, forehand, backhand, volley, smash, slice, drop_shot}`

**Annotation:**
- Use CVAT video annotation with temporal segments
- Mark frame range for each shot + label
- Export as action annotations

**Estimate:** ~10-15 hours of annotation work

### 10.3 Pre-trained Models (No Training Needed)

| Component | Model | Source | Notes |
|-----------|-------|--------|-------|
| Ball tracking | TrackNet V5 | Existing (this repo) | Train on tennis/badminton datasets |
| Player detection | YOLOv8x-pose | Ultralytics | Pre-trained on COCO, 95%+ accuracy |
| ReID (optional) | FastReID OSNet | TorchReID | For long-term player tracking |

---

## 11. Performance Targets

### Accuracy

| Metric | Target | Notes |
|--------|--------|-------|
| Ball detection | 97.5% | Existing TrackNet performance |
| Player detection | 95%+ | YOLOv8-pose on clear footage |
| Player tracking (ID persistence) | 98%+ | BoT-SORT with IoU matching |
| Court detection | 90%+ | After training on 500 frames |
| Bounce detection | 85%+ | Heuristic-based, improves with tuning |
| Hit detection | 80%+ | Heuristic-based, 90%+ with ML classifier |
| Event detection (overall) | 85%+ | Composite of above |
| Scoring accuracy | 90%+ | Depends on event detection accuracy |

### Speed

| Configuration | FPS | Notes |
|---------------|-----|-------|
| RTX 3070 (8GB) | 20-25 | Real-time at 24fps video |
| RTX 4090 (24GB) | 60-80 | Batch processing multiple frames |
| CPU only | 3-5 | Not recommended |

**Bottlenecks:**
- TrackNet: ~30 FPS (ball branch)
- YOLOv8x-pose: ~60 FPS (player branch)
- Combined: ~25 FPS (sequential processing)

**Optimization:**
- Run TrackNet and YOLO in parallel (separate GPU streams)
- Use TensorRT for YOLOv8 (2x speedup)
- Reduce YOLO resolution for distant players (trade accuracy for speed)

---

## 12. Validation and Testing

### Unit Tests

```python
# tests/test_event_detector.py

def test_bounce_detection():
    """Test bounce detection from synthetic trajectory"""
    detector = SportsEventDetector()

    # Simulate parabolic trajectory with bounce
    trajectory = generate_bounce_trajectory(
        start=(100, 50),
        bounce=(150, 200),
        end=(200, 100)
    )

    events = []
    for frame_id, pos in enumerate(trajectory):
        e = detector.process_frame(frame_id, pos, [], None)
        events.extend(e)

    # Should detect exactly 1 bounce
    bounces = [e for e in events if e['type'] == 'bounce']
    assert len(bounces) == 1
    assert abs(bounces[0]['frame'] - 25) < 3  # Near middle


def test_tennis_scoring():
    """Test tennis scoring logic"""
    scorer = TennisScorer()

    # Simulate game: Player 0 wins 4 points
    for _ in range(4):
        scorer.award_point(0)

    assert scorer.score[0]['games'] == 1
    assert scorer.score[0]['points'] == 0  # Reset after game

    # Simulate deuce scenario
    scorer.score[0]['points'] = 3
    scorer.score[1]['points'] = 3
    assert 'Deuce' in scorer.get_score_string()
```

### Integration Tests

**Test on known video:**
1. Select 1-minute tennis rally with known score
2. Manually label all events (bounces, hits, score)
3. Run pipeline, compare detected events to ground truth
4. Target: 85%+ event detection accuracy, 95%+ scoring accuracy

### Edge Cases

- **Occlusion:** Player obscures ball (shadow detection)
- **Ball near line:** Ambiguous in/out calls (use probability threshold)
- **Incomplete court view:** Keypoints partially off-screen (RANSAC for robust H)
- **Fast serves:** Motion blur on ball (TrackNet handles this well)
- **Serve let:** Ball touches net, lands in service box (track net crossings)

---

## 13. Deployment and Usage

### CLI Commands

```bash
# Train court detector
uv run python main.py train-court \
  --dataset data/court_annotations/ \
  --epochs 30 \
  --output checkpoints/court_detector.pt

# Run full scoring pipeline
uv run python main.py score \
  --video tennis_match.mp4 \
  --output results/match_annotated.mp4 \
  --sport tennis \
  --ball-model checkpoints/tennis/tracknet_v5.pt \
  --court-model checkpoints/court_detector.pt \
  --config configs/scoring_tennis.yaml

# Batch processing
uv run python main.py score-batch \
  --input-dir videos/ \
  --output-dir results/ \
  --sport tennis
```

### Config File Example

```yaml
# configs/scoring_tennis.yaml

sport: tennis
best_of: 3  # Best of 3 sets

models:
  ball_model: checkpoints/tennis/tracknet_v5.pt
  yolo_model: yolov8x-pose.pt
  court_model: checkpoints/court_detector.pt

tracking:
  track_buffer: 30  # frames
  iou_threshold: 0.7
  confidence_threshold: 0.5

event_detection:
  bounce_threshold: 10  # pixels/frame velocity change
  hit_threshold: 50     # pixels/frame velocity change
  bounce_interval_min: 10  # frames between bounces

visualization:
  show_ball: true
  show_players: true
  show_keypoints: true
  show_court: true
  show_events: true
  score_overlay: true

output:
  save_video: true
  save_json: true
  save_csv: true  # Frame-by-frame ball/player positions
```

---

## 14. Future Enhancements (Out of Scope)

### Phase 3 (Post-MVP)
- **ML-based shot classification:** Replace heuristics with LSTM/Transformer model
- **Player identification:** Face recognition or jersey number detection
- **Advanced statistics:**
  - Rally length distribution
  - Shot placement heatmaps
  - Player movement patterns (court coverage)
  - Serve speed estimation (from ball trajectory)

### Phase 4 (Research)
- **3D ball trajectory reconstruction:** Stereo vision or monocular depth
- **Spin detection:** Analyze ball rotation from high-FPS footage
- **Automated highlight generation:** Extract key points (aces, winners, long rallies)
- **Real-time streaming:** Low-latency processing for live broadcasts

### Phase 5 (Production)
- **Multi-camera fusion:** Combine views for robust tracking
- **Edge deployment:** TensorRT optimization for Jetson AGX
- **Web dashboard:** Live scoring display with WebSocket
- **Mobile app:** Offline processing on iOS/Android

---

## 15. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Court detection poor on varied cameras** | High | Medium | Use classical CV fallback (Hough lines); collect diverse training data |
| **Player tracking fails in doubles** | Medium | High | Current design assumes singles; doubles needs 4-person tracking (expand BoT-SORT) |
| **Event detection false positives** | High | Medium | Tune thresholds per sport; add ML classifier in Phase 2 |
| **Scoring errors accumulate** | Medium | High | Add manual correction UI; validate on known matches |
| **Real-time performance < 24fps** | Low | Medium | Profile and optimize bottlenecks; use TensorRT for YOLO |
| **Court occlusion (baseline camera)** | Medium | Low | Homography still works with 3/4 corners (RANSAC); use last known H |
| **Limited training data for court detector** | Medium | Low | 500 frames sufficient; augment heavily (rotation, crop, lighting) |

---

## 16. Success Criteria

### Phase 1: Proof of Concept (Week 1-2)
- [ ] TrackNet ball tracking integrated (reuse existing)
- [ ] YOLOv8-pose player detection running
- [ ] BoT-SORT tracker assigns persistent player IDs
- [ ] Visualization shows ball + 2 players on video

### Phase 2: Event Detection (Week 3-4)
- [ ] Court detector trained (90%+ accuracy on test set)
- [ ] Homography maps ball to court coordinates
- [ ] Bounce detection works (85%+ accuracy)
- [ ] Hit detection works (80%+ accuracy)
- [ ] Rally state machine tracks serve/in-play/end

### Phase 3: Scoring System (Week 5-6)
- [ ] Tennis scoring engine implemented
- [ ] Automated scoring runs on 5-minute test video
- [ ] Scoring accuracy 90%+ vs manual labels
- [ ] JSON event log exported
- [ ] Annotated video with score overlay

### Phase 4: Production Ready (Week 7-8)
- [ ] CLI commands documented and tested
- [ ] Batch processing works on directory of videos
- [ ] Performance: 20+ FPS on RTX 3070
- [ ] Unit tests for all new modules (80%+ coverage)
- [ ] Integration test on full match video
- [ ] User documentation written

---

## 17. Dependencies

### New Python Packages

```toml
# Add to pyproject.toml

[project.dependencies]
ultralytics = ">=8.0.0"        # YOLOv8
scipy = ">=1.10.0"             # Already in project (homography, etc.)
scikit-learn = ">=1.3.0"       # For Hungarian algorithm (linear_sum_assignment)

[project.optional-dependencies]
reid = [
    "torchreid>=0.2.5",        # ReID models (optional for Phase 2)
]
```

### Model Weights

| Model | Size | Download | Usage |
|-------|------|----------|-------|
| TrackNet V5 | ~8MB | Train in this repo | Ball tracking (existing) |
| YOLOv8x-pose | ~130MB | `yolov8x-pose.pt` (Ultralytics) | Player detection |
| Court detector | ~45MB | Train in this repo | Court keypoints |

---

## 18. Development Timeline

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| **1-2** | Multi-object tracking | `multi_tracker.py`, `bot_sort.py`, visualization demo |
| **3** | Court detection | `court_detector.py`, training script, trained model |
| **4** | Event detection | `event_detector.py`, `homography.py`, bounce/hit tests |
| **5** | Scoring engine | `scoring.py` (Tennis), rally state machine, unit tests |
| **6** | Integration | End-to-end pipeline, CLI commands, config files |
| **7** | Testing & validation | Integration tests, edge case handling, accuracy benchmarks |
| **8** | Documentation & polish | User guide, API docs, example videos, README update |

---

## 19. References

### Papers
- **TrackNet:** https://arxiv.org/abs/1907.03698 (ball tracking)
- **BoT-SORT:** https://arxiv.org/abs/2206.14651 (multi-object tracking)
- **YOLOv8:** https://docs.ultralytics.com (object detection)

### Datasets
- **TrackNet Tennis:** https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut
- **CoachAI Badminton:** https://nycu1-my.sharepoint.com/:u:/g/personal/tik_m365_nycu_edu_tw/...
- **COCO (for YOLO):** https://cocodataset.org

### Code References
- **TrackNetV3:** https://github.com/qaz812345/TrackNetV3
- **Ultralytics YOLOv8:** https://github.com/ultralytics/ultralytics
- **BoT-SORT (official):** https://github.com/NirAharon/BoT-SORT

---

## 20. Appendix: Example Event Log

```json
{
  "video": "tennis_match.mp4",
  "sport": "tennis",
  "total_frames": 5400,
  "fps": 30.0,
  "final_score": "40-30 | Games: 3-2 | Sets: 1-0",
  "stats": {
    "rallies": 28,
    "points_won": {
      "0": 15,
      "1": 13
    }
  },
  "events": [
    {
      "type": "serve_start",
      "frame": 120,
      "server": 0,
      "position": [850, 300]
    },
    {
      "type": "bounce",
      "frame": 145,
      "position_pixel": [920, 650],
      "position_court": [18.5, 3.2],
      "zone": "left_service_box",
      "valid": true,
      "confidence": 0.97
    },
    {
      "type": "serve_in",
      "frame": 145
    },
    {
      "type": "hit",
      "frame": 148,
      "player_id": 1,
      "shot_type": "groundstroke",
      "position": [950, 680]
    },
    {
      "type": "bounce",
      "frame": 175,
      "position_court": [5.2, 2.8],
      "zone": "out",
      "valid": false
    },
    {
      "type": "rally_end",
      "frame": 175,
      "winner": 0,
      "reason": "out"
    }
  ],
  "rally_history": [
    {
      "winner": 0,
      "reason": "out",
      "score": "15-0 | Games: 0-0 | Sets: 0-0"
    }
  ]
}
```

---

**End of Specification**
