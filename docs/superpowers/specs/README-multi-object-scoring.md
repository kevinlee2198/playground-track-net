# Multi-Object Tracking and Automated Scoring — Project Overview

**Branch:** `feat/multi-object-scoring`
**Status:** Specification Phase
**Goal:** Extend TrackNet ball tracking with player detection, court mapping, and automated scoring

---

## 📋 Documentation

### Core Specifications

1. **[Multi-Object Scoring System Design](2026-04-10-multi-object-scoring-system.md)**
   - System architecture (TrackNet + YOLO hybrid)
   - Player tracking (BoT-SORT multi-object tracker)
   - Court detection and homography
   - Event detection (bounce, hit, serve, rally tracking)
   - Automated scoring engine (tennis/badminton/pickleball)
   - 8-week implementation timeline

2. **[Training Guide: Multi-Camera Support](2026-04-10-training-guide-multi-camera.md)**
   - Open source datasets (TrackNet, CoachAI, OpenTTGames)
   - Camera angle challenges (overhead vs baseline vs side)
   - Custom data collection and annotation (CVAT workflows)
   - Training procedures for each component
   - Transfer learning strategies

3. **[Player Statistics Tracking System](2026-04-10-player-stats-tracking-system.md)**
   - Player profiles and database schema (SQLite/PostgreSQL)
   - Match statistics tracking (50+ metrics per player)
   - Sport-specific stats (tennis/badminton/pickleball)
   - Aggregate statistics and career tracking
   - Head-to-head records and leaderboards
   - REST API (FastAPI) and dashboard (Streamlit)

4. **[MVP Implementation Guide](2026-04-10-mvp-implementation-guide.md)** ⭐ **START HERE**
   - Console output first (no database/API initially)
   - Real-time score and stats logging
   - JSON export for backend integration
   - Simplified architecture for rapid development
   - Backend integration guide (when ready)

---

## 🎯 Implementation Strategy

**Phase 1: Console Output MVP** (Weeks 1-5)
- Output score and stats to console in real-time
- Export JSON files for your backend to consume
- No database, no API, no dashboard (keep it simple!)
- Focus: Get multi-object tracking working end-to-end

**Phase 2: Backend Integration** (Week 6+)
- POST JSON to your existing backend API
- Your backend handles: database, user management, authentication
- TrackNet becomes a "match processor" that feeds your system

---

## 🎯 Quick Summary

### What This Adds to TrackNet

| Current (main branch) | This Feature (feat/multi-object-scoring) |
|----------------------|------------------------------------------|
| ✅ Ball tracking (97.5% accuracy) | ✅ Ball tracking (same) |
| ❌ No player tracking | ✅ 2 players with persistent IDs + pose |
| ❌ No court detection | ✅ Court line detection + homography |
| ❌ No event detection | ✅ Serve, hit, bounce, rally tracking |
| ❌ No scoring | ✅ Automated point/game/set scoring |
| ❌ No player stats | ✅ Player profiles + 50+ match stats |
| ❌ No database | ✅ SQLite database + REST API |
| ❌ No analytics | ✅ Career stats, trends, leaderboards |
| ✅ Overhead camera only | ✅ Baseline + side camera support |

### Architecture

```
Input: Video (tennis/badminton/pickleball match)
  ↓
┌─────────────────┬─────────────────────┐
│ Ball Branch     │ Scene Branch        │
│ (TrackNet V5)   │ (YOLOv8-pose)       │
│ • Heatmap       │ • Player boxes      │
│ • Sub-pixel     │ • 17 keypoints      │
│   accuracy      │ • Pre-trained       │
└─────────────────┴─────────────────────┘
  ↓
Multi-Object Tracker (BoT-SORT)
  ↓
Court Homography
  ↓
Event Detector (bounce/hit/serve)
  ↓
Scoring Engine (tennis rules)
  ↓
Statistics Aggregator (extract 50+ metrics)
  ↓
Player Database (SQLite)
  ↓
REST API (FastAPI) + Dashboard (Streamlit)
  ↓
Output: Annotated video + Score + Stats + Leaderboards
```

---

## 📊 Training Requirements

### What Needs Training?

| Component | Training Needed? | Data Required | Open Source Data Available? |
|-----------|-----------------|---------------|----------------------------|
| **Ball Tracker** | ✅ Yes | 5K-20K frames | ✅ Yes (TrackNet: 20K overhead, CoachAI: 78K) |
| **Player Detector** | ❌ No (pre-trained) | None | ✅ COCO pre-trained (YOLOv8) |
| **Court Detector** | ✅ Yes | 500-1000 frames | ❌ No (must annotate) |
| **Event Detector** | ❌ No (heuristics) | None | N/A |
| **Scoring Engine** | ❌ No (rules-based) | None | N/A |

### Camera Angle Data Availability

| Angle | Open Source Datasets | Need to Collect? |
|-------|---------------------|------------------|
| **Overhead** | ✅ TrackNet Tennis (20K), CoachAI Badminton (78K) | ❌ No |
| **Elevated Side** | ⚠️ Limited (some badminton datasets) | ✅ Yes (recommended) |
| **Baseline** | ❌ None for tennis/badminton | ✅ Yes (required) |

**Solution:** Use transfer learning (overhead → baseline) + data augmentation (perspective transforms)

---

## 🚀 Implementation Roadmap

### Phase 1: Proof of Concept (Week 1-2)
- [ ] Integrate TrackNet ball tracking (existing)
- [ ] Add YOLOv8-pose player detection
- [ ] Implement BoT-SORT tracker
- [ ] Visualization: ball + 2 players on video

### Phase 2: Event Detection (Week 3-4)
- [ ] Train court detector (500 annotated frames)
- [ ] Implement homography mapping
- [ ] Build event detector (bounce, hit)
- [ ] Rally state machine

### Phase 3: Scoring (Week 5-6)
- [ ] Tennis scoring engine
- [ ] Automated scoring on test video (90%+ accuracy)
- [ ] JSON event log export

### Phase 4: Multi-Camera Support (Week 7-8)
- [ ] Collect baseline camera data (5 matches)
- [ ] Train baseline TrackNet model (transfer learning)
- [ ] Camera angle auto-detection
- [ ] End-to-end validation

---

## 📦 Deliverables

### Code (to be implemented)
```
models/
  multi_tracker.py         # Hybrid TrackNet + YOLO
  court_detector.py        # Court keypoint detector (ResNet-18)
  trackers/
    bot_sort.py            # Multi-object tracker
    track.py               # Single track with Kalman filter

inference/
  event_detector.py        # Game event detection
  scoring.py               # Tennis/badminton/pickleball scoring
  homography.py            # Pixel → court coordinate mapping
  stats_aggregator.py      # Extract statistics from events

utils/
  console_logger.py        # Pretty console output (Phase 1)
  json_exporter.py         # Export match data to JSON (Phase 1)

# Phase 2 - Your Backend Integration (later)
database/
  schema.sql               # [Optional] Reference schema for your backend
  stats_db.py              # [Optional] If you want local DB

api/
  backend_client.py        # POST JSON to your backend API

dashboard/
  # Your existing frontend handles this!

scripts/
  train_court_detector.py          # Train court model
  cvat_to_tracknet_csv.py          # Annotation converter
  pre_annotate_with_tracknet.py   # Speedup annotation
  generate_synthetic_baseline.py   # Data augmentation
  init_database.py                 # Initialize statistics database
  export_stats.py                  # Export stats to CSV/PDF

configs/
  scoring_tennis.yaml      # Tennis scoring config
  scoring_badminton.yaml   # Badminton scoring config
```

### Documentation
- [x] System design spec (2026-04-10-multi-object-scoring-system.md)
- [x] Training guide (2026-04-10-training-guide-multi-camera.md)
- [x] Player statistics spec (2026-04-10-player-stats-tracking-system.md) *[Phase 2]*
- [x] **MVP implementation guide (2026-04-10-mvp-implementation-guide.md)** ← START HERE
- [ ] User guide (how to use scoring system)
- [ ] API documentation *[Phase 2 - Your Backend]*

### Data Collection Tools
- [x] CVAT annotation workflows
- [x] XML → CSV conversion scripts
- [x] Pre-annotation pipeline
- [ ] Court corner annotation templates

---

## 📈 Performance Targets

### Accuracy

| Metric | Target | Notes |
|--------|--------|-------|
| Ball detection | 97.5% | Existing TrackNet (overhead) |
| Ball detection (baseline) | 85-92% | After transfer learning |
| Player detection | 95%+ | YOLOv8-pose pre-trained |
| Court detection | 90%+ | After training on 500 frames |
| Event detection | 85%+ | Heuristic-based (Phase 1) |
| Automated scoring | 90%+ | Depends on event accuracy |

### Speed

| Hardware | FPS | Real-time? |
|----------|-----|------------|
| RTX 3070 (8GB) | 20-25 | ✅ Yes (24fps video) |
| RTX 4090 (24GB) | 60-80 | ✅ Yes |
| CPU only | 3-5 | ❌ No |

---

## 🎓 Learning Resources

### Papers
- **TrackNet:** [arXiv:1907.03698](https://arxiv.org/abs/1907.03698) - Ball tracking via heatmaps
- **BoT-SORT:** [arXiv:2206.14651](https://arxiv.org/abs/2206.14651) - Multi-object tracking
- **YOLOv8:** [Ultralytics Docs](https://docs.ultralytics.com) - Object detection + pose

### Datasets
- **TrackNet Tennis:** [Google Drive](https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut)
- **CoachAI Badminton:** [SharePoint](https://nycu1-my.sharepoint.com/:u:/g/personal/tik_m365_nycu_edu_tw/...)
- **OpenTTGames:** [Website](https://opentables.epfl.ch/) (table tennis, side view)

### Tools
- **CVAT:** [GitHub](https://github.com/opencv/cvat) - Video annotation
- **Ultralytics:** [GitHub](https://github.com/ultralytics/ultralytics) - YOLOv8 framework

---

## 🔧 Quick Start (for Developers)

### 1. Read the MVP Implementation Guide
```bash
# This is your starting point!
cat docs/superpowers/specs/2026-04-10-mvp-implementation-guide.md
```

### 2. Install Dependencies
```bash
# Add to pyproject.toml
uv add ultralytics  # YOLOv8
uv add opencv-python  # Already have
uv add scikit-learn  # For Hungarian algorithm

uv sync
```

### 3. Implement MVP (Weeks 1-5)
```bash
# Week 1-2: Multi-object tracking
# - Implement BoT-SORT tracker (models/trackers/bot_sort.py)
# - Implement multi-tracker wrapper (models/multi_tracker.py)
# - Test on sample video

# Week 3-4: Event detection
# - Implement event detector (inference/event_detector.py)
# - Test bounce/hit detection

# Week 5: Console output + JSON export
# - Implement console logger (utils/console_logger.py)
# - Implement JSON exporter (utils/json_exporter.py)
# - Integrate with scoring pipeline

# Run first match!
uv run python main.py score \
  --video test_match.mp4 \
  --player1 "Player 1" \
  --player2 "Player 2"
```

### 4. Backend Integration (Week 6+)
```bash
# When MVP works, POST JSON to your backend
# See MVP guide Section 7: Backend Integration
```

---

## 📝 Next Steps

### For Implementation
1. Review both specification documents
2. Set up CVAT for annotation (if collecting custom data)
3. Download TrackNet overhead dataset (for transfer learning)
4. Implement BoT-SORT tracker (models/trackers/bot_sort.py)
5. Implement multi-tracker wrapper (models/multi_tracker.py)
6. Test on sample video

### For Custom Data Collection
1. Record 5 matches with baseline camera
2. Annotate 200 frames for court corners (1 hour)
3. Pre-annotate ball with overhead model
4. Manually correct 2 matches (~4-6 hours)
5. Train baseline TrackNet (transfer learning)
6. Train court detector

### For Questions/Discussion
- Camera angle detection heuristics
- Shot classification approach (heuristic vs ML)
- Doubles support (4 players instead of 2)
- Real-time optimization strategies

---

## 🤝 Contributing

See implementation tasks in:
- Main spec: Section 16 (Success Criteria)
- Training guide: Section 9 (Quick Start Checklist)

---

**Last Updated:** 2026-04-10
**Branch:** `feat/multi-object-scoring`
**Status:** Specification complete, implementation pending
