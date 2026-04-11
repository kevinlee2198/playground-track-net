# Training Guide: Multi-Object Scoring System with Multi-Camera Support

**Date:** 2026-04-10
**Status:** Draft
**Related:** `2026-04-10-multi-object-scoring-system.md`

---

## 1. Training Overview: What Needs Training?

| Component | Training Needed? | Pre-trained Available? | Custom Data Required? |
|-----------|-----------------|----------------------|---------------------|
| **Ball Tracker (TrackNet)** | ✅ Yes | ❌ No (sport-specific) | ✅ Yes (~20K frames) |
| **Player Detector (YOLO)** | ❌ No* | ✅ Yes (COCO) | ❌ No (optional fine-tune) |
| **Court Detector** | ✅ Yes | ❌ No (camera-specific) | ✅ Yes (~500-1000 frames) |
| **Multi-Object Tracker (BoT-SORT)** | ❌ No (algorithm) | N/A | ❌ No |
| **Event Detector** | ❌ No (heuristics) | N/A | ❌ No (Phase 1) |
| **Shot Classifier** | ⚠️ Optional (Phase 2) | ❌ No | ✅ Yes (~2000 shots) |

*Fine-tuning YOLO on sport-specific players can improve accuracy but not required initially.

### Training Priority

**Must Train:**
1. **TrackNet** (ball tracking) - Existing datasets available
2. **Court Detector** - Need to collect custom annotations

**Optional (Phase 2):**
3. **YOLO fine-tuning** - Improve player detection in sport-specific scenarios
4. **Shot Classifier** - ML-based shot type recognition

---

## 2. Open Source Datasets

### 2.1 Ball Tracking Datasets (TrackNet)

| Dataset | Sport | Frames | Resolution | Annotations | Camera Angle | Download |
|---------|-------|--------|-----------|-------------|--------------|----------|
| **TrackNet Tennis V1** | Tennis | 19,835 | 1280×720 | Ball center (x,y) + visibility | Overhead broadcast | [Google Drive](https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut) |
| **TrackNet Tennis V2** | Tennis | 20,844 | 1280×720 | Ball center + visibility | Overhead broadcast | [NYCU GitLab](https://gitlab.nol.cs.nycu.edu.tw/open-source/TrackNetv2) |
| **CoachAI Badminton** | Badminton | 78,200 | 1280×720 | Shuttlecock center + visibility | Overhead/elevated | [SharePoint](https://nycu1-my.sharepoint.com/:u:/g/personal/tik_m365_nycu_edu_tw/EWisYhAiai9Ju7L-tQp0ykEBZJd9VQkKqsFrjcqqYIDP-g) |
| **TrackNet-Pickleball** | Pickleball | ~12,000 | 1920×1080 | Ball center + visibility | Overhead | [GitHub](https://github.com/AndrewDettor/TrackNet-Pickleball) |
| **WASB-SBDT** | Badminton | Varies | Mixed | Multi-annotation types | Mixed | [GitHub](https://github.com/nttcom/WASB-SBDT) |
| **OpenTTGames** | Table Tennis | 50,000+ | 1920×1080 | Ball + segmentation | Side view | [Website](https://opentables.epfl.ch/) |

**⚠️ Camera Angle Note:** Most datasets are overhead/broadcast view. OpenTTGames has side views but for table tennis.

### 2.2 Player Detection Datasets

| Dataset | Annotations | Frames | Notes |
|---------|-------------|--------|-------|
| **COCO** | Person bbox + 17 keypoints | 200K+ | YOLOv8-pose pre-trained on this |
| **MPII Human Pose** | 25K pose annotations | 25K | General pose, not sport-specific |
| **PoseTrack** | Person + pose in video | 23K | Good for tracking scenarios |

**✅ No training needed:** YOLOv8-pose works out-of-the-box with 95%+ accuracy on tennis/badminton players.

### 2.3 Court Detection Datasets

**❌ No public datasets found** for court corner/line detection across multiple camera angles.

**Available (limited):**
- **TrackNet datasets** include court images but no keypoint annotations
- **TTNet** (table tennis) has court corners for table tennis (not applicable to tennis/badminton courts)

**Solution:** Annotate custom dataset (details in Section 4).

### 2.4 Shot Classification Datasets

| Dataset | Sport | Shots Labeled | Labels | Availability |
|---------|-------|---------------|--------|--------------|
| **TTStroke-21** | Table Tennis | ~11K strokes | 21 shot types | [Paper](https://arxiv.org/abs/2105.09957) |
| **Tennis Shot Dataset** | Tennis | Limited | Serve, forehand, backhand | No public release found |

**⚠️ Limited availability** for tennis/badminton. Need to collect custom data (Section 4.4).

### 2.5 Multi-View Sports Datasets

**Critical for camera angle robustness:**

| Dataset | Sport | Views | Frames | Annotations | Download |
|---------|-------|-------|--------|-------------|----------|
| **MMSports** | Soccer, Basketball | 3-8 cameras | ~160K | Player tracking, actions | [GitHub](https://github.com/MCG-NJU/MMSports) (2024) |
| **SoccerNet Camera Calibration** | Soccer | Multi-cam | Varies | Camera matrices, court lines | [Website](https://www.soccer-net.org/) |
| **DeepSport** | Basketball | 7 cameras | 19K | Player tracking | [GitHub](https://github.com/gabriel-vanzandycke/deepsport) |

**⚠️ Wrong sport** but useful for:
- Multi-camera calibration techniques
- Camera angle invariant architectures
- Homography estimation strategies

**Tennis/Badminton specific multi-view:** Not found in public datasets (need custom collection).

---

## 3. Camera Angle Challenges and Solutions

### 3.1 Camera Angle Types

```
OVERHEAD (Broadcast)          ELEVATED SIDE             BASELINE
     [Existing datasets]      [Medium challenge]        [Highest challenge]

       Net                         Net                      Net
    ┌────────┐                  ┌────────┐                   │
    │        │                 /          \                  │
    │   ●    │   Camera       /     ●      \                 │
    │        │      ↑        /              \                │
    └────────┘              /________________\      Camera   │    ●
                                                        ←     │
   - Full court visible    - Perspective distortion    - Severe perspective
   - Ball always clear     - Far court compressed      - Ball tiny at far end
   - Minimal occlusion     - Moderate occlusion        - Heavy player occlusion
```

### 3.2 Impact on Each Component

| Component | Overhead | Elevated Side | Baseline |
|-----------|----------|---------------|----------|
| **Ball Detection** | ✅ 97.5% (existing) | ⚠️ 85-90% (far court small) | ❌ 70-80% (occlusion) |
| **Player Detection** | ✅ 95%+ | ✅ 95%+ | ✅ 90%+ (partial occlusion) |
| **Court Detection** | ✅ Easy (4 corners visible) | ⚠️ Moderate (perspective) | ❌ Hard (only near court visible) |
| **Homography** | ✅ Straightforward | ⚠️ Needs lens correction | ❌ Complex (non-linear) |
| **Event Detection** | ✅ High accuracy | ⚠️ Moderate | ❌ Low (ball often hidden) |

### 3.3 Solutions for Non-Overhead Cameras

#### Strategy 1: Camera-Specific Models (Recommended for Phase 1)

Train separate models for each camera angle category:

```python
# models/tracknet.py - Extended

def tracknet_v5(camera_angle='overhead'):
    """
    Factory with camera-specific weights

    Args:
        camera_angle: 'overhead', 'elevated_side', 'baseline'
    """
    model = TrackNet(backbone, mdd, rstr)

    # Load camera-specific weights
    weights_map = {
        'overhead': 'checkpoints/tracknet_overhead.pt',
        'elevated_side': 'checkpoints/tracknet_elevated.pt',
        'baseline': 'checkpoints/tracknet_baseline.pt'
    }

    model.load_state_dict(torch.load(weights_map[camera_angle]))
    return model
```

**Pros:**
- Highest accuracy per camera type
- Simple implementation

**Cons:**
- Need to collect data for each angle
- User must specify camera type

#### Strategy 2: Multi-Camera Data Augmentation

Synthesize camera angles during training:

```python
# data/transforms.py

class PerspectiveAugmentation:
    """Simulate camera angle changes via perspective transform"""

    def __init__(self, elevation_range=(0, 45), rotation_range=(-30, 30)):
        self.elevation_range = elevation_range
        self.rotation_range = rotation_range

    def __call__(self, frame, ball_label):
        # Random camera elevation angle
        elevation = np.random.uniform(*self.elevation_range)
        rotation = np.random.uniform(*self.rotation_range)

        # Compute perspective transform matrix
        H = self._compute_perspective(elevation, rotation, frame.shape)

        # Warp frame
        frame_warped = cv2.warpPerspective(frame, H, frame.shape[:2][::-1])

        # Transform ball coordinates
        ball_warped = cv2.perspectiveTransform(
            np.array([[ball_label]], dtype=np.float32), H
        )[0][0]

        return frame_warped, ball_warped

    def _compute_perspective(self, elevation, rotation, shape):
        """Compute 3D rotation -> 2D projection matrix"""
        h, w = shape[:2]

        # Camera intrinsics (approximate)
        f = w  # focal length
        cx, cy = w / 2, h / 2

        # 3D rotation matrices
        theta_x = np.radians(elevation)
        theta_y = np.radians(rotation)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])

        Ry = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])

        R = Rx @ Ry

        # Camera matrix
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])

        # Homography
        H = K @ R @ np.linalg.inv(K)
        H = H / H[2, 2]  # Normalize

        return H
```

**Usage in training:**
```python
# training/trainer.py

augmentations = torchvision.transforms.v2.Compose([
    HorizontalFlip(p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2),
    PerspectiveAugmentation(elevation_range=(0, 30), rotation_range=(-20, 20)),
    Mixup(alpha=0.2)
])
```

**Pros:**
- Single model works across camera angles
- No manual angle specification needed

**Cons:**
- Synthetic augmentation may not match real angles perfectly
- Slightly lower accuracy than camera-specific models

#### Strategy 3: Camera Angle Prediction (Advanced)

Add a camera angle prediction head:

```python
# models/multi_tracker.py - Extended

class CameraAwareMultiTracker(nn.Module):
    def __init__(self):
        super().__init__()

        # Camera angle classifier
        self.angle_predictor = CameraAnglePredictor()

        # Angle-conditioned ball tracker
        self.ball_tracker = AngleConditionedTrackNet()

        # Court detector with angle adaptation
        self.court_detector = AdaptiveCourtDetector()

    def forward(self, frames):
        # Predict camera angle from first frame
        angle_features = self.angle_predictor(frames[:, 0])
        # angle_features: [elevation, rotation, fov]

        # Use angle features to modulate tracking
        ball_heatmap = self.ball_tracker(frames, angle_features)
        court_keypoints = self.court_detector(frames[:, 1], angle_features)

        return ball_heatmap, court_keypoints


class CameraAnglePredictor(nn.Module):
    """Predict camera elevation, rotation, FOV from image"""
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, 3)  # elevation, rotation, fov

    def forward(self, frame):
        return self.backbone(frame)  # [batch, 3]
```

**Pros:**
- Automatic camera angle adaptation
- No user input needed

**Cons:**
- Complex to implement
- Needs camera angle labels for training

---

## 4. Custom Data Collection Guide

### 4.1 Recording Your Own Matches

**For Baseline/Side Camera Angles:**

#### Equipment
- **Camera:** Smartphone (iPhone 13+, Pixel 7+) or GoPro Hero 11+
- **Mount:** Tripod with fluid head (stable, no shake)
- **Position:**
  - Baseline: Behind baseline, elevated ~2-3m, centered
  - Side: Courtside at net height, perpendicular to net
- **Settings:**
  - Resolution: 1920×1080 minimum (4K better)
  - Frame rate: 30fps minimum (60fps better for fast serves)
  - Bitrate: High (less compression artifacts)
  - Shutter speed: 1/500s+ (reduce motion blur)

#### Recording Checklist
- [ ] Stable mount (test by shaking gently)
- [ ] Full court visible (or near-court for baseline)
- [ ] Good lighting (no backlighting, no harsh shadows)
- [ ] Record full matches (not just rallies)
- [ ] Include variety: different players, times of day, weather

#### How Much Data?

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Ball tracking | 3-5 matches (~5K frames) | 10+ matches (~20K frames) |
| Court detection | 20-30 frames per camera angle | 100 frames per angle |
| Shot classification | 500 shots | 2000 shots |

### 4.2 Annotating Ball Positions

**Tool:** CVAT (Computer Vision Annotation Tool)

#### Setup
```bash
# Install CVAT (Docker)
docker run -d -p 8080:8080 -v ~/cvat_data:/home/django/data cvat/server
docker run -d -p 8070:8070 cvat/ui

# Access: http://localhost:8080
```

#### Annotation Workflow

1. **Create Project:** "Tennis Ball Tracking - Baseline Camera"
2. **Upload Video:** Use "Create Task" → upload MP4
3. **Configure Labels:**
   - Shape: "Points" (not boxes)
   - Attributes: "visibility" (dropdown: 0=invisible, 1=visible, 2=occluded)
4. **Annotation Strategy:**
   - **Keyframe every 5-15 frames** (depending on trajectory complexity)
   - Use **linear interpolation** between keyframes
   - Mark ball as "Outside" when not visible (maps to visibility=0)
   - **Pro tip:** Annotate bounces first (keyframe), then fills between bounces
5. **Quality Control:**
   - Double-check at 0.5x speed
   - Verify bounces align with court surface
   - Check ball doesn't "teleport" between frames
6. **Export:**
   - Format: "CVAT for Video 1.1"
   - Extract XML → Convert to CSV (script below)

#### CVAT XML → CSV Converter

```python
# scripts/cvat_to_tracknet_csv.py

import xml.etree.ElementTree as ET
import pandas as pd

def cvat_xml_to_csv(xml_path, output_csv, fps=30):
    """
    Convert CVAT XML annotations to TrackNet CSV format

    Args:
        xml_path: Path to CVAT XML export
        output_csv: Output CSV path
        fps: Video frame rate (for frame number calculation)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []

    # Parse tracks
    for track in root.findall('.//track'):
        label = track.get('label')

        for point in track.findall('.//points'):
            frame = int(point.get('frame'))
            outside = int(point.get('outside', 0))

            # Get visibility attribute
            visibility_elem = point.find('.//attribute[@name="visibility"]')
            visibility = int(visibility_elem.text) if visibility_elem is not None else (0 if outside else 1)

            # Parse coordinates
            coords = point.get('points').split(',')
            x, y = float(coords[0]), float(coords[1])

            # If outside, set coords to 0
            if outside or visibility == 0:
                x, y = 0, 0

            annotations.append({
                'Frame': frame,
                'Visibility': visibility,
                'X': int(x),
                'Y': int(y)
            })

    # Convert to DataFrame and sort
    df = pd.DataFrame(annotations)
    df = df.sort_values('Frame').reset_index(drop=True)

    # Fill missing frames with visibility=0
    all_frames = pd.DataFrame({'Frame': range(df['Frame'].max() + 1)})
    df = all_frames.merge(df, on='Frame', how='left')
    df[['Visibility', 'X', 'Y']] = df[['Visibility', 'X', 'Y']].fillna(0).astype(int)

    # Save
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} frames to {output_csv}")

# Usage
cvat_xml_to_csv('annotations.xml', 'labels/match1.csv', fps=30)
```

#### Pre-Annotation Speedup

**Use existing TrackNet to pre-annotate:**

```python
# scripts/pre_annotate.py

def pre_annotate_with_tracknet(video_path, output_xml, model_path):
    """
    Run TrackNet inference → generate CVAT XML for manual correction

    This is 5-10x faster than annotating from scratch!
    """
    # Load model
    model = tracknet_v5()
    model.load_state_dict(torch.load(model_path))
    model.eval().cuda()

    # Run inference
    predictions = run_inference(video_path, model)  # From inference/

    # Convert predictions to CVAT XML format
    xml = create_cvat_xml(predictions, video_path)

    # Save
    with open(output_xml, 'w') as f:
        f.write(xml)

    print(f"Pre-annotation saved to {output_xml}")
    print("Import this into CVAT and manually correct errors!")

# Usage
pre_annotate_with_tracknet(
    'videos/baseline_match1.mp4',
    'pre_annotations/match1.xml',
    'checkpoints/tracknet_overhead.pt'  # Use best available model
)
```

**Expected accuracy of pre-annotations:**
- Overhead model on baseline camera: ~60-70% correct
- Manual correction time: ~30 min for 1000 frames (vs 2+ hours from scratch)

### 4.3 Annotating Court Corners

**Tool:** CVAT with 4-point polygon

#### Annotation Strategy

**Sampling:**
- Sample 1 frame every 30 seconds (~2 frames per minute)
- From 10 different matches → ~100-200 frames per camera angle
- Ensure variety: different lighting, court types (hard/clay), camera positions

**Labels:**
```
Shape: Polygon (4 points)
Order: top-left, top-right, bottom-right, bottom-left
```

**Baseline camera special case:**
If far court corners not visible:
- Annotate only visible corners (2-3 points)
- Use court line intersections to extrapolate missing corners (post-processing)

#### Export and Conversion

```python
# scripts/cvat_court_to_csv.py

def cvat_court_to_csv(xml_path, output_csv):
    """Convert CVAT court polygon annotations to CSV"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []

    for image in root.findall('.//image'):
        frame_id = int(image.get('id'))
        image_name = image.get('name')

        for polygon in image.findall('.//polygon'):
            points_str = polygon.get('points')
            points = [float(x) for x in points_str.replace(';', ',').split(',')]

            # Expect 8 values (4 points × 2 coords)
            if len(points) == 8:
                annotations.append({
                    'frame_id': frame_id,
                    'image_name': image_name,
                    'tl_x': points[0], 'tl_y': points[1],
                    'tr_x': points[2], 'tr_y': points[3],
                    'br_x': points[4], 'br_y': points[5],
                    'bl_x': points[6], 'bl_y': points[7]
                })

    df = pd.DataFrame(annotations)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} court annotations")

# Usage
cvat_court_to_csv('court_annotations.xml', 'data/court_labels.csv')
```

### 4.4 Annotating Shot Types (Optional - Phase 2)

**Tool:** CVAT with temporal segments

#### Labels
```
serve, forehand, backhand, volley, smash, slice, drop_shot, lob
```

#### Annotation
- Create "Track" (not individual frames)
- Mark start frame and end frame of each shot
- Assign label
- Export as action annotations

---

## 5. Training Procedures

### 5.1 Training TrackNet for New Camera Angles

#### Prepare Dataset

```python
# data/dataset.py - Extended for camera angles

class MultiAngleTrackNetDataset(Dataset):
    def __init__(self, data_dir, camera_angle='overhead', transform=None):
        """
        Args:
            data_dir: Root directory with subdirectories per angle
                data/
                  overhead/
                    match1_video.mp4
                    match1_labels.csv
                  baseline/
                    match2_video.mp4
                    match2_labels.csv
            camera_angle: 'overhead', 'elevated_side', 'baseline', 'all'
        """
        self.data_dir = data_dir
        self.camera_angle = camera_angle
        self.transform = transform

        # Load all videos and labels
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []

        angles = [self.camera_angle] if self.camera_angle != 'all' else \
                 ['overhead', 'elevated_side', 'baseline']

        for angle in angles:
            angle_dir = os.path.join(self.data_dir, angle)
            if not os.path.exists(angle_dir):
                continue

            # Find all video/label pairs
            videos = sorted(glob.glob(f"{angle_dir}/*_video.mp4"))
            for video_path in videos:
                label_path = video_path.replace('_video.mp4', '_labels.csv')
                if os.path.exists(label_path):
                    samples.append({
                        'video': video_path,
                        'labels': label_path,
                        'angle': angle
                    })

        return samples

    def __getitem__(self, idx):
        # Extract 3 consecutive frames
        # Generate heatmaps from labels
        # Apply transforms
        # (Implementation same as existing TrackNetDataset)
        pass
```

#### Training Script

```python
# scripts/train_tracknet_baseline.py

import torch
from models.tracknet import tracknet_v5
from data.dataset import MultiAngleTrackNetDataset
from training.trainer import TrackNetTrainer

def train_baseline_camera():
    # === CONFIG ===
    config = {
        'camera_angle': 'baseline',  # or 'all' for multi-angle
        'batch_size': 2,
        'epochs': 30,
        'lr': 1e-4,
        'checkpoint_dir': 'checkpoints/tracknet_baseline/'
    }

    # === DATA ===
    train_dataset = MultiAngleTrackNetDataset(
        data_dir='data/annotated/',
        camera_angle=config['camera_angle'],
        transform=get_train_transforms()  # Include PerspectiveAugmentation
    )

    val_dataset = MultiAngleTrackNetDataset(
        data_dir='data/annotated/',
        camera_angle=config['camera_angle'],
        transform=get_val_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # === MODEL ===
    model = tracknet_v5()

    # Optional: Initialize from overhead weights (transfer learning)
    if config['camera_angle'] != 'overhead':
        pretrained = torch.load('checkpoints/tracknet_overhead.pt')
        model.load_state_dict(pretrained, strict=False)
        print("Initialized from overhead model (transfer learning)")

    # === TRAINING ===
    trainer = TrackNetTrainer(model, config)
    trainer.train(train_loader, val_loader)

def get_train_transforms():
    return torchvision.transforms.v2.Compose([
        HorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        PerspectiveAugmentation(elevation_range=(0, 30), rotation_range=(-20, 20)),
        Mixup(alpha=0.2)
    ])

if __name__ == '__main__':
    train_baseline_camera()
```

#### Training Time Estimate

| Dataset Size | GPU | Training Time | Expected F1 |
|--------------|-----|---------------|-------------|
| 5K frames | RTX 3070 | ~2-3 hours | 0.85-0.90 |
| 20K frames | RTX 3070 | ~8-10 hours | 0.90-0.95 |
| 20K frames (multi-angle aug) | RTX 3070 | ~10-12 hours | 0.92-0.96 |

### 5.2 Training Court Detector

#### Prepare Dataset

```python
# data/court_dataset.py

class CourtKeypointDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        """
        Args:
            csv_path: CSV with columns [frame_id, image_name, tl_x, tl_y, ...]
            image_dir: Directory with frame images
        """
        self.labels = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        # Load image
        image_path = os.path.join(self.image_dir, row['image_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract keypoints
        keypoints = np.array([
            [row['tl_x'], row['tl_y']],
            [row['tr_x'], row['tr_y']],
            [row['br_x'], row['br_y']],
            [row['bl_x'], row['bl_y']]
        ], dtype=np.float32)

        # Normalize keypoints to [0, 1]
        h, w = image.shape[:2]
        keypoints[:, 0] /= w
        keypoints[:, 1] /= h

        # Apply transforms
        if self.transform:
            image, keypoints = self.transform(image, keypoints)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        keypoints = torch.from_numpy(keypoints).flatten()  # [8]

        return image, keypoints
```

#### Training Script

```python
# scripts/train_court_detector.py

from models.court_detector import CourtLineDetector
from data.court_dataset import CourtKeypointDataset

def train_court_detector():
    # === DATA ===
    train_dataset = CourtKeypointDataset(
        csv_path='data/court_labels_train.csv',
        image_dir='data/frames/',
        transform=get_court_transforms()
    )

    val_dataset = CourtKeypointDataset(
        csv_path='data/court_labels_val.csv',
        image_dir='data/frames/'
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # === MODEL ===
    model = CourtLineDetector(pretrained=True).cuda()

    # === OPTIMIZER ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # === LOSS ===
    criterion = nn.MSELoss()  # Or SmoothL1Loss for robustness

    # === TRAINING LOOP ===
    best_loss = float('inf')

    for epoch in range(30):
        model.train()
        train_loss = 0

        for images, keypoints in train_loader:
            images, keypoints = images.cuda(), keypoints.cuda()

            # Forward
            pred_keypoints = model(images).flatten(1)  # [B, 8]
            loss = criterion(pred_keypoints, keypoints)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, keypoints in val_loader:
                images, keypoints = images.cuda(), keypoints.cuda()
                pred_keypoints = model(images).flatten(1)
                loss = criterion(pred_keypoints, keypoints)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/30: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/court_detector_best.pt')

        scheduler.step()

def get_court_transforms():
    """Data augmentation for court detector"""
    return torchvision.transforms.v2.Compose([
        RandomRotation(degrees=5),  # Slight rotation
        ColorJitter(brightness=0.3, contrast=0.3),
        RandomResizedCrop(size=(720, 1280), scale=(0.8, 1.0))
    ])

if __name__ == '__main__':
    train_court_detector()
```

#### Training Time Estimate

| Dataset Size | GPU | Training Time | Expected Accuracy |
|--------------|-----|---------------|-------------------|
| 500 frames | RTX 3070 | ~30-45 min | 85-90% |
| 1000 frames | RTX 3070 | ~1 hour | 90-95% |

**Accuracy metric:** Mean pixel distance between predicted and ground truth corners (target: <10 pixels on 1920×1080 image)

---

## 6. Transfer Learning Strategies

### 6.1 Overhead → Baseline Transfer

**Strategy:** Pre-train on overhead data (abundant), fine-tune on baseline data (limited)

```python
# scripts/transfer_learning.py

def transfer_overhead_to_baseline():
    # Load overhead model
    model = tracknet_v5()
    model.load_state_dict(torch.load('checkpoints/tracknet_overhead.pt'))

    # Freeze encoder (keep learned features)
    for param in model.backbone.encoder.parameters():
        param.requires_grad = False

    # Fine-tune decoder only
    baseline_dataset = MultiAngleTrackNetDataset('data/', camera_angle='baseline')

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5  # Lower LR for fine-tuning
    )

    # Train for 10 epochs
    trainer = TrackNetTrainer(model, config={'epochs': 10})
    trainer.train(baseline_dataset)

    # Unfreeze all and train 5 more epochs
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    trainer = TrackNetTrainer(model, config={'epochs': 5})
    trainer.train(baseline_dataset)
```

**Expected improvement:**
- Baseline only (1K frames): F1 ~0.75
- Transfer learning (1K frames): F1 ~0.85-0.88
- Baseline only (5K frames): F1 ~0.88
- Transfer learning (5K frames): F1 ~0.92-0.94

### 6.2 Using Synthetic Data

**Generate synthetic camera angles from overhead datasets:**

```python
# scripts/generate_synthetic_baseline.py

from data.transforms import PerspectiveAugmentation

def augment_overhead_to_baseline(overhead_dataset, output_dir, num_synthetic=10000):
    """
    Generate synthetic baseline-view data from overhead dataset

    This can supplement limited real baseline data!
    """
    augmenter = PerspectiveAugmentation(
        elevation_range=(10, 30),  # Simulate baseline camera
        rotation_range=(-15, 15)
    )

    for i in range(num_synthetic):
        # Sample random frame from overhead dataset
        idx = np.random.randint(len(overhead_dataset))
        frame, ball_label = overhead_dataset[idx]

        # Apply perspective transform
        frame_synthetic, ball_synthetic = augmenter(frame, ball_label)

        # Save
        cv2.imwrite(f"{output_dir}/synthetic_{i:05d}.jpg", frame_synthetic)
        # Save label to CSV

    print(f"Generated {num_synthetic} synthetic baseline frames")
```

**Pros:** Can generate unlimited data
**Cons:** Domain gap between synthetic and real (lighting, compression artifacts, etc.)

---

## 7. Validation and Benchmarking

### 7.1 Hold-Out Test Sets

**Critical:** Never train on test data, even accidentally!

```
data/
  overhead/
    train/  (70% of data)
    val/    (15% of data)
    test/   (15% of data) ← NEVER TOUCH UNTIL FINAL EVAL
  baseline/
    train/
    val/
    test/   ← NEVER TOUCH
```

### 7.2 Evaluation Metrics

#### Ball Tracking
```python
def evaluate_ball_tracking(model, test_loader):
    """
    Metrics:
    - Precision, Recall, F1 (detection within 4px)
    - Mean pixel error (regression)
    - Tracking accuracy (% of frames correctly detected)
    """
    all_preds = []
    all_labels = []

    for frames, labels in test_loader:
        preds = model(frames)
        all_preds.extend(heatmap_to_position(preds))
        all_labels.extend(labels)

    # Compute metrics
    precision, recall, f1 = compute_detection_metrics(all_preds, all_labels, threshold=4)
    mean_error = np.mean([euclidean_distance(p, l) for p, l in zip(all_preds, all_labels)])

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mean_pixel_error': mean_error
    }
```

#### Court Detection
```python
def evaluate_court_detection(model, test_loader):
    """
    Metrics:
    - Mean corner error (pixels)
    - Percentage of frames with all corners <20px error
    """
    all_errors = []

    for images, keypoints_gt in test_loader:
        keypoints_pred = model(images)

        # Denormalize to pixel coords
        keypoints_pred = keypoints_pred * torch.tensor([width, height])
        keypoints_gt = keypoints_gt * torch.tensor([width, height])

        # Compute L2 error per corner
        errors = torch.norm(keypoints_pred - keypoints_gt, dim=1)
        all_errors.extend(errors.flatten().tolist())

    return {
        'mean_corner_error_px': np.mean(all_errors),
        'pct_good_frames': np.mean(np.array(all_errors) < 20) * 100
    }
```

### 7.3 Camera Angle Benchmarks

**Report accuracy per camera angle:**

| Model | Overhead F1 | Elevated Side F1 | Baseline F1 | Avg F1 |
|-------|-------------|------------------|-------------|--------|
| Overhead-only | 0.975 | 0.720 | 0.650 | 0.782 |
| Multi-angle Aug | 0.960 | 0.880 | 0.820 | 0.887 |
| Transfer Learning | 0.970 | 0.920 | 0.880 | 0.923 |

---

## 8. Deployment Considerations

### 8.1 Camera Angle Detection (Auto Mode)

```python
# inference/camera_detector.py

def detect_camera_angle(frame):
    """
    Auto-detect camera angle from single frame

    Heuristics:
    - Overhead: Court appears rectangular, minimal distortion
    - Elevated Side: Court has moderate perspective distortion
    - Baseline: Heavy perspective, far court very compressed
    """
    # Detect court lines
    court_mask = detect_court_lines_hough(frame)

    # Compute aspect ratio of detected court
    aspect_ratio = compute_court_aspect_ratio(court_mask)

    # Classify
    if 0.4 < aspect_ratio < 0.6:
        return 'overhead'
    elif 0.6 < aspect_ratio < 1.2:
        return 'elevated_side'
    else:
        return 'baseline'
```

### 8.2 Model Selection

```python
# main.py - Auto mode

def score_match(video_path, camera_angle='auto', **kwargs):
    """
    Args:
        camera_angle: 'overhead', 'elevated_side', 'baseline', or 'auto'
    """
    if camera_angle == 'auto':
        # Detect from first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        camera_angle = detect_camera_angle(frame)
        print(f"Detected camera angle: {camera_angle}")

    # Load appropriate models
    ball_tracker = tracknet_v5(camera_angle=camera_angle)
    # ... continue
```

---

## 9. Quick Start: Training Checklist

### Week 1: Data Collection
- [ ] Record 5 baseline-camera matches (or download existing)
- [ ] Extract 1 frame/second for court annotations (~300 frames)
- [ ] Set up CVAT (Docker or hosted)

### Week 2: Ball Tracking Annotations
- [ ] Annotate ball positions in 2 matches (~4K frames)
- [ ] Use pre-annotation script to speed up
- [ ] Convert CVAT XML to CSV
- [ ] Train baseline TrackNet model (transfer from overhead)

### Week 3: Court Detection
- [ ] Annotate court corners in 200 frames
- [ ] Train court detector
- [ ] Validate on test set (target: <15px error)

### Week 4: Integration
- [ ] Test full pipeline on held-out match
- [ ] Measure end-to-end accuracy
- [ ] Iterate on low-accuracy components

---

## 10. Resources

### Annotation Time Estimates
| Task | Frames | Time per Frame | Total Time |
|------|--------|----------------|------------|
| Ball tracking (from scratch) | 1000 | 90 sec | 25 hours |
| Ball tracking (with pre-annotation) | 1000 | 15 sec | 4 hours |
| Court corners | 200 | 20 sec | 1 hour |
| Shot classification | 500 | 30 sec | 4 hours |

### Download Links
- **TrackNet Tennis V1:** https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut
- **CoachAI Badminton:** https://nycu1-my.sharepoint.com/:u:/g/personal/tik_m365_nycu_edu_tw/EWisYhAiai9Ju7L-tQp0ykEBZJd9VQkKqsFrjcqqYIDP-g
- **CVAT:** https://github.com/opencv/cvat
- **OpenTTGames (table tennis, side view):** https://opentables.epfl.ch/

### Community Datasets (Request Access)
- **Tennis Shot Dataset:** Contact via ResearchGate (limited availability)
- **RacketVision (435K frames):** https://github.com/OrcustD/RacketVision (monitor for release)

---

**End of Training Guide**
