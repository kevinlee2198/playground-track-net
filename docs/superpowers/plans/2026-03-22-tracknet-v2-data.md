# TrackNet V2 Data Pipeline

| Field | Value |
|-------|-------|
| **Feature** | Data pipeline for TrackNet V2 ball tracking |
| **Goal** | Build heatmap generation, dataset loading, and augmentation pipeline that feeds the V2 U-Net backbone |
| **Architecture** | Pre-extracted frame images + CSV labels -> TrackNetDataset -> DataLoader -> model |
| **Tech Stack** | Python 3.12, PyTorch 2.10, torchvision 0.25, OpenCV, numpy, pytest |

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement each task — write failing test first, implement code, verify tests pass.

---

## File Map

| File | Purpose |
|------|---------|
| `data/__init__.py` | Package init, re-exports `TrackNetDataset`, `generate_heatmap` |
| `data/heatmap.py` | `generate_heatmap(x, y, visibility, height, width, radius)` — binary filled circle heatmap |
| `data/dataset.py` | `TrackNetDataset(Dataset)` — loads pre-extracted frames + CSV labels, sliding window stride=3 |
| `data/transforms.py` | Augmentations: `HorizontalFlip`, `ColorJitter`, `Mixup` |
| `tests/test_data.py` | All data pipeline tests |

---

## Task 1: Heatmap Generation

**Files:** `data/heatmap.py`, `tests/test_data.py`

### Steps

- [ ] Create `data/__init__.py` (empty for now)
- [ ] Write failing tests in `tests/test_data.py` for heatmap generation
- [ ] Implement `generate_heatmap()` in `data/heatmap.py`
- [ ] Verify tests pass
- [ ] Commit

### Test Code

```python
# tests/test_data.py
import torch
import pytest

from data.heatmap import generate_heatmap


class TestGenerateHeatmap:
    """Tests for the generate_heatmap function."""

    def test_visible_ball_returns_circle(self):
        """A visible ball should produce a filled circle at (x, y)."""
        heatmap = generate_heatmap(x=256, y=144, visibility=1, height=288, width=512, radius=30)
        assert heatmap.shape == (288, 512)
        assert heatmap.dtype == torch.float32
        # Center pixel should be 1
        assert heatmap[144, 256] == 1.0
        # Far corner should be 0
        assert heatmap[0, 0] == 0.0

    def test_invisible_ball_returns_zeros(self):
        """visibility=0 should produce an all-zero heatmap."""
        heatmap = generate_heatmap(x=0, y=0, visibility=0, height=288, width=512, radius=30)
        assert heatmap.shape == (288, 512)
        assert heatmap.sum() == 0.0

    def test_partially_occluded_still_labeled(self):
        """visibility=2 (partially occluded) should still produce a circle."""
        heatmap = generate_heatmap(x=100, y=100, visibility=2, height=288, width=512, radius=30)
        assert heatmap[100, 100] == 1.0
        assert heatmap.sum() > 0.0

    def test_circle_radius(self):
        """Pixels at distance <= radius should be 1, pixels far away should be 0."""
        heatmap = generate_heatmap(x=256, y=144, visibility=1, height=288, width=512, radius=30)
        # Pixel at exactly radius distance along x-axis should be 1
        assert heatmap[144, 256 + 30] == 1.0
        # Pixel just beyond radius should be 0
        assert heatmap[144, 256 + 31] == 0.0

    def test_ball_at_edge_clips_to_image(self):
        """Ball near image border should not crash; circle is clipped."""
        heatmap = generate_heatmap(x=5, y=5, visibility=1, height=288, width=512, radius=30)
        assert heatmap.shape == (288, 512)
        assert heatmap[5, 5] == 1.0
        # Should have fewer lit pixels than a centered ball
        centered = generate_heatmap(x=256, y=144, visibility=1, height=288, width=512, radius=30)
        assert heatmap.sum() < centered.sum()

    def test_values_are_binary(self):
        """All values should be exactly 0.0 or 1.0."""
        heatmap = generate_heatmap(x=200, y=100, visibility=1, height=288, width=512, radius=30)
        unique_vals = torch.unique(heatmap)
        assert all(v in (0.0, 1.0) for v in unique_vals)
```

### Implementation Code

```python
# data/heatmap.py
import torch


def generate_heatmap(
    x: int,
    y: int,
    visibility: int,
    height: int = 288,
    width: int = 512,
    radius: int = 30,
) -> torch.Tensor:
    """Generate a binary heatmap with a filled circle at the ball position.

    Args:
        x: Ball x-coordinate (pixel).
        y: Ball y-coordinate (pixel).
        visibility: 0 = invisible (all-zero), 1 = visible, 2 = partially occluded.
        height: Heatmap height.
        width: Heatmap width.
        radius: Circle radius in pixels.

    Returns:
        Float32 tensor of shape (height, width) with values 0.0 or 1.0.
    """
    if visibility == 0:
        return torch.zeros(height, width, dtype=torch.float32)

    ys = torch.arange(height, dtype=torch.float32).unsqueeze(1)  # (H, 1)
    xs = torch.arange(width, dtype=torch.float32).unsqueeze(0)   # (1, W)
    dist_sq = (xs - x) ** 2 + (ys - y) ** 2
    heatmap = (dist_sq <= radius**2).float()
    return heatmap
```

### Verify

```bash
uv run pytest tests/test_data.py::TestGenerateHeatmap -v
```

---

## Task 2: Test Fixtures (Synthetic Frame Directory + CSV Labels)

**Files:** `tests/test_data.py`, `tests/conftest.py`

### Steps

- [ ] Create `tests/__init__.py` (empty)
- [ ] Create `tests/conftest.py` with pytest fixtures that generate a temp directory of synthetic frame images and a CSV label file
- [ ] Verify fixtures work by running a trivial test
- [ ] Commit

### Implementation Code

```python
# tests/conftest.py
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_frames_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Create a directory with 9 synthetic 512x288 frames and a CSV label file.

    Returns (frames_dir, csv_path).
    Frame filenames: 00000.jpg, 00001.jpg, ..., 00008.jpg
    Labels: 9 rows, balls at known positions, frame 3 invisible.
    """
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    rng = np.random.RandomState(42)

    # Known ball positions for 9 frames
    labels = [
        # (frame, visibility, x, y)
        (0, 1, 100, 80),
        (1, 1, 110, 85),
        (2, 1, 120, 90),
        (3, 0, 0, 0),       # invisible
        (4, 1, 140, 100),
        (5, 1, 150, 105),
        (6, 1, 160, 110),
        (7, 2, 170, 115),   # partially occluded
        (8, 1, 180, 120),
    ]

    for frame_idx, vis, bx, by in labels:
        # Create a random image so frames are distinct
        img = rng.randint(0, 256, (288, 512, 3), dtype=np.uint8)
        # Draw a small white circle at ball position if visible
        if vis > 0:
            cv2.circle(img, (bx, by), 5, (255, 255, 255), -1)
        filename = f"{frame_idx:05d}.jpg"
        cv2.imwrite(str(frames_dir / filename), img)

    # Write CSV
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Visibility", "X", "Y"])
        for row in labels:
            writer.writerow(row)

    return frames_dir, csv_path
```

### Verify

```bash
uv run pytest tests/conftest.py --co -v
```

---

## Task 3: TrackNetDataset — Basic Loading

**Files:** `data/dataset.py`, `tests/test_data.py`

### Steps

- [ ] Write failing tests for dataset `__len__` and `__getitem__` shape/dtype
- [ ] Implement `TrackNetDataset` in `data/dataset.py`
- [ ] Verify tests pass
- [ ] Commit

### Test Code

```python
# tests/test_data.py (append to existing)
import numpy as np
from data.dataset import TrackNetDataset


class TestTrackNetDataset:
    """Tests for TrackNetDataset loading."""

    def test_length_with_nine_frames(self, sample_frames_dir):
        """9 frames with stride=3 should give 3 samples."""
        frames_dir, csv_path = sample_frames_dir
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        assert len(ds) == 3

    def test_getitem_shapes(self, sample_frames_dir):
        """Each sample should return frames (9, 288, 512) and heatmaps (3, 288, 512)."""
        frames_dir, csv_path = sample_frames_dir
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        frames, heatmaps = ds[0]
        assert frames.shape == (9, 288, 512)
        assert heatmaps.shape == (3, 288, 512)

    def test_getitem_dtypes(self, sample_frames_dir):
        """Frames and heatmaps should be float32."""
        frames_dir, csv_path = sample_frames_dir
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        frames, heatmaps = ds[0]
        assert frames.dtype == torch.float32
        assert heatmaps.dtype == torch.float32

    def test_frame_values_normalized(self, sample_frames_dir):
        """Frame values should be in [0, 1]."""
        frames_dir, csv_path = sample_frames_dir
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        frames, _ = ds[0]
        assert frames.min() >= 0.0
        assert frames.max() <= 1.0

    def test_invisible_frame_heatmap_is_zeros(self, sample_frames_dir):
        """Sample idx=1 maps to frames [3,4,5]. Frame 3 is invisible -> heatmap[0] all zeros."""
        frames_dir, csv_path = sample_frames_dir
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        _, heatmaps = ds[1]
        assert heatmaps[0].sum() == 0.0  # frame 3 is invisible
        assert heatmaps[1].sum() > 0.0   # frame 4 is visible

    def test_all_samples_accessible(self, sample_frames_dir):
        """Iterate all samples without error."""
        frames_dir, csv_path = sample_frames_dir
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        for i in range(len(ds)):
            frames, heatmaps = ds[i]
            assert frames.shape == (9, 288, 512)
            assert heatmaps.shape == (3, 288, 512)
```

### Implementation Code

```python
# data/dataset.py
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.heatmap import generate_heatmap


class TrackNetDataset(Dataset):
    """Dataset for TrackNet V2: loads pre-extracted frames + CSV labels.

    Each sample returns 3 consecutive frames (9 channels) and 3 heatmaps.
    Sliding window stride = 3 (MIMO).

    Args:
        frames_dir: Directory containing frame images (e.g. 00000.jpg).
        label_path: CSV file with columns Frame, Visibility, X, Y.
        height: Target frame height.
        width: Target frame width.
        radius: Heatmap circle radius.
        transform: Optional callable applied to (frames, heatmaps).
    """

    def __init__(
        self,
        frames_dir: str | Path,
        label_path: str | Path,
        height: int = 288,
        width: int = 512,
        radius: int = 30,
        transform=None,
    ):
        self.frames_dir = Path(frames_dir)
        self.height = height
        self.width = width
        self.radius = radius
        self.transform = transform

        # Load labels from CSV
        self.labels: list[dict] = []
        with open(label_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.labels.append(
                    {
                        "frame": int(row["Frame"]),
                        "visibility": int(row["Visibility"]),
                        "x": int(row["X"]),
                        "y": int(row["Y"]),
                    }
                )
        # Sort by frame index
        self.labels.sort(key=lambda r: r["frame"])

        # Build sorted list of frame image paths
        self.frame_paths = sorted(self.frames_dir.glob("*.*"))
        self.num_frames = len(self.frame_paths)

        # Number of samples: ceil(num_frames / 3)
        self._num_samples = (self.num_frames + 2) // 3

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * 3
        frame_indices = [start, start + 1, start + 2]

        # Clamp indices to valid range (boundary padding by duplication)
        frame_indices = [min(max(i, 0), self.num_frames - 1) for i in frame_indices]

        # Load and preprocess frames
        frame_tensors = []
        for fi in frame_indices:
            img = cv2.imread(str(self.frame_paths[fi]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != (self.height, self.width):
                img = cv2.resize(img, (self.width, self.height))
            # Normalize to [0, 1] and convert to (3, H, W) tensor
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            frame_tensors.append(tensor)

        # Concatenate 3 frames -> (9, H, W)
        frames = torch.cat(frame_tensors, dim=0)

        # Generate heatmaps
        heatmap_list = []
        for fi in frame_indices:
            label = self.labels[fi]
            hm = generate_heatmap(
                x=label["x"],
                y=label["y"],
                visibility=label["visibility"],
                height=self.height,
                width=self.width,
                radius=self.radius,
            )
            heatmap_list.append(hm)

        # Stack heatmaps -> (3, H, W)
        heatmaps = torch.stack(heatmap_list, dim=0)

        if self.transform is not None:
            frames, heatmaps = self.transform(frames, heatmaps)

        return frames, heatmaps
```

### Verify

```bash
uv run pytest tests/test_data.py::TestTrackNetDataset -v
```

---

## Task 4: Boundary Padding

**Files:** `tests/test_data.py`, `data/dataset.py`

### Steps

- [ ] Write failing tests for boundary padding behavior (fewer than 3 frames, non-multiple-of-3 frame counts)
- [ ] Verify existing implementation handles it (it should via clamping)
- [ ] Add edge-case fixture and tests
- [ ] Commit

### Test Code

```python
# tests/test_data.py (append)

class TestDatasetBoundaryPadding:
    """Tests for boundary handling when frame count is not a multiple of 3."""

    def test_single_frame_dataset(self, tmp_path):
        """A single frame should produce 1 sample by duplicating the frame."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        img = np.random.randint(0, 256, (288, 512, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "00000.jpg"), img)

        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Visibility", "X", "Y"])
            writer.writerow([0, 1, 256, 144])

        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        assert len(ds) == 1
        frames, heatmaps = ds[0]
        assert frames.shape == (9, 288, 512)
        # All 3 frame slots should be identical (same single frame duplicated)
        assert torch.equal(frames[:3], frames[3:6])
        assert torch.equal(frames[:3], frames[6:9])

    def test_four_frames_produces_two_samples(self, tmp_path):
        """4 frames -> ceil(4/3) = 2 samples. Second sample pads last frame."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        labels = []
        for i in range(4):
            img = np.random.randint(0, 256, (288, 512, 3), dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), img)
            labels.append((i, 1, 100 + i * 10, 80 + i * 5))

        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Visibility", "X", "Y"])
            for row in labels:
                writer.writerow(row)

        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        assert len(ds) == 2
        # Second sample: frames [3, 4(clamped to 3), 5(clamped to 3)]
        frames, heatmaps = ds[1]
        assert frames.shape == (9, 288, 512)
        # Frames at positions 1 and 2 of this sample should be identical (both clamped to frame 3)
        assert torch.equal(frames[3:6], frames[6:9])
```

### Verify

```bash
uv run pytest tests/test_data.py::TestDatasetBoundaryPadding -v
```

---

## Task 5: Horizontal Flip Transform

**Files:** `data/transforms.py`, `tests/test_data.py`

### Steps

- [ ] Write failing tests for horizontal flip
- [ ] Implement `HorizontalFlip` in `data/transforms.py`
- [ ] Verify tests pass
- [ ] Commit

### Test Code

```python
# tests/test_data.py (append)
from data.transforms import HorizontalFlip


class TestHorizontalFlip:
    """Tests for the horizontal flip augmentation."""

    def test_flip_reverses_width_dimension(self):
        """Flipping should reverse the last dimension (width)."""
        frames = torch.arange(9 * 4 * 6, dtype=torch.float32).reshape(9, 4, 6)
        heatmaps = torch.arange(3 * 4 * 6, dtype=torch.float32).reshape(3, 4, 6)
        flip = HorizontalFlip(p=1.0)
        f_frames, f_heatmaps = flip(frames, heatmaps)
        assert torch.equal(f_frames, frames.flip(-1))
        assert torch.equal(f_heatmaps, heatmaps.flip(-1))

    def test_no_flip_when_p_zero(self):
        """p=0 should never flip."""
        frames = torch.randn(9, 288, 512)
        heatmaps = torch.randn(3, 288, 512)
        flip = HorizontalFlip(p=0.0)
        f_frames, f_heatmaps = flip(frames, heatmaps)
        assert torch.equal(f_frames, frames)
        assert torch.equal(f_heatmaps, heatmaps)

    def test_shapes_preserved(self):
        """Output shapes should match input shapes."""
        frames = torch.randn(9, 288, 512)
        heatmaps = torch.randn(3, 288, 512)
        flip = HorizontalFlip(p=1.0)
        f_frames, f_heatmaps = flip(frames, heatmaps)
        assert f_frames.shape == frames.shape
        assert f_heatmaps.shape == heatmaps.shape
```

### Implementation Code

```python
# data/transforms.py
from __future__ import annotations

import torch
import torchvision.transforms.v2 as T  # v2 is the recommended API for torchvision 0.25+


class HorizontalFlip:
    """Randomly flip frames and heatmaps horizontally.

    Args:
        p: Probability of flipping.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, frames: torch.Tensor, heatmaps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < self.p:
            frames = frames.flip(-1)
            heatmaps = heatmaps.flip(-1)
        return frames, heatmaps
```

### Verify

```bash
uv run pytest tests/test_data.py::TestHorizontalFlip -v
```

---

## Task 6: Color Jitter Transform

**Files:** `data/transforms.py`, `tests/test_data.py`

### Steps

- [ ] Write failing tests for color jitter
- [ ] Implement `FrameColorJitter` in `data/transforms.py`
- [ ] Verify tests pass
- [ ] Commit

### Test Code

```python
# tests/test_data.py (append)
from data.transforms import FrameColorJitter


class TestFrameColorJitter:
    """Tests for the color jitter augmentation applied to frames only."""

    def test_heatmaps_unchanged(self):
        """Color jitter should not modify heatmaps."""
        frames = torch.rand(9, 288, 512)
        heatmaps = torch.rand(3, 288, 512)
        jitter = FrameColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        _, out_heatmaps = jitter(frames, heatmaps)
        assert torch.equal(out_heatmaps, heatmaps)

    def test_frames_modified(self):
        """With non-zero jitter params, frames should (usually) change."""
        torch.manual_seed(0)
        frames = torch.full((9, 288, 512), 0.5)
        heatmaps = torch.zeros(3, 288, 512)
        jitter = FrameColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        out_frames, _ = jitter(frames, heatmaps)
        # Not guaranteed to differ every time, but with these params very likely
        assert out_frames.shape == frames.shape

    def test_output_clamped_to_01(self):
        """Output frame values should remain in [0, 1]."""
        frames = torch.rand(9, 288, 512)
        heatmaps = torch.zeros(3, 288, 512)
        jitter = FrameColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        out_frames, _ = jitter(frames, heatmaps)
        assert out_frames.min() >= 0.0
        assert out_frames.max() <= 1.0
```

### Implementation Code

```python
# data/transforms.py (append)

class FrameColorJitter:
    """Apply torchvision ColorJitter to each frame independently.

    Frames tensor has shape (9, H, W) = 3 frames x 3 RGB channels.
    ColorJitter is applied per-frame (each 3-channel slice).

    Heatmaps are not modified.

    Args:
        brightness: ColorJitter brightness param.
        contrast: ColorJitter contrast param.
        saturation: ColorJitter saturation param.
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
    ):
        self.jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation
        )

    def __call__(
        self, frames: torch.Tensor, heatmaps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # frames shape: (9, H, W) -> split into 3 frames of (3, H, W)
        frame_list = frames.chunk(3, dim=0)
        jittered = []
        for frame in frame_list:
            frame = self.jitter(frame)
            frame = frame.clamp(0.0, 1.0)
            jittered.append(frame)
        frames = torch.cat(jittered, dim=0)
        return frames, heatmaps
```

### Verify

```bash
uv run pytest tests/test_data.py::TestFrameColorJitter -v
```

---

## Task 7: Mixup Transform

**Files:** `data/transforms.py`, `tests/test_data.py`

### Steps

- [ ] Write failing tests for mixup
- [ ] Implement `Mixup` in `data/transforms.py`
- [ ] Verify tests pass
- [ ] Commit

### Test Code

```python
# tests/test_data.py (append)
from data.transforms import Mixup


class TestMixup:
    """Tests for the mixup augmentation."""

    def test_mixup_blends_frames_and_heatmaps(self):
        """Mixup should blend both frames and heatmaps with the same lambda."""
        torch.manual_seed(42)
        frames_a = torch.ones(9, 4, 6)
        heatmaps_a = torch.ones(3, 4, 6)
        frames_b = torch.zeros(9, 4, 6)
        heatmaps_b = torch.zeros(3, 4, 6)

        mixup = Mixup(alpha=1.0)
        f_out, h_out = mixup(frames_a, heatmaps_a, frames_b, heatmaps_b)

        # Result should be between 0 and 1 (blended)
        assert f_out.min() >= 0.0
        assert f_out.max() <= 1.0
        # Frames and heatmaps should use the same blend ratio
        lam = f_out[0, 0, 0].item()
        assert torch.allclose(f_out, torch.full_like(f_out, lam))
        assert torch.allclose(h_out, torch.full_like(h_out, lam))

    def test_mixup_output_shapes(self):
        """Output shapes should match input shapes."""
        frames_a = torch.randn(9, 288, 512)
        heatmaps_a = torch.randn(3, 288, 512)
        frames_b = torch.randn(9, 288, 512)
        heatmaps_b = torch.randn(3, 288, 512)
        mixup = Mixup(alpha=1.0)
        f_out, h_out = mixup(frames_a, heatmaps_a, frames_b, heatmaps_b)
        assert f_out.shape == (9, 288, 512)
        assert h_out.shape == (3, 288, 512)

    def test_mixup_lambda_from_beta_distribution(self):
        """Lambda should be drawn from Beta(alpha, alpha)."""
        torch.manual_seed(0)
        mixup = Mixup(alpha=1.0)
        frames_a = torch.ones(9, 2, 2)
        heatmaps_a = torch.ones(3, 2, 2)
        frames_b = torch.zeros(9, 2, 2)
        heatmaps_b = torch.zeros(3, 2, 2)

        lambdas = []
        for _ in range(100):
            f_out, _ = mixup(frames_a, heatmaps_a, frames_b, heatmaps_b)
            lambdas.append(f_out[0, 0, 0].item())

        # Beta(1,1) is uniform [0,1] — mean should be ~0.5
        mean_lam = sum(lambdas) / len(lambdas)
        assert 0.3 < mean_lam < 0.7
```

### Implementation Code

```python
# data/transforms.py (append)

class Mixup:
    """Mixup augmentation: blend two samples with a random lambda from Beta(alpha, alpha).

    Args:
        alpha: Beta distribution parameter. alpha=1.0 gives uniform [0,1].
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(
        self,
        frames_a: torch.Tensor,
        heatmaps_a: torch.Tensor,
        frames_b: torch.Tensor,
        heatmaps_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = torch.distributions.Beta(self.alpha, self.alpha)
        lam = dist.sample().item()
        frames = lam * frames_a + (1 - lam) * frames_b
        heatmaps = lam * heatmaps_a + (1 - lam) * heatmaps_b
        return frames, heatmaps
```

### Verify

```bash
uv run pytest tests/test_data.py::TestMixup -v
```

---

## Task 8: Compose Transform + Dataset Integration

**Files:** `data/transforms.py`, `data/dataset.py`, `tests/test_data.py`

### Steps

- [ ] Write failing tests for compose and dataset-with-transform
- [ ] Implement `Compose` in `data/transforms.py`
- [ ] Verify tests pass
- [ ] Commit

### Test Code

```python
# tests/test_data.py (append)
from data.transforms import Compose


class TestCompose:
    """Tests for composing multiple transforms."""

    def test_compose_applies_in_order(self):
        """Compose should apply transforms sequentially."""
        flip = HorizontalFlip(p=1.0)
        compose = Compose([flip, flip])  # flip twice = identity
        frames = torch.randn(9, 4, 6)
        heatmaps = torch.randn(3, 4, 6)
        f_out, h_out = compose(frames, heatmaps)
        assert torch.allclose(f_out, frames)
        assert torch.allclose(h_out, heatmaps)


class TestDatasetWithTransform:
    """Tests for dataset with transforms applied."""

    def test_transform_is_called(self, sample_frames_dir):
        """When a transform is provided, it should be applied to the output."""
        frames_dir, csv_path = sample_frames_dir
        flip = HorizontalFlip(p=1.0)
        ds_no_flip = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        ds_flip = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path, transform=flip)

        frames_orig, heatmaps_orig = ds_no_flip[0]
        frames_flip, heatmaps_flip = ds_flip[0]

        assert torch.equal(frames_flip, frames_orig.flip(-1))
        assert torch.equal(heatmaps_flip, heatmaps_orig.flip(-1))
```

### Implementation Code

```python
# data/transforms.py (append)

class Compose:
    """Compose multiple (frames, heatmaps) transforms sequentially.

    Args:
        transforms: List of callables that accept and return (frames, heatmaps).
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(
        self, frames: torch.Tensor, heatmaps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            frames, heatmaps = t(frames, heatmaps)
        return frames, heatmaps
```

### Verify

```bash
uv run pytest tests/test_data.py::TestCompose tests/test_data.py::TestDatasetWithTransform -v
```

---

## Task 9: Package Init + Exports

**Files:** `data/__init__.py`

### Steps

- [ ] Update `data/__init__.py` with public exports
- [ ] Write a smoke test for imports
- [ ] Commit

### Implementation Code

```python
# data/__init__.py
from data.dataset import TrackNetDataset
from data.heatmap import generate_heatmap
from data.transforms import Compose, FrameColorJitter, HorizontalFlip, Mixup

__all__ = [
    "TrackNetDataset",
    "generate_heatmap",
    "Compose",
    "FrameColorJitter",
    "HorizontalFlip",
    "Mixup",
]
```

### Test Code

```python
# tests/test_data.py (append)

class TestPackageImports:
    """Verify public API is importable from the data package."""

    def test_imports(self):
        from data import (
            TrackNetDataset,
            generate_heatmap,
            Compose,
            FrameColorJitter,
            HorizontalFlip,
            Mixup,
        )
        assert TrackNetDataset is not None
        assert generate_heatmap is not None
```

### Verify

```bash
uv run pytest tests/test_data.py::TestPackageImports -v
```

---

## Task 10: DataLoader Integration Test

**Files:** `tests/test_data.py`

### Steps

- [ ] Write a test that creates a DataLoader from TrackNetDataset with the spec-required config
- [ ] Verify batch shapes match what the model expects: input `(batch, 9, 288, 512)`, targets `(batch, 3, 288, 512)`
- [ ] Commit

### Test Code

```python
# tests/test_data.py (append)
from torch.utils.data import DataLoader


class TestDataLoader:
    """Integration test: DataLoader produces correct batch shapes."""

    def test_batch_shapes(self, sample_frames_dir):
        """DataLoader batches should match model expected input shapes."""
        frames_dir, csv_path = sample_frames_dir
        ds = TrackNetDataset(frames_dir=frames_dir, label_path=csv_path)
        loader = DataLoader(
            ds,
            batch_size=2,
            pin_memory=False,  # no GPU in tests
            num_workers=0,     # simpler for tests
        )
        batch_frames, batch_heatmaps = next(iter(loader))
        assert batch_frames.shape == (2, 9, 288, 512)
        assert batch_heatmaps.shape == (2, 3, 288, 512)
        assert batch_frames.dtype == torch.float32
        assert batch_heatmaps.dtype == torch.float32
```

### Verify

```bash
uv run pytest tests/test_data.py::TestDataLoader -v
```

---

## Full Test Suite

After all tasks are complete, run the entire test suite:

```bash
uv run pytest tests/test_data.py -v
```

All tests should pass. The data pipeline is then ready for integration with the model and training subsystems.
