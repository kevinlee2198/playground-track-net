from pathlib import Path

import cv2
import numpy as np

TARGET_WIDTH = 512
TARGET_HEIGHT = 288


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize frame to 512x288, normalize to [0,1], return as (C, H, W) float32."""
    resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    normalized = resized.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1))  # HWC -> CHW


def extract_frames(source: str) -> tuple[list[np.ndarray], dict]:
    """Extract and preprocess frames from a video file or image directory.

    Args:
        source: Path to a video file (mp4/avi) or directory of images.

    Returns:
        Tuple of (list of preprocessed frames as CHW float32 arrays, metadata dict).
        Metadata contains original_width, original_height, frame_count.
    """
    source_path = Path(source)
    if source_path.is_dir():
        return _extract_from_directory(source_path)
    else:
        return _extract_from_video(source_path)


def _extract_from_directory(directory: Path) -> tuple[list[np.ndarray], dict]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = sorted(
        f for f in directory.iterdir() if f.suffix.lower() in extensions
    )
    if not image_files:
        raise ValueError(f"No image files found in {directory}")

    frames = []
    metadata = {}
    for i, img_path in enumerate(image_files):
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise ValueError(f"Failed to read image: {img_path}")
        if i == 0:
            h, w = frame.shape[:2]
            metadata = {
                "original_width": w,
                "original_height": h,
                "frame_count": len(image_files),
            }
        frames.append(preprocess_frame(frame))
    return frames, metadata


def _extract_from_video(video_path: Path) -> tuple[list[np.ndarray], dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata = {
        "original_width": w,
        "original_height": h,
        "frame_count": frame_count,
    }

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))
    cap.release()
    return frames, metadata


def create_sliding_windows(
    frames: list[np.ndarray], window_size: int = 3, stride: int = 3
) -> list[np.ndarray]:
    """Create sliding windows of concatenated frames with boundary padding.

    Stride=3 (MIMO): windows are [0,1,2], [3,4,5], etc.
    At boundaries, pad by duplicating the first/last frame.

    Args:
        frames: List of preprocessed frames, each shape (C, H, W).
        window_size: Number of frames per window.
        stride: Step between window start indices.

    Returns:
        List of concatenated frame windows, each shape (C*window_size, H, W).
    """
    n = len(frames)
    if n == 0:
        return []

    windows = []
    for start in range(0, n, stride):
        window_frames = []
        for i in range(window_size):
            idx = start + i
            if idx < 0:
                idx = 0
            elif idx >= n:
                idx = n - 1
            window_frames.append(frames[idx])
        concatenated = np.concatenate(window_frames, axis=0)
        windows.append(concatenated)
    return windows
