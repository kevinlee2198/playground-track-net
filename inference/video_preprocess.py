from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import closing
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

TARGET_WIDTH = 512
TARGET_HEIGHT = 288
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR frame to RGB (the order the model is trained on),
    resize to 512x288, normalize to [0,1], return as (C, H, W) float32."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (TARGET_WIDTH, TARGET_HEIGHT))
    normalized = resized.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1))  # HWC -> CHW


class SourceInfo(NamedTuple):
    """What ``infer`` needs to know about a source besides its frames."""

    width: int  # of the source, in pixels
    height: int
    fps: float  # 30.0 for an image directory or a video that reports none


def read_source_info(source: str | Path) -> SourceInfo:
    """The size and frame rate of a video file or image directory, without
    decoding the video. The size of a directory is that of its first image."""
    path = Path(source)
    if path.is_dir():
        first = _read_image(_image_files(path)[0])
        h, w = first.shape[:2]
        return SourceInfo(w, h, 30.0)
    cap = _open_video(path)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return SourceInfo(w, h, cap.get(cv2.CAP_PROP_FPS) or 30.0)
    finally:
        cap.release()


def iter_frames(source: str | Path) -> Generator[np.ndarray, None, None]:
    """The frames of a video file or image directory, preprocessed as by
    ``preprocess_frame``, decoded one at a time as they are asked for.

    A video yields every frame that decodes, until the first that does not; a
    directory yields its images in name order. Raises ValueError at once when
    the video cannot be opened or the directory holds no images.
    """
    return (preprocess_frame(frame) for frame in _decode(source))


def count_frames(source: str | Path) -> int:
    """The number of frames ``iter_frames`` yields, found by decoding them all
    (not from the container's frame count, which can be wrong)."""
    return sum(1 for _ in _decode(source))


class SourceEndedError(ValueError):
    """A source ended before a frame ``read_frames_at`` was asked for."""

    def __init__(self, source: str | Path, decoded: int, wanted: int):
        super().__init__(
            f"{source} ended after {decoded} frames, before frame {wanted}"
        )
        self.decoded = decoded  # the number of frames the source had


def read_frames_at(source: str | Path, indices: Sequence[int]) -> list[np.ndarray]:
    """The preprocessed frames at ``indices``, which must increase, in one pass
    that stops after the last of them. Raises SourceEndedError (a ValueError)
    if the source ends before it."""
    wanted = list(indices)
    chosen: list[np.ndarray] = []
    decoded = 0
    with closing(_decode(source)) as frames:
        while len(chosen) < len(wanted):
            frame = next(frames, None)
            if frame is None:
                raise SourceEndedError(source, decoded, wanted[len(chosen)])
            if decoded == wanted[len(chosen)]:
                chosen.append(preprocess_frame(frame))
            decoded += 1
    return chosen


def iter_windows(
    frames: Iterable[np.ndarray],
    window_size: int = 3,
    background: np.ndarray | None = None,
) -> Iterator[tuple[np.ndarray, int]]:
    """The windows ``create_sliding_windows(frames, window_size, stride=window_size,
    background=background)`` returns, built as the frames arrive and holding only
    one window's frames at a time.

    Each window comes with the number of the clip's frames in it: ``window_size``,
    except in a last, incomplete window, which is padded by repeating the clip's
    last frame.
    """
    pending: list[np.ndarray] = []
    for frame in frames:
        pending.append(frame)
        if len(pending) == window_size:
            yield _concatenate(pending, background), window_size
            pending = []
    if pending:
        real = len(pending)
        pending += [pending[-1]] * (window_size - real)
        yield _concatenate(pending, background), real


def _concatenate(frames: list[np.ndarray], background: np.ndarray | None) -> np.ndarray:
    if background is not None:
        frames = [*frames, background]
    return np.concatenate(frames, axis=0)


def _image_files(directory: Path) -> list[Path]:
    image_files = sorted(
        f for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No image files found in {directory}")
    return image_files


def _read_image(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Failed to read image: {path}")
    return frame


def _open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path}")
    return cap


def _decode(source: str | Path) -> Generator[np.ndarray, None, None]:
    """The source's frames as decoded (BGR uint8), one at a time. Checks the
    source before returning, so a bad one raises at the call."""
    path = Path(source)
    if path.is_dir():
        return (_read_image(image) for image in _image_files(path))
    return _decode_video(_open_video(path))


def _decode_video(cap: cv2.VideoCapture) -> Generator[np.ndarray, None, None]:
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                return
            yield frame
    finally:
        cap.release()


def extract_frames(source: str) -> tuple[list[np.ndarray], dict]:
    """Extract and preprocess every frame of a video file or image directory.

    This holds the whole clip in memory: 1.77 MB per frame (float32,
    3x288x512), so 6.4 GB for two minutes of 30 fps video. ``infer`` reads
    frames one at a time through ``iter_frames`` instead.

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
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    metadata = {
        "original_width": w,
        "original_height": h,
        "frame_count": frame_count,
        "fps": fps,
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
    frames: list[np.ndarray],
    window_size: int = 3,
    stride: int = 3,
    background: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Create sliding windows of concatenated frames with boundary padding.

    Stride=3 (MIMO): windows are [0,1,2], [3,4,5], etc.
    At boundaries, pad by duplicating the first/last frame.

    Args:
        frames: List of preprocessed frames, each shape (C, H, W).
        window_size: Number of frames per window.
        stride: Step between window start indices.
        background: Optional (C, H, W) image appended to every window after
            its frames, such as the clip's median background frame.

    Returns:
        List of concatenated frame windows, each shape (C*window_size, H, W),
        or (C*(window_size + 1), H, W) with a background.
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
        if background is not None:
            window_frames.append(background)
        concatenated = np.concatenate(window_frames, axis=0)
        windows.append(concatenated)
    return windows
