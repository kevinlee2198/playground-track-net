"""Ball tracker wrapper for easy testing and integration."""

from typing import Optional

import cv2
import numpy as np
import torch

from inference.postprocess import heatmap_to_coordinates
from inference.video_preprocess import TARGET_HEIGHT, TARGET_WIDTH


class BallTrackerWrapper:
    """Wrapper around TrackNet for easy ball tracking.

    Supports both stub mode (for testing without a model) and real mode
    (with a TrackNet model checkpoint).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """Initialize ball tracker.

        Args:
            model_path: Path to TrackNet checkpoint (.pt file).
                       If None, runs in stub mode (returns fake positions).
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
            threshold: Heatmap detection threshold (default: 0.5).
        """
        self.model_path = model_path
        self.threshold = threshold

        if model_path is None:
            # Stub mode
            self.model = None
            self.device = None
        else:
            # Real mode - load model
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)

            from models.tracknet import TrackNet

            self.model = TrackNet()  # TrackNet V2 (9 channels, 3 frames)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

            # Handle both raw state_dict and checkpoint dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()

    def track_ball(
        self, frames: list[np.ndarray], orig_width: int, orig_height: int
    ) -> Optional[tuple[float, float, float]]:
        """Track ball in a sequence of frames.

        Args:
            frames: List of 3 consecutive frames (raw BGR frames from cv2, HWC uint8).
            orig_width: Original video width (for coordinate scaling).
            orig_height: Original video height (for coordinate scaling).

        Returns:
            (x, y, confidence) in original resolution, or None if no ball detected.
        """
        if len(frames) != 3:
            raise ValueError(f"Expected 3 frames, got {len(frames)}")

        if self.model is None:
            # Stub mode - return fake ball position
            return (orig_width / 2, orig_height / 2, 0.95)

        # Real mode - run TrackNet
        # Preprocess frames
        preprocessed = self._preprocess_frames(frames)

        # Run model inference
        with torch.no_grad():
            heatmaps = self.model(preprocessed)  # Shape: (1, 3, H, W)

        # Extract ball position from middle frame (frame index 1)
        heatmap = heatmaps[0, 1].cpu().numpy()  # (H, W)
        ball_pos = heatmap_to_coordinates(
            heatmap,
            orig_width=orig_width,
            orig_height=orig_height,
            threshold=self.threshold,
        )

        return ball_pos

    def _preprocess_frames(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Preprocess frames for TrackNet.

        Args:
            frames: List of 3 BGR frames (HWC uint8).

        Returns:
            Tensor of shape (1, 9, 288, 512) ready for TrackNet.
        """
        preprocessed = []
        for frame in frames:
            # Resize to 512x288
            resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            # Convert to CHW
            chw = np.transpose(normalized, (2, 0, 1))  # (3, 288, 512)
            preprocessed.append(chw)

        # Stack 3 frames along channel dimension: (9, 288, 512)
        stacked = np.concatenate(preprocessed, axis=0)

        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(stacked).unsqueeze(0)  # (1, 9, 288, 512)

        if self.device is not None:
            tensor = tensor.to(self.device)

        return tensor
