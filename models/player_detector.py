"""Player detector using YOLOv8-pose."""

from typing import Optional

import numpy as np


class PlayerDetector:
    """YOLO person detection for tracking players.

    Detects up to 2 players in a frame using YOLOv8-pose pretrained model.
    Supports stub mode for testing without YOLO installed.
    """

    def __init__(self, stub: bool = False, model_name: str = "yolov8x-pose.pt"):
        """Initialize player detector.

        Args:
            stub: If True, returns fake detections instead of running YOLO.
            model_name: YOLO model name (default: yolov8x-pose.pt).
        """
        self.stub = stub
        self.model_name = model_name
        self.model = None

        if not stub:
            try:
                from ultralytics import YOLO

                self.model = YOLO(model_name)
            except ImportError:
                raise ImportError(
                    "ultralytics package required for real player detection. "
                    "Install with: pip install ultralytics"
                )

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect players in a frame.

        Args:
            frame: BGR image (HWC uint8).

        Returns:
            Array of shape (N, 5) where N is number of detections (max 2).
            Each row is [x1, y1, x2, y2, confidence].
            Returns top 2 detections by confidence.
        """
        if self.stub:
            # Return 2 fake players for testing
            h, w = frame.shape[:2]
            return np.array(
                [
                    [w * 0.1, h * 0.2, w * 0.3, h * 0.9, 0.95],  # Player 1 (left)
                    [w * 0.7, h * 0.2, w * 0.9, h * 0.9, 0.93],  # Player 2 (right)
                ],
                dtype=np.float32,
            )

        # Real YOLO detection
        results = self.model(frame, verbose=False)

        # Extract person detections (class 0 in COCO dataset)
        boxes = results[0].boxes
        person_mask = boxes.cls == 0
        person_boxes = boxes[person_mask]

        if len(person_boxes) == 0:
            return np.array([], dtype=np.float32).reshape(0, 5)

        # Get bounding boxes and confidences
        xyxy = person_boxes.xyxy.cpu().numpy()  # (N, 4)
        conf = person_boxes.conf.cpu().numpy()  # (N,)

        # Combine into (N, 5)
        detections = np.column_stack([xyxy, conf])

        # Take top 2 by confidence
        if len(detections) > 2:
            top_indices = np.argsort(conf)[::-1][:2]  # Descending order, top 2
            detections = detections[top_indices]

        return detections.astype(np.float32)

    def detect_with_keypoints(self, frame: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Detect players with keypoints.

        Args:
            frame: BGR image (HWC uint8).

        Returns:
            Tuple of (detections, keypoints):
            - detections: Array of shape (N, 5) [x1, y1, x2, y2, confidence]
            - keypoints: Array of shape (N, 17, 3) [x, y, confidence] or None if stub mode
        """
        if self.stub:
            # Return fake detections without keypoints
            detections = self.detect(frame)
            return detections, None

        # Real YOLO detection
        results = self.model(frame, verbose=False)

        # Extract person detections
        boxes = results[0].boxes
        person_mask = boxes.cls == 0
        person_boxes = boxes[person_mask]

        if len(person_boxes) == 0:
            return np.array([], dtype=np.float32).reshape(0, 5), None

        # Get bounding boxes and confidences
        xyxy = person_boxes.xyxy.cpu().numpy()
        conf = person_boxes.conf.cpu().numpy()
        detections = np.column_stack([xyxy, conf])

        # Get keypoints if available
        keypoints = None
        if results[0].keypoints is not None:
            kpts = results[0].keypoints[person_mask]
            keypoints = kpts.data.cpu().numpy()  # (N, 17, 3)

            # Filter to top 2
            if len(detections) > 2:
                top_indices = np.argsort(conf)[::-1][:2]
                detections = detections[top_indices]
                keypoints = keypoints[top_indices]
        else:
            # No keypoints available, just take top 2 detections
            if len(detections) > 2:
                top_indices = np.argsort(conf)[::-1][:2]
                detections = detections[top_indices]

        return detections.astype(np.float32), keypoints
