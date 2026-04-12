"""Simplified tracker using IoU matching.

This is a lightweight alternative to BoT-SORT for Week 3.
For production, consider using a full BoT-SORT implementation.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU (Intersection over Union) between two bounding boxes.

    Args:
        box1: Bounding box [x1, y1, x2, y2]
        box2: Bounding box [x1, y1, x2, y2]

    Returns:
        IoU score in [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


class SimpleTracker:
    """Simplified multi-object tracker using IoU matching.

    This tracker maintains persistent IDs for detected objects across frames
    using Hungarian algorithm for matching based on IoU.

    For production use, consider full BoT-SORT with:
    - Kalman filter for motion prediction
    - ReID features for appearance matching
    - More sophisticated track management
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        """Initialize tracker.

        Args:
            iou_threshold: Minimum IoU for matching detections to tracks.
            max_age: Maximum frames a track can be lost before deletion.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []
        self.next_id = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Update tracker with new detections.

        Args:
            detections: Array of shape (N, 4) or (N, 5) with bounding boxes.
                       Format: [x1, y1, x2, y2] or [x1, y1, x2, y2, conf]

        Returns:
            Tracked objects as array of shape (M, 5): [x1, y1, x2, y2, track_id]
        """
        # Handle empty detections
        if len(detections) == 0:
            # Age all tracks
            for track in self.tracks:
                track["age"] += 1

            # Remove old tracks
            self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

            # Return existing tracks
            if len(self.tracks) == 0:
                return np.array([], dtype=np.float32).reshape(0, 5)

            return np.array(
                [[*track["bbox"], track["id"]] for track in self.tracks],
                dtype=np.float32,
            )

        # Extract bounding boxes (ignore confidence if present)
        det_boxes = detections[:, :4]

        # Initialize tracks if empty
        if len(self.tracks) == 0:
            for det in det_boxes:
                self.tracks.append({"id": self.next_id, "bbox": det, "age": 0})
                self.next_id += 1

            return np.array(
                [[*track["bbox"], track["id"]] for track in self.tracks],
                dtype=np.float32,
            )

        # Match detections to existing tracks
        matched_pairs, unmatched_dets, unmatched_tracks = self._match(
            det_boxes, self.tracks
        )

        # Update matched tracks
        for det_idx, track_idx in matched_pairs:
            self.tracks[track_idx]["bbox"] = det_boxes[det_idx]
            self.tracks[track_idx]["age"] = 0

        # Age unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx]["age"] += 1

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self.tracks.append(
                {"id": self.next_id, "bbox": det_boxes[det_idx], "age": 0}
            )
            self.next_id += 1

        # Remove old tracks
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

        # Return active tracks (age == 0 means detected this frame)
        active_tracks = [t for t in self.tracks if t["age"] == 0]

        if len(active_tracks) == 0:
            return np.array([], dtype=np.float32).reshape(0, 5)

        return np.array(
            [[*track["bbox"], track["id"]] for track in active_tracks],
            dtype=np.float32,
        )

    def _match(
        self, detections: np.ndarray, tracks: list[dict]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Match detections to tracks using Hungarian algorithm on IoU.

        Args:
            detections: Array of shape (N, 4) with bounding boxes
            tracks: List of track dictionaries

        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                iou_matrix[i, j] = compute_iou(det, track["bbox"])

        # Hungarian algorithm (maximize IoU, so negate)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        # Filter matches by IoU threshold
        matched_pairs = []
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched_pairs.append((r, c))

        # Find unmatched detections
        matched_det_indices = set(r for r, _ in matched_pairs)
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det_indices]

        # Find unmatched tracks
        matched_track_indices = set(c for _, c in matched_pairs)
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track_indices]

        return matched_pairs, unmatched_dets, unmatched_tracks

    def reset(self):
        """Reset tracker (clear all tracks)."""
        self.tracks = []
        self.next_id = 0
