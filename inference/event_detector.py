"""Event detector for ball sports (tennis, badminton, pickleball).

Detects game events from ball trajectory and player positions:
- Bounce: Ball hits the ground (y-velocity reversal)
- Hit: Ball comes into contact with player
- Serve: Special case of hit at rally start
- Rally end: Ball goes out or multiple bounces
"""

from typing import Optional

import numpy as np


class EventDetector:
    """Detect game events from ball and player tracking data.

    Uses heuristic-based detection:
    - Bounce: Velocity reversal in y-direction
    - Hit: Ball proximity to player bounding box
    - Serve: First hit of rally
    - Rally end: Out of bounds or invalid bounce
    """

    def __init__(self, sport: str = "tennis", hit_distance_threshold: float = 50.0):
        """Initialize event detector.

        Args:
            sport: Sport type ('tennis', 'badminton', 'pickleball')
            hit_distance_threshold: Maximum distance (pixels) between ball and player for hit detection
        """
        self.sport = sport
        self.hit_distance_threshold = hit_distance_threshold

        # State tracking
        self.ball_history: list[tuple[int, Optional[tuple[float, float]]]] = []
        self.last_bounce_frame: Optional[int] = None
        self.last_hit_frame: Optional[int] = None
        self.last_hit_player: Optional[int] = None
        self.rally_active = False
        self.rally_start_frame: Optional[int] = None

        # Event detection parameters
        self.bounce_velocity_threshold = 5.0  # Minimum y-velocity change for bounce
        self.max_history_length = 10  # Frames to keep in history

    def process_frame(
        self,
        frame_id: int,
        ball_pos: Optional[tuple[float, float]],
        player_tracks: np.ndarray,
    ) -> list[dict]:
        """Process a single frame and detect events.

        Args:
            frame_id: Current frame number
            ball_pos: Ball position (x, y) or None if not detected
            player_tracks: Array of shape (N, 5) with player tracks [x1, y1, x2, y2, track_id]

        Returns:
            List of event dictionaries. Each event has:
            - type: 'bounce', 'hit', 'serve_start', 'rally_end'
            - frame: Frame number
            - Additional event-specific data
        """
        events = []

        # Update ball history
        self.ball_history.append((frame_id, ball_pos))
        if len(self.ball_history) > self.max_history_length:
            self.ball_history.pop(0)

        # Only detect events if ball is visible
        if ball_pos is None:
            return events

        # Detect bounce
        bounce_detected = self._detect_bounce()
        if bounce_detected:
            events.append({
                "type": "bounce",
                "frame": frame_id,
                "data": {
                    "position": ball_pos,
                    "zone": "unknown",  # Court zone detection not implemented yet
                    "valid": True,  # Assume valid for now (no court boundaries)
                },
            })
            self.last_bounce_frame = frame_id

        # Detect hit
        hit_player = self._detect_hit(ball_pos, player_tracks, frame_id)
        if hit_player is not None:
            # Check if this is a serve (first hit of rally)
            if not self.rally_active:
                events.append({
                    "type": "serve_start",
                    "frame": frame_id,
                    "player_id": hit_player,
                    "data": {
                        "is_first_serve": True,  # Simplified (no fault tracking yet)
                    },
                })
                self.rally_active = True
                self.rally_start_frame = frame_id
            else:
                events.append({
                    "type": "hit",
                    "frame": frame_id,
                    "player_id": hit_player,
                    "data": {
                        "shot_type": "groundstroke",  # Simplified (no shot classification)
                    },
                })

            self.last_hit_frame = frame_id
            self.last_hit_player = hit_player

        return events

    def _detect_bounce(self) -> bool:
        """Detect bounce from y-velocity reversal.

        Returns:
            True if bounce detected in recent history
        """
        if len(self.ball_history) < 5:
            return False

        # Extract recent positions (only non-None)
        recent_positions = [
            (frame, pos) for frame, pos in self.ball_history[-5:]
            if pos is not None
        ]

        if len(recent_positions) < 5:
            return False

        # Extract y-coordinates
        y_coords = [pos[1] for _, pos in recent_positions]

        # Compute velocities (dy/dt)
        velocities = np.diff(y_coords)

        if len(velocities) < 2:
            return False

        # Detect velocity reversal: downward motion -> upward motion
        # In screen coordinates: y increases downward, so:
        # - Positive velocity = moving down
        # - Negative velocity = moving up
        # Bounce = transition from positive to negative

        v_prev = velocities[-2]
        v_curr = velocities[-1]

        # Bounce condition: was moving down (v > threshold), now moving up (v < -threshold)
        bounce = v_prev > self.bounce_velocity_threshold and v_curr < -self.bounce_velocity_threshold

        # Prevent duplicate bounce detection (require gap between bounces)
        if bounce and self.last_bounce_frame is not None:
            frame_gap = self.ball_history[-1][0] - self.last_bounce_frame
            if frame_gap < 5:  # Minimum 5 frames between bounces
                return False

        return bounce

    def _detect_hit(
        self,
        ball_pos: tuple[float, float],
        player_tracks: np.ndarray,
        frame_id: int,
    ) -> Optional[int]:
        """Detect if ball is near any player.

        Args:
            ball_pos: Ball position (x, y)
            player_tracks: Array of player tracks [x1, y1, x2, y2, track_id]
            frame_id: Current frame number

        Returns:
            Track ID of player who hit the ball, or None
        """
        if len(player_tracks) == 0:
            return None

        # Prevent detecting same hit multiple times
        if self.last_hit_frame is not None:
            frame_gap = frame_id - self.last_hit_frame
            if frame_gap < 3:  # Minimum 3 frames between hits
                return None

        ball_x, ball_y = ball_pos

        for track in player_tracks:
            x1, y1, x2, y2, track_id = track

            # Expand bounding box slightly for hit detection
            margin = self.hit_distance_threshold
            x1_expanded = x1 - margin
            y1_expanded = y1 - margin
            x2_expanded = x2 + margin
            y2_expanded = y2 + margin

            # Check if ball is within expanded bbox
            if x1_expanded <= ball_x <= x2_expanded and y1_expanded <= ball_y <= y2_expanded:
                # Prevent hitting same player consecutively
                if self.last_hit_player == int(track_id):
                    continue

                return int(track_id)

        return None

    def end_rally(self, frame_id: int, winner: int, reason: str) -> dict:
        """Manually end a rally (called by external logic).

        Args:
            frame_id: Frame where rally ended
            winner: Player ID who won the point
            reason: Reason for rally end ('ace', 'winner', 'out', 'fault', etc.)

        Returns:
            Rally end event dictionary
        """
        self.rally_active = False

        event = {
            "type": "rally_end",
            "frame": frame_id,
            "winner": winner,
            "reason": reason,
        }

        return event

    def reset(self):
        """Reset detector state (e.g., for new match or game)."""
        self.ball_history = []
        self.last_bounce_frame = None
        self.last_hit_frame = None
        self.last_hit_player = None
        self.rally_active = False
        self.rally_start_frame = None
