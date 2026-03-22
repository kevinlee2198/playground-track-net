import numpy as np


class KalmanBallTracker:
    """Kalman filter for smoothing ball position detections.

    Pure NumPy implementation (no filterpy dependency).
    State: [x, y, vx, vy] -- position and velocity.
    Measurement: [x, y] -- detected ball position.
    Constant-velocity model with higher process noise for fast ball dynamics.
    """

    def __init__(self, process_noise: float = 50.0, measurement_noise: float = 5.0):
        # State vector: [x, y, vx, vy]
        self.x = np.zeros((4, 1), dtype=np.float64)

        # State transition: constant velocity model
        # x' = x + vx*dt, y' = y + vy*dt (dt=1 frame)
        self.F = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Measurement function: observe x, y only
        self.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float64,
        )

        # Process noise -- higher than typical for fast ball dynamics
        self.Q = np.eye(4, dtype=np.float64) * process_noise
        self.Q[2, 2] *= 2.0  # Extra noise on velocity components
        self.Q[3, 3] *= 2.0

        # Measurement noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

        # Covariance matrix -- high initial uncertainty
        self.P = np.eye(4, dtype=np.float64) * 1000.0

        self._initialized = False

    def _predict(self):
        """Predict step: project state and covariance forward."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _correct(self, z: np.ndarray):
        """Update step: incorporate measurement."""
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z.reshape(2, 1) - self.H @ self.x
        self.x = self.x + K @ y
        identity = np.eye(4, dtype=np.float64)
        self.P = (identity - K @ self.H) @ self.P

    def update(self, x: float, y: float) -> tuple[float, float]:
        """Update tracker with a new measurement and return smoothed position."""
        measurement = np.array([x, y], dtype=np.float64)

        if not self._initialized:
            self.x[:2] = measurement.reshape(2, 1)
            self._initialized = True
        else:
            self._predict()
            self._correct(measurement)

        return (float(self.x[0, 0]), float(self.x[1, 0]))

    def predict(self) -> tuple[float, float]:
        """Predict next position without a measurement."""
        self._predict()
        return (float(self.x[0, 0]), float(self.x[1, 0]))

    def reset(self):
        """Reset tracker to initial state."""
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1000.0
        self._initialized = False
