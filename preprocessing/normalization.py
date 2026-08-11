"""Translation and scale normalization for MediaPipe Pose landmarks."""
from __future__ import annotations

import numpy as np

LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER = 23, 24, 11, 12


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Center landmarks at the hips and scale them by torso length.

    Accepts ``(33, 3)`` xyz landmarks and returns a finite array of identical
    shape.  This removes image location and most body-size variation.
    """
    points = np.asarray(landmarks, dtype=np.float32).copy()
    if points.shape != (33, 3):
        raise ValueError(f"Expected (33, 3) landmarks, got {points.shape}")
    hip_center = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2
    shoulder_center = (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) / 2
    scale = np.linalg.norm(shoulder_center - hip_center)
    scale = max(float(scale), 1e-6)
    return (points - hip_center) / scale
