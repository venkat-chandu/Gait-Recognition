"""MediaPipe Pose extraction, imported lazily so synthetic workflows stay light."""
from __future__ import annotations

import numpy as np
from .normalization import normalize_landmarks


def extract_pose_landmarks(rgb_frame: np.ndarray, pose) -> tuple[np.ndarray | None, np.ndarray]:
    """Return normalized (33, 3) landmarks and the annotated RGB frame."""
    import mediapipe as mp

    result = pose.process(rgb_frame)
    annotated = rgb_frame.copy()
    if not result.pose_landmarks:
        return None, annotated
    mp.solutions.drawing_utils.draw_landmarks(
        annotated, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
    )
    raw = np.array([[p.x, p.y, p.z] for p in result.pose_landmarks.landmark], dtype=np.float32)
    return normalize_landmarks(raw), annotated
