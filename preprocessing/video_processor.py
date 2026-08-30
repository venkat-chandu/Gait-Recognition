"""Convert a walking video into one fixed-length pose sequence."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from .pose_extractor import extract_pose_landmarks


def _resample(sequence: np.ndarray, sequence_length: int) -> np.ndarray:
    positions = np.linspace(0, len(sequence) - 1, sequence_length).round().astype(int)
    return sequence[positions]

``
def video_to_sequence(video_path: str | Path, sequence_length: int = 30) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract pose frames and resample them to ``(sequence_length, 99)``.

    Raises a helpful error when too few frames contain a detectable person.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")
    frames, preview = [], None
    from .pose_extractor import PoseDetector
    with PoseDetector() as pose:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            landmarks, annotated = extract_pose_landmarks(rgb, pose)
            if landmarks is not None:
                frames.append(landmarks.reshape(-1))
                preview = annotated
    cap.release()
    if len(frames) < 2:
        raise ValueError("No reliable pose sequence was found. Use a clear, single-person walking video.")
    return _resample(np.asarray(frames, dtype=np.float32), sequence_length), preview
