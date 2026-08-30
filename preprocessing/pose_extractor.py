"""MediaPipe Tasks pose extraction for recorded and live RGB frames."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from .normalization import normalize_landmarks


DEFAULT_POSE_MODEL = Path("models_artifacts/pose_landmarker_lite.task")


class PoseDetector:
    """Small synchronous wrapper around MediaPipe's supported Tasks API."""

    def __init__(self, model_path: str | Path = DEFAULT_POSE_MODEL) -> None:
        import mediapipe as mp
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Pose model not found: {path}. Download pose_landmarker_lite.task "
                "to models_artifacts before using camera or video pose extraction."
            )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=.5,
            min_pose_presence_confidence=.5,
            min_tracking_confidence=.5,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect(self, rgb_frame: np.ndarray):
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        return self._landmarker.detect(image)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def extract_pose_landmarks(rgb_frame: np.ndarray, detector: PoseDetector) -> tuple[np.ndarray | None, np.ndarray]:
    """Return normalized ``(33, 3)`` landmarks and an annotated RGB frame."""
    result = detector.detect(rgb_frame)
    annotated = rgb_frame.copy()
    if not result.pose_landmarks:
        return None, annotated
    landmarks = result.pose_landmarks[0]
    raw = np.array([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)
    height, width = annotated.shape[:2]
    for point in raw:
        cv2.circle(annotated, (int(point[0] * width), int(point[1] * height)), 2, (0, 255, 0), -1)
    return normalize_landmarks(raw), annotated
