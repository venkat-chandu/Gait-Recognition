"""Download the official lightweight MediaPipe Pose Landmarker model."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
DESTINATION = Path("models_artifacts/pose_landmarker_lite.task")


def main() -> None:
    DESTINATION.parent.mkdir(exist_ok=True)
    if DESTINATION.exists() and DESTINATION.stat().st_size > 1_000_000:
        print(f"Pose model already exists: {DESTINATION}")
        return
    temporary = DESTINATION.with_suffix(".pending.task")
    urlretrieve(URL, temporary)
    temporary.replace(DESTINATION)
    print(f"Downloaded pose model: {DESTINATION}")


if __name__ == "__main__":
    main()
