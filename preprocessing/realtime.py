"""Stateful components for real-time, multi-person gait perception.

The classes in this module are intentionally independent of Streamlit and the
neural-network runtime.  They can therefore be used by a camera application,
a robotics process, or unit tests.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import cv2
import numpy as np


BBox = tuple[int, int, int, int]


def bbox_center(box: BBox) -> np.ndarray:
    x, y, w, h = box
    return np.asarray((x + w / 2, y + h / 2), dtype=np.float32)


def iou(first: BBox, second: BBox) -> float:
    ax, ay, aw, ah = first; bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


@dataclass
class Track:
    track_id: int
    bbox: BBox
    last_seen: float
    positions: deque[tuple[float, np.ndarray]] = field(default_factory=lambda: deque(maxlen=12))
    sequence: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    label: str = "Collecting pose"
    confidence: float = 0.0

    def add_position(self, position: np.ndarray, timestamp: float) -> None:
        self.positions.append((timestamp, position.astype(np.float32)))

    def movement(self) -> tuple[str, float]:
        """Return image-space direction and speed in pixels/second."""
        if len(self.positions) < 2:
            return "STOPPED", 0.0
        start_time, start = self.positions[0]
        end_time, end = self.positions[-1]
        elapsed = max(end_time - start_time, 1e-6)
        delta = end - start
        speed = float(np.linalg.norm(delta) / elapsed)
        if speed < 12:
            return "STOPPED", speed
        if abs(delta[0]) > abs(delta[1]):
            return ("RIGHT" if delta[0] > 0 else "LEFT"), speed
        return ("FORWARD" if delta[1] < 0 else "BACKWARD"), speed


class CentroidTracker:
    """Lightweight persistent-ID tracker using IoU and centroid distance."""

    def __init__(self, max_age_seconds: float = 1.2, max_distance: float = 130.0):
        self.max_age_seconds = max_age_seconds
        self.max_distance = max_distance
        self._next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, boxes: list[BBox], timestamp: float | None = None) -> list[Track]:
        now = perf_counter() if timestamp is None else timestamp
        available = set(self.tracks)
        assigned: set[int] = set()
        for box in boxes:
            center = bbox_center(box)
            candidates = [
                (track_id, iou(self.tracks[track_id].bbox, box),
                 float(np.linalg.norm(bbox_center(self.tracks[track_id].bbox) - center)))
                for track_id in available
            ]
            candidates = [item for item in candidates if item[1] >= 0.08 or item[2] <= self.max_distance]
            if candidates:
                track_id = max(candidates, key=lambda item: item[1] - item[2] / (self.max_distance * 4))[0]
                track = self.tracks[track_id]
                track.bbox, track.last_seen = box, now
                available.remove(track_id)
            else:
                track = Track(self._next_id, box, now)
                self.tracks[track.track_id] = track
                self._next_id += 1
            assigned.add(track.track_id)
        self.tracks = {key: value for key, value in self.tracks.items() if now - value.last_seen <= self.max_age_seconds}
        return [self.tracks[key] for key in sorted(assigned) if key in self.tracks]


class OpenCVPersonDetector:
    """CPU-only person detector; replace with a trained detector for deployment."""

    def __init__(self) -> None:
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, bgr_frame: np.ndarray) -> list[BBox]:
        boxes, _ = self.hog.detectMultiScale(bgr_frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        return [tuple(map(int, box)) for box in boxes]


class GaitPerception:
    """Adds pose sequences, recognition and movement information to tracks."""

    def __init__(self, classifier: Callable[[np.ndarray], tuple[str, float]] | None = None,
                 sequence_length: int = 30, unknown_threshold: float = 0.65):
        self.tracker = CentroidTracker()
        self.classifier = classifier
        self.sequence_length = sequence_length
        self.unknown_threshold = unknown_threshold

    def update_detections(self, boxes: list[BBox], timestamp: float | None = None) -> list[Track]:
        return self.tracker.update(boxes, timestamp)

    def add_pose(self, track: Track, normalized_landmarks: np.ndarray, position: np.ndarray,
                 timestamp: float | None = None) -> None:
        now = perf_counter() if timestamp is None else timestamp
        track.sequence.append(np.asarray(normalized_landmarks, dtype=np.float32).reshape(-1))
        track.add_position(position, now)
        if self.classifier is not None and len(track.sequence) >= self.sequence_length:
            label, confidence = self.classifier(np.asarray(track.sequence, dtype=np.float32))
            track.confidence = confidence
            track.label = label if confidence >= self.unknown_threshold else "UNKNOWN"

    def target_state(self, track_id: int | None) -> dict[str, object] | None:
        if track_id is None or track_id not in self.tracker.tracks:
            return None
        track = self.tracker.tracks[track_id]
        direction, speed = track.movement()
        position = track.positions[-1][1].round().astype(int).tolist() if track.positions else None
        state = {"id": f"person_{track.track_id:02d}", "gait": track.label,
                "confidence": track.confidence, "position": position,
                "direction": direction, "speed_px_s": speed, "tracking": True}
        if len(track.sequence) >= 4:
            from .gait_cycle import analyse_gait_cycle
            state["gait_cycle"] = analyse_gait_cycle(np.asarray(track.sequence), fps=30.0)
        return state
