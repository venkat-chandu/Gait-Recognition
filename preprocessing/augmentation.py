"""Safe, skeletal sequence augmentation for training only."""
from __future__ import annotations

import numpy as np


def augment_sequence(sequence: np.ndarray, rng: np.random.Generator | None = None,
                     noise_std: float = .008, scale_range: tuple[float, float] = (.94, 1.06),
                     rotation_degrees: float = 6.0, temporal_jitter: int = 2) -> np.ndarray:
    """Apply small scale/rotation/noise and temporal variations to a pose sequence.

    Input is ``(frames, 99)`` or ``(frames, 33, 3)``; output matches its shape.
    """
    values = np.asarray(sequence, dtype=np.float32)
    original_shape = values.shape
    if values.ndim == 2 and values.shape[1] == 99:
        points = values.reshape(len(values), 33, 3).copy()
    elif values.ndim == 3 and values.shape[1:] == (33, 3):
        points = values.copy()
    else:
        raise ValueError("Expected (frames, 99) or (frames, 33, 3) pose sequence")
    rng = rng or np.random.default_rng()
    scale = rng.uniform(*scale_range)
    angle = np.deg2rad(rng.uniform(-rotation_degrees, rotation_degrees))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    points[..., :2] = points[..., :2] @ rotation.T * scale
    points += rng.normal(0, noise_std, points.shape).astype(np.float32)
    if temporal_jitter:
        source = np.clip(np.arange(len(points)) + rng.integers(-temporal_jitter, temporal_jitter + 1, len(points)), 0, len(points) - 1)
        points = points[source]
    return points.reshape(original_shape)


def augment_dataset(x: np.ndarray, y: np.ndarray, copies: int = 1, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return originals plus augmented copies with the same labels."""
    if copies < 0:
        raise ValueError("copies must be non-negative")
    rng = np.random.default_rng(seed)
    batches = [np.asarray(x, dtype=np.float32)]
    for _ in range(copies):
        batches.append(np.asarray([augment_sequence(item, rng) for item in x], dtype=np.float32))
    return np.concatenate(batches), np.tile(y, copies + 1)
