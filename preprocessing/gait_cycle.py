"""Approximate gait-cycle measures derived from normalized skeletal poses."""
from __future__ import annotations

import numpy as np


LEFT_ANKLE, RIGHT_ANKLE = 27, 28


def analyse_gait_cycle(sequence: np.ndarray, fps: float) -> dict[str, float | int | list[int]]:
    """Estimate cadence and cycle duration from alternating ankle separation peaks.

    Values are estimates suitable for research visualization, not clinical use.
    """
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim == 2 and values.shape[1] == 99:
        values = values.reshape(len(values), 33, 3)
    if values.ndim != 3 or values.shape[1:] != (33, 3) or fps <= 0:
        raise ValueError("Expected (frames, 99)/(frames, 33, 3) and positive fps")
    signal = np.abs(values[:, LEFT_ANKLE, 0] - values[:, RIGHT_ANKLE, 0])
    peaks = [index for index in range(1, len(signal) - 1) if signal[index] > signal[index - 1] and signal[index] >= signal[index + 1]]
    if len(peaks) >= 2:
        intervals = np.diff(peaks) / fps
        cycle = float(np.median(intervals) * 2)  # alternating peak to same-foot cycle
        cadence = float(120 / cycle) if cycle else 0.0
    else:
        cycle, cadence = 0.0, 0.0
    return {"cycle_duration_s": cycle, "cadence_steps_min": cadence,
            "peak_frames": peaks, "left_right_separation": float(signal.mean())}
