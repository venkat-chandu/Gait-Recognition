"""Image-to-ground-plane conversion for calibrated camera deployments."""
from __future__ import annotations

import numpy as np


def image_to_world(point: tuple[float, float] | np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Project an image point onto a calibrated planar ground coordinate system."""
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")
    projected = matrix @ np.array([point[0], point[1], 1.0])
    if abs(projected[2]) < 1e-12:
        raise ValueError("Point projects to infinity")
    return projected[:2] / projected[2]
