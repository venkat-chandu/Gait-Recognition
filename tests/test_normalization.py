import numpy as np
from preprocessing.normalization import normalize_landmarks


def test_normalization_centers_hip_midpoint_and_scales_torso():
    landmarks = np.zeros((33, 3), dtype=np.float32)
    landmarks[23], landmarks[24] = [2, 4, 0], [4, 4, 0]
    landmarks[11], landmarks[12] = [2, 0, 0], [4, 0, 0]
    output = normalize_landmarks(landmarks)
    assert np.allclose((output[23] + output[24]) / 2, 0)
    assert np.isclose(np.linalg.norm((output[11] + output[12]) / 2), 1)
