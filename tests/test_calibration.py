import numpy as np
from preprocessing.calibration import image_to_world


def test_identity_homography_preserves_point():
    assert np.allclose(image_to_world((12, 34), np.eye(3)), [12, 34])
