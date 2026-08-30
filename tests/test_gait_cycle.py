import numpy as np
from preprocessing.gait_cycle import analyse_gait_cycle


def test_gait_cycle_returns_expected_fields():
    sequence = np.zeros((30, 33, 3), dtype=np.float32)
    sequence[:, 27, 0] = np.sin(np.linspace(0, 4 * np.pi, 30))
    report = analyse_gait_cycle(sequence, fps=30)
    assert report["cadence_steps_min"] >= 0
    assert isinstance(report["peak_frames"], list)
