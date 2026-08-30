import numpy as np

from preprocessing.realtime import CentroidTracker, GaitPerception


def test_tracker_keeps_id_when_person_moves():
    tracker = CentroidTracker(max_distance=100)
    first = tracker.update([(10, 10, 40, 80)], timestamp=0)[0]
    second = tracker.update([(22, 12, 40, 80)], timestamp=.1)[0]
    assert first.track_id == second.track_id == 1


def test_low_confidence_prediction_is_unknown_and_reports_motion():
    perception = GaitPerception(classifier=lambda sequence: ("Person 1", .4), sequence_length=2)
    track = perception.update_detections([(0, 0, 20, 40)], timestamp=0)[0]
    perception.add_pose(track, np.zeros((33, 3)), np.array([10, 10]), timestamp=0)
    perception.add_pose(track, np.zeros((33, 3)), np.array([40, 10]), timestamp=.2)
    state = perception.target_state(track.track_id)
    assert track.label == "UNKNOWN"
    assert state["direction"] == "RIGHT"
    assert state["position"] == [40, 10]
