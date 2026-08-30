"""Controlled perturbation evaluation for robustness reporting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from preprocessing.augmentation import augment_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="outputs/test_split.npz")
    parser.add_argument("--model", default="models_artifacts/gait_model.h5")
    args = parser.parse_args()
    import tensorflow as tf
    data = np.load(args.test_data, allow_pickle=False); model = tf.keras.models.load_model(args.model, compile=False)
    scenarios = {"baseline": data["X"], "small_pose_noise": np.array([augment_sequence(x, noise_std=.02) for x in data["X"]]),
                 "temporal_jitter": np.array([augment_sequence(x, temporal_jitter=4) for x in data["X"]])}
    results = {name: accuracy_score(data["y"], model.predict(x, verbose=0).argmax(axis=1)) for name, x in scenarios.items()}
    Path("outputs/robustness.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
