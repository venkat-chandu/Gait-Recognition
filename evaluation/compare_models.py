"""Train comparable CNN, LSTM, and CNN-LSTM gait baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from models.cnn_lstm import build_cnn, build_cnn_lstm, build_lstm
from preprocessing.augmentation import augment_dataset


BUILDERS = {"cnn": build_cnn, "lstm": build_lstm, "cnn_lstm": build_cnn_lstm}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic/gait_sequences.npz")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--augment-copies", type=int, default=0)
    args = parser.parse_args()
    import tensorflow as tf
    archive = np.load(args.data, allow_pickle=False)
    x, y = archive["X"], archive["y"]
    train, test = train_test_split(np.arange(len(y)), test_size=.3, stratify=y, random_state=42)
    train_x, train_y = x[train], y[train]
    if args.augment_copies:
        train_x, train_y = augment_dataset(train_x, train_y, args.augment_copies)
    results = []
    for name, builder in BUILDERS.items():
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(42)
        model = builder(x.shape[1], x.shape[2], len(archive["class_names"]))
        model.fit(train_x, train_y, epochs=args.epochs, batch_size=8, verbose=0)
        started = perf_counter(); predicted = model.predict(x[test], verbose=0).argmax(axis=1)
        elapsed = (perf_counter() - started) * 1000 / len(test)
        precision, recall, f1, _ = precision_recall_fscore_support(y[test], predicted, average="weighted", zero_division=0)
        results.append({"model": name, "accuracy": accuracy_score(y[test], predicted), "precision": precision,
                        "recall": recall, "f1": f1, "inference_ms_per_sequence": elapsed})
    output = Path("outputs/model_comparison.json"); output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
