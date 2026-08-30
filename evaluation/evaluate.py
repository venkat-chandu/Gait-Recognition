"""Evaluate the saved model on its held-out test split."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def main() -> None:
    import tensorflow as tf
    test = np.load("outputs/test_split.npz", allow_pickle=False)
    x, y, names = test["X"], test["y"], test["class_names"].tolist()
    model = tf.keras.models.load_model("models_artifacts/gait_model.h5", compile=False)
    predicted = model.predict(x, verbose=0).argmax(axis=1)
    report = classification_report(y, predicted, target_names=names, output_dict=True, zero_division=0)
    report["accuracy"] = accuracy_score(y, predicted)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    matrix = confusion_matrix(y, predicted)
    plt.figure(figsize=(6, 5)); sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=160); plt.close()
    print(f"Held-out accuracy: {report['accuracy']:.3f}")


if __name__ == "__main__":
    main()
