"""Plot compact training and evaluation figures."""
from pathlib import Path
import matplotlib.pyplot as plt


def plot_history(history: dict, output_dir: Path) -> None:
    for key, label, filename in [("accuracy", "Accuracy", "accuracy_curve.png"), ("loss", "Loss", "loss_curve.png")]:
        plt.figure(figsize=(6, 4)); plt.plot(history[key], label="Training")
        if f"val_{key}" in history: plt.plot(history[f"val_{key}"], label="Validation")
        plt.xlabel("Epoch"); plt.ylabel(label); plt.legend(); plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160); plt.close()
