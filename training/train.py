"""Train the CNN-LSTM model on synthetic or supplied pose sequences."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from models.cnn_lstm import build_cnn_lstm
from evaluation.visualization import plot_history
from preprocessing.augmentation import augment_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic/gait_sequences.npz")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment-copies", type=int, default=0, help="Additional augmented copies of training sequences")
    args = parser.parse_args()
    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    archive = np.load(args.data, allow_pickle=False)
    x, y, class_names = archive["X"], archive["y"], archive["class_names"].tolist()
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=.30, stratify=y, random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=.20, stratify=y[train_idx], random_state=args.seed)
    train_x, train_y = x[train_idx], y[train_idx]
    if args.augment_copies:
        train_x, train_y = augment_dataset(train_x, train_y, args.augment_copies, args.seed)
    model = build_cnn_lstm(x.shape[1], x.shape[2], len(class_names))
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
    history = model.fit(train_x, train_y, validation_data=(x[val_idx], y[val_idx]),
                        epochs=args.epochs, batch_size=args.batch_size, verbose=2, callbacks=callbacks)
    Path("models_artifacts").mkdir(exist_ok=True); Path("outputs").mkdir(exist_ok=True)
    # Write the model atomically so an interrupted process never replaces a
    # previously valid artifact with a partial .keras archive.
    model_path = Path("models_artifacts/gait_model.h5")
    temporary_model = model_path.with_name("gait_model.pending.h5")
    temporary_model.unlink(missing_ok=True)
    model.save(temporary_model)
    os.replace(temporary_model, model_path)
    np.savez_compressed("outputs/test_split.npz", X=x[test_idx], y=y[test_idx], class_names=np.asarray(class_names))
    plot_history(history.history, Path("outputs"))
    Path("models_artifacts/class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print("Training complete. Run: python -m evaluation.evaluate")


if __name__ == "__main__":
    main()
