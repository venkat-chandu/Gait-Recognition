"""Generate a clearly labelled *demonstration-only* skeletal gait dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

SEQUENCE_LENGTH, LANDMARKS, CHANNELS = 30, 33, 3


def generate(samples_per_person: int, people: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.18, (LANDMARKS, CHANNELS)).astype(np.float32)
    # A loose symmetric skeleton ensures the normalizer's hip/shoulder anchors exist.
    base[23], base[24] = [-.14, .25, 0], [.14, .25, 0]
    base[11], base[12] = [-.22, -.35, 0], [.22, -.35, 0]
    sequences, labels = [], []
    for person in range(people):
        cadence = .75 + person * .16
        stride = .16 + person * .035
        arm = .07 + person * .018
        posture = (person - people / 2) * .025
        for _ in range(samples_per_person):
            phase = rng.uniform(0, 2 * np.pi)
            time = np.linspace(0, 2 * np.pi, SEQUENCE_LENGTH) + phase
            item = np.repeat(base[None], SEQUENCE_LENGTH, axis=0)
            wave = np.sin(cadence * time)
            item[:, [25, 27, 29, 31], 0] += (stride * wave)[:, None]
            item[:, [26, 28, 30, 32], 0] -= (stride * wave)[:, None]
            item[:, [13, 15, 17, 19, 21], 0] += (arm * wave)[:, None]
            item[:, [14, 16, 18, 20, 22], 0] -= (arm * wave)[:, None]
            item[:, :, 1] += posture
            item += rng.normal(0, .018, item.shape)
            sequences.append(item.reshape(SEQUENCE_LENGTH, -1))
            labels.append(person)
    return np.asarray(sequences, np.float32), np.asarray(labels, np.int64), [f"Person {i + 1}" for i in range(people)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-person", type=int, default=20)
    parser.add_argument("--people", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/synthetic/gait_sequences.npz")
    args = parser.parse_args()
    x, y, names = generate(args.samples_per_person, args.people, args.seed)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=x, y=y, class_names=np.asarray(names))
    print(f"Saved {len(x)} synthetic sequences to {path}")


if __name__ == "__main__":
    main()
