import numpy as np
from preprocessing.augmentation import augment_dataset, augment_sequence


def test_augmentation_preserves_shape_and_changes_values():
    source = np.ones((30, 99), dtype=np.float32)
    output = augment_sequence(source, np.random.default_rng(1))
    assert output.shape == source.shape
    assert not np.array_equal(output, source)


def test_augment_dataset_repeats_labels():
    x = np.zeros((2, 30, 99), np.float32); y = np.array([0, 1])
    expanded_x, expanded_y = augment_dataset(x, y, copies=2)
    assert expanded_x.shape[0] == 6
    assert expanded_y.tolist() == [0, 1, 0, 1, 0, 1]
