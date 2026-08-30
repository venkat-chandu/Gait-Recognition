"""CNN-LSTM classifier for normalized skeletal gait sequences."""
from __future__ import annotations


def build_cnn_lstm(sequence_length: int, feature_count: int, class_count: int):
    import tensorflow as tf
    inputs = tf.keras.Input(shape=(sequence_length, feature_count), name="pose_sequence")
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(96, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="identity")(x)
    model = tf.keras.Model(inputs, outputs, name="gait_cnn_lstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(sequence_length: int, feature_count: int, class_count: int):
    """CNN-only baseline for architecture comparison."""
    import tensorflow as tf
    inputs = tf.keras.Input(shape=(sequence_length, feature_count), name="pose_sequence")
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(96, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="identity")(x)
    model = tf.keras.Model(inputs, outputs, name="gait_cnn")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(sequence_length: int, feature_count: int, class_count: int):
    """LSTM-only baseline for architecture comparison."""
    import tensorflow as tf
    inputs = tf.keras.Input(shape=(sequence_length, feature_count), name="pose_sequence")
    x = tf.keras.layers.LSTM(64)(inputs)
    x = tf.keras.layers.Dropout(.3)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="identity")(x)
    model = tf.keras.Model(inputs, outputs, name="gait_lstm")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
