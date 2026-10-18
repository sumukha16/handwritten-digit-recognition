"""
train.py - Train a CNN to recognize handwritten digits (0-9) on MNIST.

What this script does:
  1. Loads the MNIST dataset (60,000 training + 10,000 test images)
  2. Normalizes the pixel values (0-255 -> 0-1)
  3. Builds and trains a small CNN
  4. Evaluates it on the held-out test images
  5. Saves the trained model and a few sample test images (including a 7 and a 3)

Run:  python train.py
"""

import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

# Settings (change these if you like)
MODEL_PATH = "mnist_cnn.keras"
SAMPLES_DIR = "sample_images"
EPOCHS = 5
BATCH_SIZE = 128
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data():
    """Download MNIST (first run only) and return train/test splits."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(f"Training images: {x_train.shape}, Test images: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


def preprocess(images):
    """Scale pixels to 0-1 and add a channel dimension: (N, 28, 28) -> (N, 28, 28, 1)."""
    images = images.astype("float32") / 255.0
    return np.expand_dims(images, axis=-1)


def build_model():
    """A small CNN: two conv+pooling blocks, then a dense layer and 10 outputs."""
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=3, activation="relu"),  # finds simple shapes
            layers.MaxPooling2D(pool_size=2),                     # shrinks the image
            layers.Conv2D(64, kernel_size=3, activation="relu"),  # finds bigger shapes
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dropout(0.3),                                  # helps avoid memorizing
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),               # one output per digit
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_sample_images(x_test_raw, y_test, digits=(7, 3), per_digit=3):
    """Save a few raw test images as PNG files, named like digit7_1.png.
    These are used later to test predict.py (TC1, TC2)."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    for digit in digits:
        indices = np.where(y_test == digit)[0][:per_digit]
        for n, idx in enumerate(indices, start=1):
            path = os.path.join(SAMPLES_DIR, f"digit{digit}_{n}.png")
            Image.fromarray(x_test_raw[idx]).save(path)
    print(f"Saved sample images to '{SAMPLES_DIR}/'")


def main():
    (x_train_raw, y_train), (x_test_raw, y_test) = load_data()

    x_train = preprocess(x_train_raw)
    x_test = preprocess(x_test_raw)

    model = build_model()
    model.summary()

    print("\nTraining...")
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,  # 10% of training data used to check progress
        verbose=2,
    )

    print("\nEvaluating on the 10,000 held-out test images...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # Show a few actual vs predicted examples
    print("\nSample predictions (actual vs predicted):")
    probs = model.predict(x_test[:10], verbose=0)
    for i, p in enumerate(probs):
        print(f"  Image {i}: Actual = {y_test[i]}, Predicted = {int(np.argmax(p))}")

    model.save(MODEL_PATH)
    print(f"\nModel saved to '{MODEL_PATH}'")

    save_sample_images(x_test_raw, y_test)


if __name__ == "__main__":
    main()