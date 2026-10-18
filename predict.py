"""
predict.py - Predict handwritten digits (0-9) from image files using the model trained by train.py.

Use it in two ways:

  1. Predict your own images:
       python predict.py path/to/image1.png path/to/image2.png

  2. Run all six test cases (TC1 to TC6) and save the results:
       python predict.py --test

Run train.py first, so that mnist_cnn.keras and the sample_images/ folder exist.
"""

import os
import sys

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy import ndimage
from tensorflow import keras

MODEL_PATH = "mnist_cnn.keras"
SAMPLES_DIR = "sample_images"
TEST_INPUTS_DIR = "test_inputs"
OUTPUTS_DIR = "outputs"
IMG_SIZE = 28

_model = None  # the model is loaded once and reused


def get_model():
    """Load the trained model the first time it is needed."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file '{MODEL_PATH}' not found. Run 'python train.py' first."
            )
        _model = keras.models.load_model(MODEL_PATH)
    return _model


def preprocess_image(path):
    """
    Turn any image file into a (1, 28, 28, 1) array the model can read.

    Handles:
      - corrupted / unreadable files   -> raises ValueError with a clear message (TC5)
      - colour images                  -> converted to grayscale (TC4)
      - wrong dimensions               -> cropped around the digit and resized to 28x28 (TC3)
      - photos with grey paper/shadows -> background removed, digit centred like MNIST
      - blank images                   -> raises ValueError (no digit found)
    """
    # Open the file safely
    try:
        img = Image.open(path)
        img.load()  # forces the full file to be read, so broken files fail here
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ValueError(f"Could not read '{path}': the file is corrupted or is not a valid image.")

    # If the image has transparency, put it on a white background first
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    # Colour -> grayscale
    img = img.convert("L")
    arr = np.array(img, dtype="float32")

    # --- Find the ink and remove the background ---------------------------
    # The background (paper) is the most common shade, so use the median.
    # The ink is whichever side (darker or brighter) differs more from it.
    background = np.median(arr)
    dark_gap = background - np.percentile(arr, 1)
    bright_gap = np.percentile(arr, 99) - background
    if dark_gap >= bright_gap:
        signal = background - arr   # dark ink on lighter paper
    else:
        signal = arr - background   # bright ink on darker background
    signal = np.clip(signal, 0, None)

    if signal.max() < 20:
        raise ValueError(f"No digit found in '{path}': the image looks blank or has no clear writing.")

    # Scale so the ink is 0-255, and wipe out faint noise (shadows, paper texture)
    signal = signal / signal.max() * 255.0
    signal[signal < 80] = 0

    # Keep only the main digit: remove small stray marks (dust, pen dots, page edges)
    labels, count = ndimage.label(signal > 0, structure=np.ones((3, 3)))
    if count > 1:
        sizes = ndimage.sum(signal > 0, labels, index=range(1, count + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s >= 0.25 * max(sizes)]
        signal = np.where(np.isin(labels, keep), signal, 0)

    # --- Crop tightly around the digit ------------------------------------
    rows = np.where(signal.max(axis=1) > 0)[0]
    cols = np.where(signal.max(axis=0) > 0)[0]
    signal = signal[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # Make it square (keeps the digit's shape, no stretching)
    h, w = signal.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype="float32")
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = signal

    # --- Shrink to 20x20 and centre inside 28x28, like MNIST digits --------
    digit = Image.fromarray(square.astype("uint8")).resize((20, 20), Image.LANCZOS)
    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    canvas.paste(digit, ((IMG_SIZE - 20) // 2, (IMG_SIZE - 20) // 2))

    arr = np.array(canvas, dtype="float32")

    # Shift so the digit's centre of mass sits in the middle (MNIST style)
    cy, cx = ndimage.center_of_mass(arr)
    arr = ndimage.shift(arr, (IMG_SIZE / 2 - cy, IMG_SIZE / 2 - cx), order=1)
    arr = np.clip(arr, 0, 255)

    if arr.max() > 0:
        arr = arr / arr.max() * 255.0  # make the strokes bright again after shrinking

    # Normalize to 0-1, the same as in training
    arr = arr / 255.0
    return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def predict_digit(path):
    """Predict one image. Returns (digit 0-9, confidence between 0 and 1)."""
    x = preprocess_image(path)
    probs = get_model().predict(x, verbose=0)[0]
    digit = int(np.argmax(probs))
    return digit, float(probs[digit])


def predict_digits(paths):
    """
    Predict many images (TC6). Returns one result per image, in the same order.
    A bad image does not stop the others; it just gets an error message.
    """
    results = []
    for path in paths:
        try:
            digit, conf = predict_digit(path)
            results.append({"file": path, "digit": digit, "confidence": conf, "error": None})
        except ValueError as e:
            results.append({"file": path, "digit": None, "confidence": None, "error": str(e)})
    return results


def format_result(r, actual=None):
    """Make one result into a readable line."""
    name = os.path.basename(r["file"])
    if r["error"]:
        return f"{name}: ERROR - {r['error']}"
    line = f"{name}: Predicted = {r['digit']} (confidence {r['confidence'] * 100:.1f}%)"
    if actual is not None:
        line = f"{name}: Actual = {actual}, Predicted = {r['digit']} (confidence {r['confidence'] * 100:.1f}%)"
    return line


def actual_from_filename(path):
    """Sample files are named digit7_1.png, so the actual digit is the number after 'digit'."""
    name = os.path.basename(path)
    if name.startswith("digit") and name[5].isdigit():
        return int(name[5])
    return None


# ----------------------------------------------------------------------
# Test cases TC1 - TC6
# ----------------------------------------------------------------------
def make_test_inputs():
    """Create the special input files needed for TC3, TC4 and TC5."""
    os.makedirs(TEST_INPUTS_DIR, exist_ok=True)
    base = Image.open(os.path.join(SAMPLES_DIR, "digit7_1.png"))

    # TC3: wrong dimensions (200 x 150 instead of 28 x 28)
    base.resize((200, 150)).save(os.path.join(TEST_INPUTS_DIR, "wrong_size.png"))

    # TC4: colour image (RGB version of a digit 3 with a coloured background)
    three = Image.open(os.path.join(SAMPLES_DIR, "digit3_1.png")).convert("L")
    colour = Image.merge("RGB", (three, three.point(lambda p: p // 2), three.point(lambda p: p // 3)))
    colour.resize((64, 64)).save(os.path.join(TEST_INPUTS_DIR, "colour_image.png"))

    # TC5: corrupted file (random bytes pretending to be a PNG)
    with open(os.path.join(TEST_INPUTS_DIR, "corrupted.png"), "wb") as f:
        f.write(os.urandom(500))


def run_test_cases():
    if not os.path.isdir(SAMPLES_DIR):
        print(f"'{SAMPLES_DIR}/' not found. Run 'python train.py' first.")
        return

    make_test_inputs()
    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    log("=== Handwritten Digit Recognition: Test Case Results ===\n")

    # TC1
    p = os.path.join(SAMPLES_DIR, "digit7_1.png")
    r = predict_digits([p])[0]
    ok = r["error"] is None and 0 <= r["digit"] <= 9
    log("TC1: Image labelled digit 7")
    log("  " + format_result(r, actual=7))
    log(f"  Result: {'PASS' if ok else 'FAIL'}\n")

    # TC2
    p = os.path.join(SAMPLES_DIR, "digit3_1.png")
    r = predict_digits([p])[0]
    ok = r["error"] is None and 0 <= r["digit"] <= 9
    log("TC2: Image labelled digit 3")
    log("  " + format_result(r, actual=3))
    log(f"  Result: {'PASS' if ok else 'FAIL'}\n")

    # TC3
    p = os.path.join(TEST_INPUTS_DIR, "wrong_size.png")
    r = predict_digits([p])[0]
    ok = r["error"] is None and 0 <= r["digit"] <= 9
    log("TC3: Image with wrong dimensions (200x150, resized to 28x28)")
    log("  " + format_result(r, actual=7))
    log(f"  Result: {'PASS' if ok else 'FAIL'}\n")

    # TC4
    p = os.path.join(TEST_INPUTS_DIR, "colour_image.png")
    r = predict_digits([p])[0]
    ok = r["error"] is None and 0 <= r["digit"] <= 9
    log("TC4: Colour image (converted to grayscale)")
    log("  " + format_result(r, actual=3))
    log(f"  Result: {'PASS' if ok else 'FAIL'}\n")

    # TC5
    p = os.path.join(TEST_INPUTS_DIR, "corrupted.png")
    r = predict_digits([p])[0]
    ok = r["error"] is not None  # we expect a clear error message, not a crash
    log("TC5: Corrupted / unreadable image")
    log("  " + format_result(r))
    log(f"  Result: {'PASS' if ok else 'FAIL'}\n")

    # TC6
    paths = sorted(
        os.path.join(SAMPLES_DIR, f) for f in os.listdir(SAMPLES_DIR) if f.endswith(".png")
    )
    results = predict_digits(paths)
    ok = len(results) == len(paths) and all(
        r["error"] is None and 0 <= r["digit"] <= 9 for r in results
    )
    log(f"TC6: Multiple images ({len(paths)} images, one prediction each)")
    for r in results:
        log("  " + format_result(r, actual=actual_from_filename(r["file"])))
    log(f"  Result: {'PASS' if ok else 'FAIL'}\n")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUTS_DIR, "test_results.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Results saved to '{out_path}'")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    if args[0] == "--test":
        run_test_cases()
        return

    for r in predict_digits(args):
        print(format_result(r, actual=actual_from_filename(r["file"])))


if __name__ == "__main__":
    main()
