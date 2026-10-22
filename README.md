# Handwritten Digit Recognition

A simple system that looks at an image of a handwritten digit and predicts which digit (0-9) it is.
It uses a small CNN (Convolutional Neural Network) trained on the MNIST dataset.

## What this project does

- Loads the MNIST dataset (60,000 training images and 10,000 test images)
- Scales pixel values from 0-255 to 0-1
- Trains a small CNN
- Tests the model on 10,000 images it has never seen
- Predicts the digit for any image you give it, and shows actual vs predicted for the sample images
- Handles bad inputs safely: wrong size, colour images, corrupted files, and many images at once

## Project structure

```
handwritten-digit-recognition/
├── train.py            # trains the model and saves it
├── predict.py          # predicts digits and runs the test cases
├── requirements.txt    # libraries needed
├── sample_images/      # sample digits 7 and 3 (created by train.py)
├── test_inputs/        # special test files for TC3, TC4, TC5 (created by predict.py --test)
└── outputs/            # saved results of the test cases
```

## Setup option 1: Run on your own computer (local)

**You need:** Python 3.9 to 3.12 and an internet connection (MNIST downloads on the first run).

1. Download the project:
   ```bash
   git clone https://github.com/sumukha16/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac / Linux: `source venv/bin/activate`

3. Install the libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the model (takes a few minutes):
   ```bash
   python train.py
   ```
   This saves the model as `mnist_cnn.keras` and creates the `sample_images/` folder.

5. Run all six test cases:
   ```bash
   python predict.py --test
   ```

6. Predict your own images:
   ```bash
   python predict.py path/to/image1.png path/to/image2.png
   ```

## Setup option 2: Run on Google Colab (no installation)

1. Open [colab.research.google.com](https://colab.research.google.com) and click **New notebook**.
2. Paste each block below into its own cell and run them in order.

**Cell 1: Download the project**
```python
!git clone https://github.com/sumukha16/handwritten-digit-recognition.git
%cd handwritten-digit-recognition
```

**Cell 2: Install the libraries**
```python
!pip install -r requirements.txt
```
(Colab already has most of these, so this is quick. If Colab asks you to restart the runtime, do it, then run Cell 1's `%cd` line again.)

**Cell 3: Train the model**
```python
!python train.py
```

**Cell 4: Run all six test cases**
```python
!python predict.py --test
```

**Cell 5: Predict your own image**

Upload an image using the folder icon on the left side of Colab, then run:
```python
!python predict.py /content/my_digit.png
```
Change `my_digit.png` to the name of your uploaded file.

**Optional: See an image with its prediction**
```python
from PIL import Image
import matplotlib.pyplot as plt
from predict import predict_digit

path = "sample_images/digit7_1.png"
digit, confidence = predict_digit(path)

plt.imshow(Image.open(path), cmap="gray")
plt.title(f"Predicted: {digit} ({confidence * 100:.1f}%)")
plt.axis("off")
plt.show()
```

> Tip: Colab files are deleted when the session ends. Download `outputs/test_results.txt` before closing if you want to keep it.

## How the model works

| Step | What happens |
| --- | --- |
| 1 | Load MNIST (28 x 28 grayscale images) |
| 2 | Divide pixel values by 255 so they are between 0 and 1 |
| 3 | Two Conv + MaxPooling blocks find shapes in the digit |
| 4 | Dropout (0.3) reduces memorising the training data |
| 5 | A Dense layer of 128 units, then 10 outputs (one per digit) |
| 6 | Trained for 5 epochs with the Adam optimizer |

## How `predict.py` prepares any image

Before predicting, every image goes through these steps:

1. Opens the file safely (a corrupted file gives a clear error message instead of crashing)
2. Converts colour images to grayscale
3. Removes the paper background and small stray marks
4. Crops around the digit, makes it square, and resizes it to fit MNIST style (20 x 20 inside 28 x 28)
5. Centres the digit and scales the pixels to 0-1

## Test cases

| Test case | Input | Expected behaviour | How it is handled |
| --- | --- | --- | --- |
| TC1 | Image of digit 7 | Prediction from 0 to 9 | Prediction returned |
| TC2 | Image of digit 3 | Prediction from 0 to 9 | Prediction returned |
| TC3 | Wrong dimensions (200 x 150) | Resize or clear error | Cropped and resized to 28 x 28 |
| TC4 | Colour image | Convert or reject | Converted to grayscale |
| TC5 | Corrupted file | Handle gracefully | Clear error message, no crash |
| TC6 | Multiple images | One prediction per image | One result per image, in order |

Run `python predict.py --test` to see all of them. The results are also saved in `outputs/test_results.txt`.

## Sample output

**Training (`python train.py`)**
```
Test accuracy: 98.93%

Sample predictions (actual vs predicted):
  Image 0: Actual = 7, Predicted = 7
  Image 1: Actual = 2, Predicted = 2
  Image 2: Actual = 1, Predicted = 1
  Image 3: Actual = 0, Predicted = 0
  Image 4: Actual = 4, Predicted = 4
  Image 5: Actual = 1, Predicted = 1
  Image 6: Actual = 4, Predicted = 4
  Image 7: Actual = 9, Predicted = 9
  Image 8: Actual = 5, Predicted = 5
  Image 9: Actual = 9, Predicted = 9
```

**Test cases (`python predict.py --test`)**
```
TC1: Image labelled digit 7
  digit7_1.png: Actual = 7, Predicted = 7 (confidence 100.0%)
  Result: PASS

TC2: Image labelled digit 3
  digit3_1.png: Actual = 3, Predicted = 5 (confidence 51.7%)
  Result: PASS

TC3: Image with wrong dimensions (200x150, resized to 28x28)
  wrong_size.png: Actual = 7, Predicted = 7 (confidence 100.0%)
  Result: PASS

TC4: Colour image (converted to grayscale)
  colour_image.png: Actual = 3, Predicted = 5 (confidence 41.2%)
  Result: PASS

TC5: Corrupted / unreadable image
  corrupted.png: ERROR - Could not read 'test_inputs/corrupted.png': the file is corrupted or is not a valid image.
  Result: PASS

TC6: Multiple images (6 images, one prediction each)
  digit3_1.png: Actual = 3, Predicted = 5 (confidence 51.7%)
  digit3_2.png: Actual = 3, Predicted = 3 (confidence 100.0%)
  digit3_3.png: Actual = 3, Predicted = 3 (confidence 100.0%)
  digit7_1.png: Actual = 7, Predicted = 7 (confidence 100.0%)
  digit7_2.png: Actual = 7, Predicted = 7 (confidence 100.0%)
  digit7_3.png: Actual = 7, Predicted = 7 (confidence 100.0%)
  Result: PASS
```

> Numbers above are from an actual run on Colab (Tesla T4 GPU, 5 epochs).

## Libraries used

- TensorFlow / Keras: build and train the CNN
- NumPy: work with image arrays
- Pillow: open and convert images
- SciPy: clean up and centre the digit

## Common problems

- **`Model file 'mnist_cnn.keras' not found`**: run `python train.py` first.
- **`sample_images/ not found`**: run `python train.py` first.
- **TensorFlow will not install**: check your Python version (use 3.9 to 3.12), or use the Colab option.
- **Error: file corrupted or not a valid image**: this is the expected message for TC5. The program is handling it correctly.

## Author

Sumukha Nayak
