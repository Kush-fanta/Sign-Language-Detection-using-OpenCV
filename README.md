# Sign-Language-Detection-using-OpenCV

#  Real-Time ASL Alphabet Recognition

This project is a real-time ASL (American Sign Language) alphabet recognition system that detects hand signs using a webcam and predicts the corresponding English letter (A–Z). It leverages MediaPipe-based hand detection with a TensorFlow Lite classification model for efficient and lightweight inference.

---

## Features

- Real-time webcam-based ASL detection
- 26-letter classification using a `.tflite` model
- Fast and portable — runs on most CPUs with no GPU required
- Clean preprocessing pipeline using hand cropping and resizing
- Live prediction display with OpenCV overlays

---

## Tech Stack

- **Python 3.7+**
- **OpenCV** (video capture & UI)
- **cvzone + MediaPipe** (hand tracking)
- **TensorFlow Lite** (model inference)
- **NumPy & Math** (processing support)

---

## 📂 Structure

```
.
├── cropped_data/             # Preprocessed dataset of hand signs (A–Z)
├── model.tflite              # Trained TFLite classification model
├── labels.txt                # One label per line (A to Z)
├── tflite_testing.py         # Real-time detection and prediction script




