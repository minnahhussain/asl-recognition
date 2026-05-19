# ASL Gesture Recognition
Real-time American Sign Language gesture recognition using MediaPipe and an LSTM neural network. Built to improve accessibility for ASL users in video call environments like Zoom and Google Meet.

## Motivation
ASL users in remote work settings face a significant disadvantage in meetings. While spoken conversation flows naturally, ASL users are often limited to typing responses — which is slower and disrupts the flow of discussion. This project explores gesture recognition as a step toward bridging that gap.

## How It Works
1. **MediaPipe** detects 21 hand landmarks per frame from a webcam feed
2. Landmark coordinates (x, y, z) are collected across multiple frames to form a gesture sequence
3. Sequences are normalized to 45 frames using even sampling
4. An **LSTM model** classifies the sequence into one of the supported gestures
5. The predicted gesture is displayed on screen in real time

## Supported Gestures
- YES
- NO
- PLEASE
- THANK YOU

## Project Structure
```
asl_dynamic.py        # Real-time gesture recognition (main)
asl_static.py         # Static letter classifier (A-D) with data collection
extract_landmarks.py  # Extracts MediaPipe landmarks from WLASL videos
train_model.py        # Trains the LSTM model
train_knn.py          # Trains the static k-NN classifier
url_collector.py      # Downloads gesture videos from WLASL dataset
data/                 # Processed landmark sequences (CSV)
```

## Tech Stack
- Python
- MediaPipe
- TensorFlow / Keras
- OpenCV
- scikit-learn
- NumPy

## Training Data
Videos sourced from the [WLASL dataset](https://github.com/dxli94/WLASL) (Word-Level American Sign Language). Landmark sequences were extracted frame-by-frame using MediaPipe and augmented with scaling, mirroring, and noise to improve generalization across different hand sizes and orientations.

## How to Run
```bash
pip install mediapipe opencv-python tensorflow scikit-learn numpy
python asl_dynamic.py
```
Perform a gesture in front of your webcam. Lower your hand to trigger a prediction.

## Limitations
The current model is trained on a small dataset (~84 samples after augmentation). Accuracy improves with more training data and additional signers.

## Future Work
- Expand gesture vocabulary
- Collect additional training data for improved generalization
- Build browser extension for direct Zoom/Google Meet integration
