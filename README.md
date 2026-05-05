# ASL Recognition with MediaPipe

A real-time American Sign Language (ASL) hand sign classifier built with Python and MediaPipe.

## How it works
Uses MediaPipe's hand landmark detection to extract 21 hand keypoints from a webcam feed. 
Landmark coordinates are fed into a trained classifier to predict the corresponding ASL letter.

## Currently recognizes
Letters A, B, C, D

## How to run
1. Install dependencies
pip install mediapipe opencv-python scikit-learn

2. Run the classifier
python videocapture.py
