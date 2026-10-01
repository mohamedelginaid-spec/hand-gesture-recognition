# Hand Gesture Recognition 🖐️

A webcam application that recognizes three hand gestures using MediaPipe hand landmarks and a custom machine learning classifier.

Developed by **Mohamed Elginaid** as a first practical computer vision project.

## Overview

This project demonstrates a complete workflow: capturing webcam frames, detecting hand landmarks, collecting labeled samples, training a classifier, and displaying predictions on live video.

The classifier uses the geometry of the hand rather than raw images.

### Supported Gestures

| Gesture | Description |
|---------|-------------|
| Open | An open hand |
| Fist | A closed fist |
| Point | An extended index finger |

## Features

- Webcam-based hand detection.
- Tracking and visualization of 21 hand landmarks.
- Collection of custom labeled gesture samples.
- Landmark normalization relative to the wrist.
- A custom MLP classifier trained with scikit-learn.
- Evaluation using a separate recording session.
- Live display of predicted gesture names.

## How It Works

1. OpenCV captures a webcam frame.
2. MediaPipe detects the hand and estimates 21 landmarks.
3. The x and y coordinates are converted into pixel coordinates.
4. The wrist coordinates are subtracted from every landmark.
5. The coordinates are divided by their maximum absolute value.
6. The resulting 42 features are passed to the trained classifier.
7. The predicted gesture is displayed on the video.

Normalization reduces sensitivity to the hand's position and size in the frame. It does not make the system fully invariant to hand rotation or camera angle.

## Technologies

| Technology | Purpose |
|------------|---------|
| Python | Application development |
| OpenCV | Webcam capture and visualization |
| MediaPipe | Hand detection and landmark estimation |
| NumPy | Landmark preprocessing |
| scikit-learn | Classifier training and evaluation |
| joblib | Saving and loading the trained model |

MediaPipe provides the pretrained hand landmark detector. The custom gesture classifier is a scikit-learn MLP, not a TensorFlow Lite gesture model.

## Project Files

| File | Purpose |
|------|---------|
| `camera_test.py` | Test webcam access |
| `hand_tracking.py` | Detect and display hand landmarks |
| `collect_gestures.py` | Collect labeled landmark samples |
| `gestures.csv` | Dataset containing landmarks, labels, and session IDs |
| `train_gestures.py` | Train and evaluate the gesture classifier |
| `gesture_model.joblib` | Saved classifier and associated metadata |
| `recognize_gestures.py` | Run live gesture recognition |
| `requirements.txt` | Python dependencies |
| `gesture_interface.py` | Separate touchless interface experiment |

The touchless interface is a separate experiment. The classification results below apply only to the hand gesture recognition model.

## Installation

The project was developed on an Intel Mac using Python 3.9.

### 1. Download the project

```bash
git clone https://github.com/mohamedelginaid-spec/hand-gesture-recognition.git
cd hand-gesture-recognition
```

Alternatively, download the repository as a ZIP file, extract it, and open the folder in VS Code.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --no-compile -r requirements.txt
```

The development environment used MediaPipe 0.10.14 and scikit-learn 1.5.2.

This implementation uses the legacy MediaPipe Hands API. Migrating to the MediaPipe Tasks API is a future improvement.

## Run the Application

From the project folder, with the virtual environment activated:

```bash
python recognize_gestures.py
```

Show an open hand, a closed fist, or a pointing gesture to the webcam.

Press **q** while the video window is focused to close the application.

The supplied `gesture_model.joblib` file must remain alongside the recognition script.

## Collect Your Own Data

Run:

```bash
python collect_gestures.py
```

With the camera window focused and your hand visible, use:

| Key | Action |
|-----|--------|
| `0` | Save an Open sample |
| `1` | Save a Fist sample |
| `2` | Save a Point sample |
| `q` | End the recording session |

Each run creates a recording session ID. Samples are appended to `gestures.csv`.

Vary the position, distance, and angle of your hand slightly while collecting samples.

The current training script expects exactly two recording sessions: the first for training and the second for testing. Both sessions must contain all three gestures.

To build a new dataset, back up the existing `gestures.csv` and move it out of the project folder before collecting two new sessions.

## Train the Classifier

Run:

```bash
python train_gestures.py
```

The script trains an MLP classifier, evaluates it on the second session, and saves the model to `gesture_model.joblib`.

Training again replaces the saved model, so back it up if you want to preserve it.

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Input features | 42 normalized x/y coordinates |
| Hidden layers | 32 and 16 neurons |
| Solver | L-BFGS |
| Alpha | 0.01 |
| Maximum iterations | 1,000 |
| Random seed | 42 |

## Evaluation Results

The initial dataset was recorded by one person across two sessions.

| Gesture | Training Samples | Test Samples |
|---------|-----------------:|-------------:|
| Open | 30 | 15 |
| Fist | 30 | 16 |
| Point | 31 | 15 |
| **Total** | **91** | **46** |

**Test accuracy: 95.7% — 44 correct predictions out of 46 samples.**

### Confusion Matrix

Rows represent the actual gesture; columns represent the predicted gesture.

| Actual / Predicted | Open | Fist | Point |
|--------------------|-----:|-----:|------:|
| Open | 15 | 0 | 0 |
| Fist | 0 | 16 | 0 |
| Point | 2 | 0 | 13 |

Two Point samples were classified as Open.

These results describe a small, single-person experiment. They do not establish the same accuracy for other users, environments, or continuous live video.

## What This Project Demonstrates

- Integration of webcam capture, hand tracking, and machine learning.
- Conversion of hand landmarks into numerical features.
- Collection and labeling of a custom dataset.
- Training and saving a lightweight classifier.
- Evaluation on a separate recording session.
- Deployment of the classifier in an interactive webcam application.

## Limitations

- Only three static gestures are supported.
- The dataset is small and was collected from one person.
- Unfamiliar hand poses may still be assigned to one of the three known classes.
- Lighting, occlusion, motion blur, and camera angle can affect predictions.
- Only x and y landmark coordinates are used.
- The application does not interpret sign language or translate sentences.
- Frame rate and end-to-end latency have not been formally measured.

## Future Development

- Collect more samples from multiple users.
- Add more gesture classes.
- Add an unknown-gesture rejection mechanism.
- Reduce prediction flicker across video frames.
- Evaluate performance across different lighting conditions and cameras.
- Recognize sequences of movements.
- Migrate hand tracking to the MediaPipe Tasks API.

### Potential Healthcare Extensions

Possible future research directions include:

- Touchless navigation of information displays.
- Interactive hand exercise activities.
- Simple gesture-based selection of predefined requests.
- Hand movement visualization for rehabilitation research.

These are development ideas. This project has not been clinically validated and is not a medical device or a complete sign language translation system.

## Troubleshooting

### The webcam does not open

- Allow camera access for the application launching Python.
- Close other applications that may be using the camera.
- Run one camera script at a time.

The current scripts were developed using the macOS AVFoundation camera backend. Other operating systems may require changing the camera initialization.

### The model cannot be found

Make sure `gesture_model.joblib` is in the same folder as `recognize_gestures.py`.

### A Python dependency is missing

Activate the virtual environment and install the requirements again:

```bash
source .venv/bin/activate
python -m pip install --no-compile -r requirements.txt
```

### Recognition is inconsistent

Keep the entire hand visible, improve lighting, and use poses similar to the training examples. To improve generalization, collect a larger and more varied dataset and evaluate on new sessions.

## Inspiration

Inspired by the landmark-based recognition approach presented in:

[kinivi/hand-gesture-recognition-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe)

This project uses its own collected samples and a scikit-learn MLP classifier. The results reported here belong to this project's experiment.

## Author

**Mohamed Elginaid**  
Robotics & AI Solutions Architect

A practical learning project exploring computer vision and human-computer interaction.
