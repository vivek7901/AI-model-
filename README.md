[README.md](https://github.com/user-attachments/files/35322104/README.md)
# Hand Gesture Recognition and Character-to-Word Prediction

## Project overview

This project aims to translate hand gestures into text. A camera captures a hand, MediaPipe detects the hand and extracts landmark information, and a machine-learning classifier uses numerical features to predict a character. A higher-level prediction stage combines character predictions to form words. The system is intended to run as an integrated application on Raspberry Pi hardware.

> **Documentation note:** This README is based on the project discussions available, not a current inspection of the source code. Details such as the exact Raspberry Pi model, camera interface, feature-vector format, class labels, selected final classifier, and word-commit rules must be checked against the implementation before submission.

## End-to-end workflow

1. Capture frames from a camera.
2. Process each frame using MediaPipe Hands.
3. Detect a hand and obtain its 21 hand landmarks.
4. Convert the landmark positions into a numerical feature vector.
5. Apply the same preprocessing used during model training.
6. Pass the feature vector to the trained machine-learning model.
7. Receive a predicted gesture class/character.
8. Apply prediction stabilization or temporal logic, if implemented.
9. Accumulate accepted characters and form a word.
10. Display or otherwise output the recognized text.

```text
Camera
  |
  v
Frame acquisition
  |
  v
MediaPipe hand detection
  |
  v
21 hand landmarks
  |
  v
Feature extraction / preprocessing
  |
  v
Trained ML classifier
  |
  v
Character prediction
  |
  v
Prediction stabilization and character buffer
  |
  v
Word formation
  |
  v
Text output
```

## Main project modules

| Module | Responsibility |
|---|---|
| Camera/input | Supplies live frames to the application |
| MediaPipe | Detects the hand and estimates landmark positions |
| Feature extraction | Converts landmarks into numeric model input |
| Dataset preparation | Organizes labelled examples for training/testing |
| Model training | Fits and evaluates candidate classifiers |
| Character prediction | Predicts a character from a gesture |
| Word formation | Combines accepted characters into words |
| Raspberry Pi runtime | Runs the integrated application on target hardware |

## Machine-learning approach

The project discussions include Support Vector Machine (SVM) and Decision Tree as models considered or evaluated. SVM is suitable for structured numerical features but requires careful selection of kernel and hyperparameters. A single Decision Tree is easy to interpret, but can be less robust and more sensitive to changes in training data than an ensemble.

Do not describe a model as the final deployed model until its name is confirmed from the training and inference scripts. If additional models were evaluated, list them only after checking the project code or results.

## Character and word prediction

A character prediction is the classifier's output for one processed hand pose. Word formation is a separate, higher-level task: the application must decide when a character is stable enough to accept, append it to a text buffer, and determine when a word is complete. The exact acceptance, repetition, pause, deletion, and space rules depend on the implemented controls.

A frame-by-frame classifier does not automatically understand a complete word. It predicts individual gesture classes; temporal logic in the application combines those predictions.

## Raspberry Pi deployment

The intended deployment places the application and its required model/runtime files on Raspberry Pi hardware. The application is launched through the Python environment available on the Pi; project discussions specifically included running an integrated script through Thonny.

A typical deployment checklist:

- Confirm the exact Raspberry Pi board and operating-system version.
- Confirm that the camera is detected and accessible.
- Install compatible Python packages, including the project's MediaPipe, OpenCV, and ML dependencies.
- Copy the trained model and any preprocessing/scaler files to the Pi.
- Keep training-time preprocessing identical at inference time.
- Run the integrated entry-point script in Thonny or from a terminal.
- Test camera capture, hand detection, prediction, word formation, and output independently.

A Raspberry Pi Pico is a microcontroller and is not interchangeable with a Linux-capable Raspberry Pi computer. If the actual target is a Pico, the deployment architecture and supported libraries would need to be substantially different. Verify the board model before describing the operating system, Thonny runtime, or MediaPipe deployment.

## Suggested repository layout

```text
project/
├── README.md
├── docs/
│   ├── 01_project_overview.md
│   ├── 02_mediapipe_hand_detection.md
│   ├── 03_dataset_and_features.md
│   ├── 04_model_training.md
│   ├── 05_character_prediction.md
│   ├── 06_word_formation.md
│   ├── 07_raspberry_pi_deployment.md
│   ├── 08_integration_and_execution.md
│   ├── 09_testing_and_troubleshooting.md
│   └── 10_limitations_and_future_scope.md
├── src/
├── models/
└── data/
```

The folder names above are a suggested organization, not a claim about the current repository.

## Important items to verify before final submission

- Exact Raspberry Pi model (for example, Pi 3/4/5 versus Pico).
- Camera type and capture method.
- Whether one hand or both hands are supported.
- Exact landmark representation and feature-vector length.
- Normalization or scaling method, if any.
- Gesture labels and character mapping.
- All candidate algorithms and the final selected model.
- Measured test accuracy and evaluation method.
- How predictions are stabilized and committed.
- How the user signals a space/end-of-word or corrects a character.
- Actual script names and model file names.

## Documentation index

- [Project overview](docs/01_project_overview.md)
- [MediaPipe hand detection](docs/02_mediapipe_hand_detection.md)
- [Dataset and feature extraction](docs/03_dataset_and_features.md)
- [Model training](docs/04_model_training.md)
- [Character prediction](docs/05_character_prediction.md)
- [Word formation](docs/06_word_formation.md)
- [Raspberry Pi deployment](docs/07_raspberry_pi_deployment.md)
- [Integration and execution](docs/08_integration_and_execution.md)
- [Testing and troubleshooting](docs/09_testing_and_troubleshooting.md)
- [Limitations and future scope](docs/10_limitations_and_future_scope.md)
