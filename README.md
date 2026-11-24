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





