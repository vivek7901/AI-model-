import tracemalloc
import cv2
import numpy as np
import mediapipe as mp
import joblib

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


#=============================================================
#RAM CALCULATION IMPORT
tracemalloc.start()
#=============================================================

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"F:\7TH SEMESTER\PROJECT\ISL_project\Trial_2\hand_landmarker.task"

CLASSIFIER_PATH = r"F:\7TH SEMESTER\PROJECT\ISL_project\Trial_2\isl_landmark_random_forest.pkl"


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

print()
print("Loading classifier...")

classifier = joblib.load(
    CLASSIFIER_PATH
)

print("Classifier loaded successfully.")


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)


options = vision.HandLandmarkerOptions(

    base_options=base_options,

    running_mode=vision.RunningMode.VIDEO,

    num_hands=2,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# NORMALIZE ONE HAND
# ============================================================

def normalize_hand(hand, handedness):

    points = []

    for landmark in hand:

        points.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])


    points = np.array(
        points,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Wrist = origin
    # --------------------------------------------------------

    wrist = points[0].copy()

    points = points - wrist


    # --------------------------------------------------------
    # Normalize hand size
    # --------------------------------------------------------

    distances = np.linalg.norm(
        points,
        axis=1
    )


    scale = np.max(distances)


    if scale < 1e-6:

        scale = 1.0


    points = points / scale


    # --------------------------------------------------------
    # RIGHT HAND → CANONICAL ORIENTATION
    # --------------------------------------------------------

    if handedness == "Right":

        points[:, 0] = -points[:, 0]


    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    return points.flatten()


# ============================================================
# PROCESS DETECTED HANDS
# ============================================================

def create_features(result):

    if not result.hand_landmarks:

        return None


    normalized_hands = []


    # --------------------------------------------------------
    # Process each detected hand
    # --------------------------------------------------------

    for index, hand in enumerate(
        result.hand_landmarks
    ):

        if (
            result.handedness
            and index < len(result.handedness)
            and len(result.handedness[index]) > 0
        ):

            handedness = (
                result.handedness[index][0].category_name
            )

        else:

            handedness = "Unknown"


        features = normalize_hand(
            hand,
            handedness
        )


        wrist = hand[0]


        normalized_hands.append({

            "features": features,

            "wrist_x": wrist.x,

            "handedness": handedness
        })


    # --------------------------------------------------------
    # SAME SORTING USED DURING DATASET EXTRACTION
    # --------------------------------------------------------

    normalized_hands.sort(
        key=lambda h: h["wrist_x"]
    )


    # --------------------------------------------------------
    # HAND 1
    # --------------------------------------------------------

    hand1 = normalized_hands[0]["features"]


    # --------------------------------------------------------
    # HAND 2
    # --------------------------------------------------------

    if len(normalized_hands) >= 2:

        hand2 = normalized_hands[1]["features"]

    else:

        hand2 = np.zeros(
            63,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # 126 FEATURES
    # --------------------------------------------------------

    combined = np.concatenate([
        hand1,
        hand2
    ])


    return combined


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    landmarker.close()

    exit()


print()
print("==============================================")
print("       ISL REAL-TIME LANDMARK TEST")
print("==============================================")
print()
print("Show your hand to the camera.")
print()
print("Press Q to quit.")
print()


# ============================================================
# TIMESTAMP
# ============================================================

timestamp_ms = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print("ERROR: Could not read webcam.")

        break


    # --------------------------------------------------------
    # Mirror webcam
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    height, width, _ = frame.shape


    # --------------------------------------------------------
    # BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MediaPipe image
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp_ms += 33


    # --------------------------------------------------------
    # Detect hands
    # --------------------------------------------------------

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # IF HAND DETECTED
    # ========================================================

    if result.hand_landmarks:


        # ----------------------------------------------------
        # Create SAME 126 features as training
        # ----------------------------------------------------

        features = create_features(
            result
        )


        if features is not None:


            # ------------------------------------------------
            # Reshape for Random Forest
            # ------------------------------------------------

            X_live = features.reshape(
                1,
                -1
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = classifier.predict(
                X_live
            )[0]


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            probabilities = classifier.predict_proba(
                X_live
            )[0]


            confidence = np.max(
                probabilities
            ) * 100


            # ------------------------------------------------
            # Display prediction
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Prediction: {prediction}",
                (20, height - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )


            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}%",
                (20, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        # ----------------------------------------------------
        # Draw hand landmarks
        # ----------------------------------------------------

        for hand in result.hand_landmarks:


            # Draw points

            for i, landmark in enumerate(hand):

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )


                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


                # Yellow numbers

                cv2.putText(
                    frame,
                    str(i),
                    (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    2
                )


            # ------------------------------------------------
            # Connections
            # ------------------------------------------------

            connections = [

                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),

                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),

                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),

                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),

                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),

                (5, 9),
                (9, 13),
                (13, 17)
            ]


            for start, end in connections:

                x1 = int(
                    hand[start].x * width
                )

                y1 = int(
                    hand[start].y * height
                )

                x2 = int(
                    hand[end].x * width
                )

                y2 = int(
                    hand[end].y * height
                )


                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )


        # ----------------------------------------------------
        # Number of hands
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Hands: {len(result.hand_landmarks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )


    # ========================================================
    # NO HAND
    # ========================================================

    else:

        cv2.putText(
            frame,
            "NO HAND DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "ISL Real-Time Landmark Recognition",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()


print()
print("==============================================")
print("Real-time test stopped.")
print("==============================================")

#CALCULATION OF RAM USAGE

current, peak = tracemalloc.get_traced_memory()
print(f"Peak RAM usage: {peak / 10**6:.2f} MB")
tracemalloc.stop()
#==============================================================