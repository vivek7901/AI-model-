#!/usr/bin/env python3
# MODIFICATION: Fixed the leftover model-input validation check to require 126 features.
# MODIFICATION: Adapted for isl_word_bilstm_attention.keras (30 x 126 input). Inference now uses only the 126 independently wrist-centred and scale-normalized left/right hand landmark values per frame; the 63 paired-delta values and 2 presence flags used by the 191-feature model are not appended. Step 3/4 diagnostics, capture controls, class labels, confidence threshold, and TTS retained.
# MODIFICATION: Step 4 - Added raw-to-30-frame resampling diagnostics and model-input validation. No change to resampling, features, model, threshold, class order, capture controls, or prediction behavior.
# MODIFICATION: Step 3 - Use real monotonic frame timestamps for MediaPipe and
# report captured-sequence timing/continuity diagnostics. Model preprocessing,
# class order, confidence threshold, camera controls, and capture stop rules remain unchanged.

"""Receive CSI camera frames and recognize isolated ISL words, with text-to-speech."""



import socket

import struct

import time

import queue

import threading



import cv2

import mediapipe as mp

import numpy as np

import tensorflow as tf

import pyttsx3



from mediapipe.tasks import python

from mediapipe.tasks.python import vision



HOST = "127.0.0.1"

PORT = 8765

MODEL_PATH = "/home/admin4/isl_improvement/isl_word_bilstm_attention.keras"

HAND_MODEL_PATH = "/home/admin4/isl_improvement/hand_landmarker.task"



CLASS_NAMES = [

    "HELLO", "THANK_YOU", "HELP", "YES", "NO",

    "WATER", "EAT", "GO", "YOU", "TODAY"

]



NO_HAND_FRAMES_TO_FINISH = 4

MIN_ACTIVE_FRAMES = 5

PREDICTION_THRESHOLD = 0.50



# Gesture-control settings copied from rpi_gesture_controls_test.py

STABLE_FRAMES = 5

RELEASE_FRAMES = 4

GESTURE_COOLDOWN = 0.8



# Timing that separates control gestures from sign capture.

START_COUNTDOWN_SECONDS = 3.0

END_GRACE_SECONDS = 2.0

NO_HAND_FRAMES_TO_CONFIRM_END = 4



# Speech runs in a background thread so audio playback does not block video.

# Reliable background speech

TTS_ENABLED = True

TTS_RATE = 150

TTS_VOLUME = 1.0



speech_queue = queue.Queue()

tts_ready = threading.Event()





def speech_worker():

    engine = None



    try:

        print("[TTS] Initializing speech engine...", flush=True)



        engine = pyttsx3.init(driverName="espeak")

        engine.setProperty("rate", TTS_RATE)

        engine.setProperty("volume", TTS_VOLUME)



        tts_ready.set()

        print("[TTS] Engine ready!", flush=True)



    except Exception as exc:

        print(f"[TTS ERROR] Initialization failed: {exc}", flush=True)

        return



    while True:

        word = speech_queue.get()



        try:

            if word is None:

                break



            spoken_text = word.replace("_", " ").lower()



            print(f"[TTS] Speaking: {spoken_text}", flush=True)



            engine.say(spoken_text)

            engine.runAndWait()



            print("[TTS] Speech completed.", flush=True)



        except Exception as exc:

            print(f"[TTS ERROR] Playback failed: {exc}", flush=True)



        finally:

            speech_queue.task_done()



    engine.stop()





def speak_word(word):

    if TTS_ENABLED:

        if tts_ready.is_set():

            speech_queue.put(word)

            print(f"[TTS] Queued: {word}", flush=True)

        else:

            print("[TTS WARNING] Speech engine is not ready!", flush=True)



def receive_exact(connection, length):

    """Receive exactly length bytes, or return None if connection closes."""

    chunks = []

    remaining = length

    while remaining:

        chunk = connection.recv(remaining)

        if not chunk:

            return None

        chunks.append(chunk)

        remaining -= len(chunk)

    return b"".join(chunks)





def frame_reader(connection, frame_state, frame_lock, frame_ready):

    """Continuously read the socket and retain only the newest JPEG frame."""

    try:

        while True:

            header = receive_exact(connection, 4)

            if header is None:

                break



            frame_length = struct.unpack("!I", header)[0]

            if frame_length <= 0 or frame_length > 10_000_000:

                raise RuntimeError(f"Invalid frame size received: {frame_length}")



            jpg_bytes = receive_exact(connection, frame_length)

            if jpg_bytes is None:

                break



            with frame_lock:

                frame_state["jpeg"] = jpg_bytes

                frame_state["counter"] += 1

                frame_ready.set()



    except (OSError, RuntimeError) as exc:

        with frame_lock:

            frame_state["error"] = str(exc)

    finally:

        with frame_lock:

            frame_state["closed"] = True

            frame_ready.set()





def has_hand(frame_landmarks):

    return not (np.allclose(frame_landmarks[:63], 0) and

                np.allclose(frame_landmarks[63:], 0))





def normalize_frame(frame):

    output = np.zeros_like(frame, dtype=np.float32)

    for start in (0, 63):

        hand = frame[start:start + 63].reshape(21, 3).copy()

        if np.allclose(hand, 0):

            continue

        hand -= hand[0].copy()

        scale = np.max(np.linalg.norm(hand, axis=1))

        if scale > 1e-8:

            hand /= scale

        output[start:start + 63] = hand.reshape(-1)

    return output





def extract_landmarks(landmarker, rgb_frame, timestamp_ms):

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    values = np.zeros(126, dtype=np.float32)



    for index, hand_landmarks in enumerate(result.hand_landmarks):

        coords = np.asarray(

            [[point.x, point.y, point.z] for point in hand_landmarks],

            dtype=np.float32

        ).reshape(-1)

        handedness = ""

        if index < len(result.handedness) and result.handedness[index]:

            handedness = result.handedness[index][0].category_name

        if handedness == "Left":

            values[:63] = coords

        elif handedness == "Right":

            values[63:] = coords

        else:

            if np.allclose(values[:63], 0):

                values[:63] = coords

            else:

                values[63:] = coords

    return values, result.hand_landmarks





def report_resampling_diagnostics(sequence, target_length):
    """
    Step 4 diagnostics only. This reports exactly how the captured raw
    sequence is cropped and mapped to target_length frames. It does not
    alter the sequence or prediction.
    """
    raw = np.asarray(sequence, dtype=np.float32)

    if raw.ndim != 2 or raw.shape[1] != 126:
        print(
            "[RESAMPLING CHECK] Invalid raw shape:",
            raw.shape,
            "(expected frames x 126)",
            flush=True
        )
        return

    active_indices = [
        i for i, frame in enumerate(raw) if has_hand(frame)
    ]

    if len(active_indices) < MIN_ACTIVE_FRAMES:
        print(
            "[RESAMPLING CHECK] Not enough active frames:",
            len(active_indices),
            flush=True
        )
        return

    first_active = active_indices[0]
    last_active = active_indices[-1]
    cropped = raw[first_active:last_active + 1]

    # Deliberately identical to the training and inference resampling rule.
    sampled_indices = np.linspace(
        0, len(cropped) - 1, target_length
    ).astype(int)

    unique_indices = np.unique(sampled_indices)
    repeated_selections = target_length - len(unique_indices)
    source_frames_not_selected = len(cropped) - len(unique_indices)

    left_count = 0
    right_count = 0
    both_count = 0
    no_hand_count = 0

    for index in sampled_indices:
        frame = cropped[index]
        left_present = not np.allclose(frame[:63], 0)
        right_present = not np.allclose(frame[63:], 0)

        if left_present and right_present:
            both_count += 1
        elif left_present:
            left_count += 1
        elif right_present:
            right_count += 1
        else:
            no_hand_count += 1

    if len(cropped) > target_length:
        mode = "DOWNSAMPLING"
    elif len(cropped) < target_length:
        mode = "UPSAMPLING (some source frames repeated)"
    else:
        mode = "1:1 FRAME MAPPING"

    print("\n[STEP 4 - RESAMPLING VALIDATION]", flush=True)
    print(
        f"Raw captured frames: {len(raw)} | "
        f"Active frame range: {first_active}-{last_active} | "
        f"Cropped frames: {len(cropped)}",
        flush=True
    )
    print(
        f"Target frames: {target_length} | Mode: {mode}",
        flush=True
    )
    print(
        f"Unique source frames selected: {len(unique_indices)} | "
        f"Repeated selections: {repeated_selections} | "
        f"Source frames not selected: {source_frames_not_selected}",
        flush=True
    )
    print(
        "Hand presence among sampled frames: "
        f"left-only={left_count}, right-only={right_count}, "
        f"both={both_count}, neither={no_hand_count}",
        flush=True
    )
    print(
        "First 5 selected cropped-frame indices:",
        sampled_indices[:5].tolist(),
        "| Last 5:",
        sampled_indices[-5:].tolist(),
        flush=True
    )
    print(
        "[RESAMPLING CHECK] Uses the same "
        "np.linspace(...).astype(int) rule as training.",
        flush=True
    )


def prepare_sequence(sequence, target_length):
    """
    Build the 126-feature representation expected by
    isl_word_bilstm_attention.keras.

    Feature order per frame:
      0:63    left-hand landmarks, wrist-centred and scale-normalized
      63:126  right-hand landmarks, wrist-centred and scale-normalized

    Important:
      This intentionally does NOT append the 63 paired spatial-delta values
      or the 2 hand-presence flags used by the spatial 191-feature model.
      Cropping and raw-frame resampling retain the same training/inference
      rule used by the existing script.
    """
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.size == 0:
        return None

    if sequence.ndim != 2 or sequence.shape[1] != 126:
        raise ValueError(
            f"Expected raw sequence shape (frames, 126), got {sequence.shape}"
        )

    active = [i for i, frame in enumerate(sequence) if has_hand(frame)]

    if len(active) < MIN_ACTIVE_FRAMES:
        return None

    # Match the existing preprocessing: crop to first/last active frame,
    # then resample raw 126-value frames to the model's required length.
    sequence = sequence[active[0]:active[-1] + 1]
    indices = np.linspace(0, len(sequence) - 1, target_length).astype(int)
    sampled_raw = sequence[indices]

    # The 126-feature model receives only the normalized left/right landmarks.
    prepared = np.asarray(
        [normalize_frame(frame) for frame in sampled_raw],
        dtype=np.float32
    )

    if prepared.shape != (target_length, 126):
        raise RuntimeError(
            f"Internal feature shape error: got {prepared.shape}, "
            f"expected ({target_length}, 126)"
        )

    return prepared



def finger_extended(hand, tip, pip):

    """Check whether a finger is extended in image coordinates."""

    return hand[tip].y < hand[pip].y - 0.025





def classify_gesture(hand):

    """Return OPEN_PALM, V_SIGN, THUMBS_UP, or None."""

    index = finger_extended(hand, 8, 6)

    middle = finger_extended(hand, 12, 10)

    ring = finger_extended(hand, 16, 14)

    pinky = finger_extended(hand, 20, 18)



    thumb_up = (

        hand[4].y < hand[3].y - 0.025

        and hand[4].y < hand[2].y - 0.04

    )



    if thumb_up and not index and not middle and not ring and not pinky:

        return "THUMBS_UP"

    if index and middle and not ring and not pinky:

        return "V_SIGN"

    if index and middle and ring and pinky:

        return "OPEN_PALM"

    return None





def draw_hand_landmarks(frame, hand):

    """Draw the control hand, matching the standalone gesture test preview."""

    height, width = frame.shape[:2]

    points = [

        (int(landmark.x * width), int(landmark.y * height))

        for landmark in hand

    ]

    for start, end in mp.solutions.hands.HAND_CONNECTIONS:

        cv2.line(frame, points[start], points[end], (80, 220, 80), 2)

    for x, y in points:

        cv2.circle(frame, (x, y), 3, (0, 180, 255), -1)







def main():

    speech_thread = None

    connection = None



    if TTS_ENABLED:

        speech_thread = threading.Thread(target=speech_worker, daemon=True)

        speech_thread.start()



    print("Loading TensorFlow model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded. Input shape:", model.input_shape, "Output shape:", model.output_shape)



    input_shape = model.input_shape

    if isinstance(input_shape, list):

        raise RuntimeError("This script expects a single-input Keras model.")

    target_length = input_shape[1] or 30

    feature_count = input_shape[2]

    if feature_count != 126:
        raise RuntimeError(
            f"This script generates 126 features, but model expects {feature_count}. "
            "Use the matching isl_word_bilstm_attention.keras model."
        )

    if target_length != 30:

        raise RuntimeError(

            f"Training used 30 frames, but model expects {target_length}. "

            "Verify the training configuration before proceeding."

        )

    output_count = model.output_shape[-1]

    if output_count != len(CLASS_NAMES):

        raise RuntimeError(

            f"Model has {output_count} outputs, but CLASS_NAMES contains "

            f"{len(CLASS_NAMES)} labels. Verify the original class order."

        )



    options = vision.HandLandmarkerOptions(

        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),

        running_mode=vision.RunningMode.VIDEO,

        num_hands=2

    )



    try:

        with vision.HandLandmarker.create_from_options(options) as landmarker:

            print(f"Connecting to camera server at {HOST}:{PORT}...")

            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            try:

                for attempt in range(20):

                    try:

                        connection.connect((HOST, PORT))

                        break

                    except ConnectionRefusedError:

                        if attempt == 19:

                            raise

                        time.sleep(0.5)



                print("Connected to camera server.")

                print("OPEN PALM: begin a 3-second start countdown.")

                print("Lower your hand / move it out of view to stop recording.")

                print("Prediction follows after a 2-second delay.")

                print("THUMBS UP: exit.")

                print("Press Q or ESC in the video window to quit.")



                # Retain the existing camera reader and its single socket client.

                frame_state = {

                    "jpeg": None,

                    "counter": 0,

                    "closed": False,

                    "error": None,

                }

                frame_lock = threading.Lock()

                frame_ready = threading.Event()

                reader_thread = threading.Thread(

                    target=frame_reader,

                    args=(connection, frame_state, frame_lock, frame_ready),

                    daemon=True,

                )

                reader_thread.start()

                last_frame_counter = 0



                # Open Palm starts capture; no-hand detection ends capture; Thumbs Up exits.

                state = "WAITING"

                countdown_deadline = None

                grace_deadline = None

                no_hand_end_count = 0

                stable_candidate = None

                stable_count = 0

                no_gesture_count = 0

                latched = False

                last_action_time = 0.0

                prediction_count = 0

                capture_start = None



                # Existing recognition state.

                sequence = []

                timestamp_ms = 0

                latest_word = "Show OPEN PALM to start"

                latest_score = 0.0

                latest_gesture = "None"

                should_exit = False



                while not should_exit:

                    frame_ready.wait(timeout=1.0)



                    with frame_lock:

                        if frame_state["counter"] == last_frame_counter:

                            if frame_state["closed"]:

                                if frame_state["error"]:

                                    print("Camera reader stopped:", frame_state["error"])

                                else:

                                    print("Camera server closed the connection.")

                                break

                            frame_ready.clear()

                            continue



                        jpg_bytes = frame_state["jpeg"]

                        last_frame_counter = frame_state["counter"]

                        frame_ready.clear()



                    if jpg_bytes is None:

                        continue



                    frame_bgr = cv2.imdecode(

                        np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR

                    )

                    if frame_bgr is None:

                        continue



                    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                    rgb_small = cv2.resize(

                        rgb_frame,

                        (320, 240),

                        interpolation=cv2.INTER_LINEAR,

                    )



                    timestamp_ms += 33

                    start_time = time.perf_counter()

                    landmarks, detected_hands = extract_landmarks(

                        landmarker, rgb_small, timestamp_ms

                    )

                    processing_ms = (time.perf_counter() - start_time) * 1000



                    if processing_ms > 50:

                        print(f"MediaPipe processing: {processing_ms:.1f} ms")



                    # The first detected hand controls the app, matching the

                    # standalone test. The same detection supplies ML landmarks.

                    detected_gesture = None

                    if detected_hands:

                        control_hand = detected_hands[0]

                        detected_gesture = classify_gesture(control_hand)

                        draw_hand_landmarks(frame_bgr, control_hand)



                    latest_gesture = detected_gesture or "None"



                    # Five stable frames, four release frames, and 0.8s cooldown.

                    if detected_gesture:

                        no_gesture_count = 0

                        if detected_gesture == stable_candidate:

                            stable_count += 1

                        else:

                            stable_candidate = detected_gesture

                            stable_count = 1

                    else:

                        stable_count = 0

                        stable_candidate = None

                        no_gesture_count += 1

                        if no_gesture_count >= RELEASE_FRAMES:

                            latched = False



                    now = time.monotonic()

                    if (

                        detected_gesture

                        and stable_count >= STABLE_FRAMES

                        and not latched

                        and now - last_action_time >= GESTURE_COOLDOWN

                    ):

                        latched = True

                        last_action_time = now



                        if detected_gesture == "THUMBS_UP":

                            print("\nTHUMBS UP detected: exiting.")

                            should_exit = True



                        elif state == "WAITING" and detected_gesture == "OPEN_PALM":

                            state = "COUNTDOWN"

                            sequence = []

                            capture_start = None

                            countdown_deadline = now + START_COUNTDOWN_SECONDS

                            latest_word = "Get ready..."

                            latest_score = 0.0

                            print(

                                f"\\\nOPEN PALM detected: recording starts in "

                                f"{START_COUNTDOWN_SECONDS:.0f} seconds."

                            )





                    # Capture sign frames while at least one hand is detected.

                    # A short detection dropout is tolerated; no-hand frames are

                    # not added to the model sequence.

                    # After countdown, wait for the signing hand to appear.
                    # This avoids ending an empty capture while the hand is lowered.
                    if state == "ARMED":
                        if detected_hands:
                            state = "CAPTURING"
                            capture_start = now
                            sequence = [landmarks]
                            sequence_timestamps = [now]
                            no_hand_end_count = 0
                            latest_word = "Recording sign..."
                            print("Hand detected: sign recording started.")

                    elif state == "CAPTURING":
                        if detected_hands:
                            sequence.append(landmarks)
                            sequence_timestamps.append(now)
                            no_hand_end_count = 0
                        else:
                            no_hand_end_count += 1

                        if (len(sequence) >= MIN_ACTIVE_FRAMES and
                                no_hand_end_count >= NO_HAND_FRAMES_TO_CONFIRM_END):
                            state = "PREDICTION_DELAY"
                            grace_deadline = now + END_GRACE_SECONDS
                            latest_word = "Hand lowered - preparing prediction..."
                            print(
                                f"No hand detected for "
                                f"{NO_HAND_FRAMES_TO_CONFIRM_END} consecutive frames. "
                                f"Capture stopped; predicting in "
                                f"{END_GRACE_SECONDS:.0f} seconds."
                            )

                    # Non-blocking 3-second start countdown.

                    if state == "COUNTDOWN" and countdown_deadline is not None:

                        if now >= countdown_deadline:

                            state = "ARMED"

                            capture_start = None

                            countdown_deadline = None

                            no_hand_end_count = 0

                            latest_word = "Show your hand to begin signing"

                            print("Countdown complete: waiting for the signing hand.")



                    # Non-blocking 2-second prediction delay. The captured sequence

                    # remains frozen; no post-sign or hand-lowering frames are added.

                    if state == "PREDICTION_DELAY" and grace_deadline is not None:

                        if now >= grace_deadline:

                            duration = (

                                now - capture_start if capture_start else 0.0

                            )

                            # Step 3 diagnostics: report the actual captured sequence
                            # before the existing crop/resample-to-30 preprocessing.
                            captured_count = len(sequence)
                            if len(sequence_timestamps) >= 2:
                                captured_duration = (
                                    sequence_timestamps[-1] - sequence_timestamps[0]
                                )
                                intervals = np.diff(sequence_timestamps)
                                mean_interval = float(np.mean(intervals))
                                effective_fps = (
                                    (captured_count - 1) / captured_duration
                                    if captured_duration > 0 else 0.0
                                )
                                largest_gap = float(np.max(intervals))
                            else:
                                captured_duration = 0.0
                                mean_interval = 0.0
                                effective_fps = 0.0
                                largest_gap = 0.0

                            print(
                                "[CAPTURE DIAGNOSTICS] "
                                f"frames={captured_count}, "
                                f"duration={captured_duration:.2f}s, "
                                f"effective_fps={effective_fps:.1f}, "
                                f"mean_gap={mean_interval * 1000:.1f}ms, "
                                f"largest_gap={largest_gap * 1000:.1f}ms",
                                flush=True
                            )

                            # Step 4: inspect the exact 30-frame mapping before
                            # preprocessing. This is diagnostic only and does not
                            # modify the captured sequence or prediction.
                            report_resampling_diagnostics(sequence, target_length)

                            prepared = prepare_sequence(sequence, target_length)

                            if prepared is not None:
                                print(
                                    "[MODEL INPUT CHECK] 126-feature model input:",
                                    prepared.shape,
                                    "| finite values:",
                                    bool(np.isfinite(prepared).all()),
                                    flush=True
                                )



                            if prepared is not None:

                                probabilities = np.asarray(

                                    model.predict(

                                        prepared[np.newaxis, ...], verbose=0

                                    )

                                )[0]

                                best_index = int(np.argmax(probabilities))

                                latest_score = float(probabilities[best_index])

                                prediction_count += 1



                                if latest_score >= PREDICTION_THRESHOLD:

                                    latest_word = CLASS_NAMES[best_index]

                                    print(

                                        f"Prediction #{prediction_count}: "

                                        f"{latest_word} ({latest_score:.1%})"

                                    )

                                    speak_word(latest_word)

                                else:

                                    latest_word = (

                                        f"Uncertain: {CLASS_NAMES[best_index]}"

                                    )

                                    print(

                                        f"Low confidence: {CLASS_NAMES[best_index]} "

                                        f"({latest_score:.1%})"

                                    )

                            else:

                                latest_word = "Sign too short"

                                latest_score = 0.0

                                print(

                                    "Sign too short: not enough active hand frames "

                                    "for prediction."

                                )



                            print(

                                f"Prediction delay finished "

                                f"(recording duration {duration:.1f}s)."

                            )

                            print("Returning to WAITING for the next OPEN PALM.")

                            state = "WAITING"

                            sequence = []

                            capture_start = None

                            grace_deadline = None

                            no_hand_end_count = 0



                    if should_exit:

                        break



                    if state == "WAITING":

                        status = "WAITING - show OPEN PALM"

                    elif state == "COUNTDOWN":

                        remaining = max(0.0, countdown_deadline - time.monotonic())

                        status = f"STARTING IN {remaining:.1f}s - get ready"

                    elif state == "ARMED":

                        status = "READY - show your hand to start signing"

                    elif state == "CAPTURING":

                        status = "CAPTURING - lower hand when sign is complete"

                    else:

                        remaining = max(0.0, grace_deadline - time.monotonic())

                        status = f"NO HAND - PREDICTING IN {remaining:.1f}s"

                    cv2.putText(

                        frame_bgr, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,

                        0.65, (0, 255, 255), 2, cv2.LINE_AA

                    )

                    cv2.putText(

                        frame_bgr, latest_word, (10, 60),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,

                        (0, 255, 0), 2, cv2.LINE_AA

                    )

                    if latest_score > 0:

                        cv2.putText(

                            frame_bgr, f"Confidence: {latest_score:.1%}", (10, 90),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,

                            (0, 255, 0), 2, cv2.LINE_AA

                        )

                    cv2.putText(

                        frame_bgr,

                        f"Gesture: {latest_gesture} | Predictions: {prediction_count}",

                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,

                        (255, 255, 255), 1, cv2.LINE_AA

                    )

                    cv2.putText(

                        frame_bgr,

                        "OPEN PALM = START | LOWER HAND = PREDICT | THUMBS UP = EXIT",

                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.42,

                        (255, 255, 255), 1, cv2.LINE_AA

                    )



                    cv2.imshow("ISL Word Recognition", frame_bgr)

                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord("q"), 27):

                        should_exit = True



            finally:

                if connection is not None:

                    try:

                        connection.shutdown(socket.SHUT_RDWR)

                    except OSError:

                        pass

                    connection.close()

                cv2.destroyAllWindows()

                print("Recognition stopped.")



    finally:

        if TTS_ENABLED and speech_thread is not None:

            speech_queue.put(None)

            speech_thread.join(timeout=3)





if __name__ == "__main__":

    main()


