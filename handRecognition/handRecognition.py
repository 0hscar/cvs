import os
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from camera.cameraCapture import CameraCapture
from gestureActions.actions import handleGesture

modelPath = os.path.join(
    os.path.dirname(__file__), "..", "models", "gesture_recognizer.task"
)
modelPath = os.path.abspath(modelPath)

GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
VisionRunningMode = vision.RunningMode

# Shared state for the latest result
latest_result = {"gestures": [], "handedness": [], "landmarks": []}
lock = threading.Lock()

# MediaPipe Tasks API drawing utilities and connections
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles


def printResult(result, outputImage: mp.Image, timestampMs: int):
    gestures = []
    handedness = []
    landmarks = []

    if result.gestures:
        for i, hand_gestures in enumerate(result.gestures):
            if hand_gestures:
                gestures.append(hand_gestures[0].category_name)
            else:
                gestures.append(None)
    if result.handedness:
        for hand in result.handedness:
            if hand:
                handedness.append(hand[0].category_name)
            else:
                handedness.append(None)
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            landmarks.append(
                hand_landmarks
            )  # Store the list of landmark objects directly

    with lock:
        latest_result["gestures"] = gestures
        latest_result["handedness"] = handedness
        latest_result["landmarks"] = landmarks


def handRecognition():
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=modelPath),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=printResult,
        num_hands=2,
    )
    camera = CameraCapture()

    with GestureRecognizer.create_from_options(options) as recognizer:
        while camera.isOpened():
            shouldExit = False
            frame = camera.readFrame()
            if frame is None:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            recognizer.recognize_async(mp_image, timestamp_ms)

            # Draw landmarks and info
            with lock:
                gestures = latest_result["gestures"]
                handedness = latest_result["handedness"]
                landmarks = latest_result["landmarks"]

            for i, hand_landmarks in enumerate(landmarks):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,  # This is already the correct type from the Tasks API
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                # Display gesture and handedness
                label = ""
                if handedness and i < len(handedness):
                    label += handedness[i] + " "
                if gestures and i < len(gestures) and gestures[i]:
                    label += gestures[i]
                if label:
                    x0 = hand_landmarks[0].x
                    y0 = hand_landmarks[0].y

                    h, w, _ = frame.shape
                    cv2.putText(
                        frame,
                        label,
                        (int(x0 * w), int(y0 * h) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            for i, gesture in enumerate(gestures):
                if gesture:
                    hand = handedness[i] if handedness and i < len(handedness) else None
                    if handleGesture(gesture, handedness=hand, handIndex=i):
                        shouldExit = True
            camera.showFrame("Hand Gesture Recognition", frame)
            if camera.waitKey(1) == ord("q"):
                break
            if shouldExit:
                break

    camera.release()
