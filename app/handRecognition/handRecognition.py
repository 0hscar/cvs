import os
import threading
import time

import cv2
import mediapipe as mp
from camera.cameraCapture import CameraCapture
from gestureActions.actionHandler import handleGesture
from mediapipe.tasks.python import BaseOptions, vision

# Default from mediapipe docs
# modelPath = os.path.join(
#     os.path.dirname(__file__), "..", "..", "models", "gesture_recognizer.task"
# )
# modelPath = os.path.abspath(modelPath)

# Trained model path
modelPath = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "models",
    "exported",
    "gesture_model_29012026-134142",
    "gesture_recognizer.task",
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

# Mouse-tracking / smoothing state
# Try to import the normalized mouse mover helper and a monitor-detection helper.
# We import here to avoid circular imports at module load time.
try:
    from app.input.mouse_control import detect_monitors, move_mouse_normalized
except Exception:
    try:
        from input.mouse_control import detect_monitors, move_mouse_normalized
    except Exception:
        move_mouse_normalized = None
        detect_monitors = None

# Try to import hold controls from gesture actions (non-blocking; may not exist)
try:
    from gestureActions.gestures import start_hold, stop_hold
except Exception:
    try:
        # alternative import path if module layout differs
        from app.gestureActions.gestures import start_hold, stop_hold
    except Exception:
        start_hold = None
        stop_hold = None

# Smoothing: alpha in (0..1] where larger alpha makes the mouse follow more
# immediately. Typical values: 0.4-0.8. 0.6 is a good starting point.
MOUSE_SMOOTH_ALPHA = 0.6

# Sensitivity multiplier for fingertip movement.
# Values >1.0 make smaller finger movements map to larger cursor movement
# (amplifies movement around the center 0.5). Try 1.1-1.4 for modest increases.
MOUSE_SENSITIVITY = 1.2

# Per-hand last smoothed normalized coordinates (hand index -> (nx, ny))
_last_smoothed = {}

# Gesture name that enables tracking while active
TRACK_WHILE_GESTURE = "06_index"

# If camera image is flipped vertically relative to screen, set True.
# Default changed to False so hand down moves the cursor down.
INVERT_Y = False


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
    # Determine the target monitor geometry (prefer DP-1) and pass it as a fallback
    # to the mouse mover. detect_monitors() returns a list of tuples:
    # (name, width, height, offset_x, offset_y, is_primary)
    monitor_box = None
    try:
        # detect_monitors is provided by app.input.mouse_control; it may be None if import failed.
        if "detect_monitors" in globals() and callable(detect_monitors):
            mons = detect_monitors()
            if mons:
                for mon in mons:
                    # mon: (name, w, h, ox, oy, is_primary)
                    if mon[0] == "DP-1":
                        # move_mouse_normalized expects (w, h, ox, oy) as fallback_screen_size
                        monitor_box = (mon[1], mon[2], mon[3], mon[4])
                        break
                # If DP-1 not found, prefer primary or first monitor
                if monitor_box is None:
                    for mon in mons:
                        if mon[5]:
                            monitor_box = (mon[1], mon[2], mon[3], mon[4])
                            break
                if monitor_box is None and mons:
                    mon = mons[0]
                    monitor_box = (mon[1], mon[2], mon[3], mon[4])
    except Exception:
        monitor_box = None

    # Track previous gesture per-hand so we can detect transitions (enter/exit).
    _prev_gestures = {}

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

            # Detect transitions for held-click gesture ("03_fist"):
            # If gesture changed from not-fist -> fist: start_hold
            # If gesture changed from fist -> not-fist: stop_hold
            try:
                # ensure start_hold/stop_hold are available
                can_start = "start_hold" in globals() and callable(start_hold)
                can_stop = "stop_hold" in globals() and callable(stop_hold)
                # Ensure gestures list is aligned with landmarks length; iterate by index
                max_hands = max(len(gestures), len(landmarks))
                for hi in range(max_hands):
                    prev = _prev_gestures.get(hi)
                    curr = gestures[hi] if gestures and hi < len(gestures) else None
                    # Entering fist: start hold
                    if prev != "03_fist" and curr == "03_fist":
                        if can_start:
                            try:
                                start_hold(handIndex=hi)
                                print(
                                    f"[handRecognition] start_hold invoked for hand {hi}"
                                )
                            except Exception as e:
                                print(f"[handRecognition] start_hold error: {e}")
                    # Exiting fist: stop hold
                    if prev == "03_fist" and curr != "03_fist":
                        if can_stop:
                            try:
                                stop_hold()
                                print(
                                    f"[handRecognition] stop_hold invoked for hand {hi}"
                                )
                            except Exception as e:
                                print(f"[handRecognition] stop_hold error: {e}")
                    # Remember current gesture
                    if curr is not None:
                        _prev_gestures[hi] = curr
                    else:
                        # if no gesture detected for this hand, clear previous state
                        _prev_gestures.pop(hi, None)
            except Exception as e:
                # Defensive: do not let hold detection break the recognition loop
                print(f"[handRecognition] hold transition detection error: {e}")

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

                # If the configured TRACK_WHILE_GESTURE is active for this hand,
                # attempt to move the OS mouse to the index fingertip position.
                try:
                    if (
                        gestures
                        and i < len(gestures)
                        and gestures[i] == TRACK_WHILE_GESTURE
                    ):
                        if move_mouse_normalized is None:
                            # No available backend to move the mouse
                            # (move_mouse_normalized not available or failed to import)
                            # Do not raise; just skip tracking.
                            pass
                        else:
                            # MediaPipe index fingertip is landmark 8
                            if len(hand_landmarks) > 8:
                                lm = hand_landmarks[8]
                                nx = float(lm.x)
                                ny = float(lm.y)
                                if INVERT_Y:
                                    ny = 1.0 - ny
                                # Apply sensitivity multiplier (scale movement around center 0.5)
                                # This makes small finger movements map to larger cursor movements.
                                nx = 0.5 + (nx - 0.5) * MOUSE_SENSITIVITY
                                ny = 0.5 + (ny - 0.5) * MOUSE_SENSITIVITY
                                # Clamp inputs to valid normalized range
                                nx = max(0.0, min(1.0, nx))
                                ny = max(0.0, min(1.0, ny))
                                # Smooth with exponential moving average
                                prev = _last_smoothed.get(i, (nx, ny))
                                alpha = float(MOUSE_SMOOTH_ALPHA)
                                sx = prev[0] * (1.0 - alpha) + nx * alpha
                                sy = prev[1] * (1.0 - alpha) + ny * alpha
                                _last_smoothed[i] = (sx, sy)
                                # Debug: print normalized and smoothed coordinates, then compute abs coords for the chosen monitor
                                try:
                                    print(
                                        f"[handRecognition] hand {i} normalized (sx,sy)=({sx:.4f},{sy:.4f})"
                                    )
                                except Exception:
                                    pass

                                if "monitor_box" in locals() and monitor_box:
                                    # monitor_box is (w, h, ox, oy) passed from detect_monitors
                                    try:
                                        w_mon, h_mon, ox, oy = monitor_box
                                        abs_x = int(ox + sx * (w_mon - 1))
                                        abs_y = int(oy + sy * (h_mon - 1))
                                        print(
                                            f"[handRecognition] mapped to monitor_box abs=({abs_x},{abs_y}) monitor={w_mon}x{h_mon}+{ox}+{oy}"
                                        )
                                    except Exception as e:
                                        print(
                                            f"[handRecognition] failed to compute abs coords from monitor_box: {e}"
                                        )
                                    moved = move_mouse_normalized(
                                        sx, sy, fallback_screen_size=monitor_box
                                    )
                                else:
                                    # Try to compute which monitor would be used by detection and print the computed coords
                                    try:
                                        if "detect_monitors" in globals() and callable(
                                            detect_monitors
                                        ):
                                            mons = detect_monitors()
                                            if mons:
                                                # pick DP-1 if present else first monitor
                                                target = next(
                                                    (m for m in mons if m[0] == "DP-1"),
                                                    mons[0],
                                                )
                                                (
                                                    name,
                                                    w_mon,
                                                    h_mon,
                                                    ox,
                                                    oy,
                                                    is_primary,
                                                ) = target
                                                abs_x = int(ox + sx * (w_mon - 1))
                                                abs_y = int(oy + sy * (h_mon - 1))
                                                print(
                                                    f"[handRecognition] mapped to detected monitor {name} abs=({abs_x},{abs_y}) monitor={w_mon}x{h_mon}+{ox}+{oy}"
                                                )
                                    except Exception as e:
                                        print(
                                            f"[handRecognition] monitor detection/debug failed: {e}"
                                        )
                                    moved = move_mouse_normalized(sx, sy)
                                if not moved:
                                    # If movement failed once, do not spam, but print a short diagnostic.
                                    print(
                                        "[handRecognition] move_mouse_normalized failed"
                                    )
                            else:
                                # No landmark 8 available yet
                                pass
                except Exception as e:
                    # Defensive: do not allow tracking errors to break the recognition loop.
                    print(f"[handRecognition] tracking error: {e}")

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

    # Ensure any ongoing hold is released before exit
    try:
        if "stop_hold" in globals() and callable(stop_hold):
            stop_hold()
    except Exception as e:
        print(f"[handRecognition] final stop_hold error: {e}")
    camera.release()
