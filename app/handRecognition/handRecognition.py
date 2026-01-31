import os
import time

import cv2
import mediapipe as mp
from camera.cameraCapture import CameraCapture
from gestureActions.actionHandler import handleGesture

# Use our new small modules
try:
    from .hold_manager import HoldManager
    from .recognizer import RecognizerRunner
    from .renderer import draw_landmarks_and_labels
    from .result_store import ResultStore
    from .tracker import Tracker
except Exception:
    # fall back to absolute imports if package is executed differently
    from handRecognition.hold_manager import HoldManager  # type: ignore
    from handRecognition.recognizer import RecognizerRunner  # type: ignore
    from handRecognition.renderer import draw_landmarks_and_labels  # type: ignore
    from handRecognition.result_store import ResultStore  # type: ignore
    from handRecognition.tracker import Tracker  # type: ignore

# Trained model path (same as before)
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

# MediaPipe Tasks drawing utilities
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

# Try to import the normalized mouse mover helper and a monitor-detection helper.
# Keep the same import fallbacks to remain compatible with original layout.
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
        from app.gestureActions.gestures import start_hold, stop_hold
    except Exception:
        start_hold = None
        stop_hold = None

# Parameter defaults (same semantics as before)
MOUSE_SMOOTH_ALPHA = 0.6
MOUSE_SENSITIVITY = 1.2
TRACK_WHILE_GESTURE = "06_index"
INVERT_Y = False


def handRecognition():
    """
    Orchestrator: compose ResultStore, RecognizerRunner, Tracker and camera loop.

    This preserves the original behavior but delegates storage, recognition callback
    handling, and mouse tracking to small, focused classes.
    """
    # Create shared result store and recognizer runner
    rs = ResultStore()
    runner = RecognizerRunner(model_path=modelPath, result_store=rs, num_hands=2)

    # Initialize tracker with injected backends
    tracker = Tracker(
        move_mouse_normalized=move_mouse_normalized,
        detect_monitors=detect_monitors,
        smooth_alpha=MOUSE_SMOOTH_ALPHA,
        sensitivity=MOUSE_SENSITIVITY,
        invert_y=INVERT_Y,
        track_gesture_name=TRACK_WHILE_GESTURE,
    )

    camera = CameraCapture()

    # Determine a preferred monitor_box (w,h,ox,oy) to pass as fallback when available
    monitor_box = None
    try:
        if callable(detect_monitors):
            mons = detect_monitors()
            if mons:
                for mon in mons:
                    if mon[0] == "DP-1":
                        monitor_box = (mon[1], mon[2], mon[3], mon[4])
                        break
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

    # Use HoldManager to detect hold transitions (replaces manual _prev_gestures tracking)
    hold_manager = HoldManager(
        start_hold=start_hold, stop_hold=stop_hold, hold_gesture_name="03_fist"
    )

    # Create the recognizer context and run the main loop
    with runner.create_recognizer() as recognizer:
        while camera.isOpened():
            shouldExit = False
            frame = camera.readFrame()
            if frame is None:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)

            # Submit the image asynchronously; the RecognizerRunner callback will update ResultStore
            runner.recognize_async(recognizer, mp_image, timestamp_ms)

            # Snapshot results from the result store (thread-safe)
            gestures, handedness, landmarks, _ts = rs.snapshot()

            # Hold transition detection: delegate to HoldManager which encapsulates per-hand state
            try:
                hold_manager.update(gestures, handedness, landmarks)
            except Exception as e:
                print(f"[handRecognition] hold transition detection error: {e}")

            # Draw landmarks and labels using the renderer module (if available).
            # This keeps rendering logic in one place and reduces duplication.
            try:
                if callable(draw_landmarks_and_labels):
                    try:
                        draw_landmarks_and_labels(
                            frame,
                            landmarks,
                            gestures,
                            handedness,
                            label_font_scale=1.0,
                            label_color=(0, 255, 255),
                        )
                    except Exception:
                        # Defensive: do not let rendering break the loop
                        pass
            except Exception:
                # If draw_landmarks_and_labels is not defined or fails, continue without raising
                pass

            # Track each detected hand (Tracker handles gating by gesture)
            for i, hand_landmarks in enumerate(landmarks):
                try:
                    gesture_name = (
                        gestures[i] if gestures and i < len(gestures) else None
                    )
                    moved = tracker.track_hand(
                        hand_index=i,
                        hand_landmarks=hand_landmarks,
                        gesture_name=gesture_name,
                        monitor_box=monitor_box,
                        require_gesture=True,
                    )
                    # moved boolean is informational; debug prints are inside Tracker
                except Exception as e:
                    print(f"[handRecognition] tracking error: {e}")

            # Dispatch gesture actions using the existing handler
            for i, gesture in enumerate(gestures):
                if gesture:
                    hand = handedness[i] if handedness and i < len(handedness) else None
                    try:
                        if handleGesture(gesture, handedness=hand, handIndex=i):
                            shouldExit = True
                    except Exception as e:
                        print(f"[handRecognition] action handler error: {e}")

            # Show frame and handle input keys
            camera.showFrame("Hand Gesture Recognition", frame)
            if camera.waitKey(1) == ord("q"):
                break
            if shouldExit:
                break

    # Ensure any ongoing hold is released before exit (ask HoldManager to release)
    try:
        hold_manager.release_all()
    except Exception as e:
        print(f"[handRecognition] final hold_manager release_all error: {e}")

    camera.release()
