"""
Renderer utilities for drawing hand landmarks and labels onto OpenCV frames.

This module centralizes OpenCV and MediaPipe drawing logic used by the main
loop. It keeps drawing defensive (exceptions caught) so rendering problems
don't break the recognition loop.

Public API:
 - draw_landmarks_and_labels(frame, landmarks, gestures=None, handedness=None,
                            label_font_scale=1.0, label_color=(0,255,255))
 - draw_landmarks_only(frame, landmarks)
 - draw_labels_only(frame, frame_shape, landmarks, gestures=None, handedness=None,
                    font_scale=1.0, color=(0,255,255))

Notes:
 - `landmarks` is expected to be a sequence (list/tuple) where each item is a
   sequence-like collection of landmark objects. Each landmark object must expose
   `.x` and `.y` attributes (MediaPipe landmark objects satisfy this).
 - Coordinates are expected to be normalized in [0,1] relative to the image.
"""

from typing import Any, List, Optional, Sequence, Tuple

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - environment-dependent
    mp = None  # type: ignore

try:
    import cv2
except Exception:  # pragma: no cover - environment-dependent
    cv2 = None  # type: ignore


# Try to get MediaPipe drawing helpers if available. If not, drawing will be limited
_mp_drawing = None
_mp_hands = None
_mp_drawing_styles = None
if mp is not None:
    try:
        _mp_drawing = mp.tasks.vision.drawing_utils
        _mp_hands = mp.tasks.vision.HandLandmarksConnections
        _mp_drawing_styles = mp.tasks.vision.drawing_styles
    except Exception:
        # Keep None and fall back to minimal drawing behavior
        _mp_drawing = None
        _mp_hands = None
        _mp_drawing_styles = None


def _safe_put_text(
    frame: Any,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> None:
    """
    Safe wrapper for cv2.putText that avoids raising when cv2 is not available
    or when drawing fails.
    """
    try:
        if cv2 is None:
            return
        cv2.putText(
            frame,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    except Exception:
        # Never raise from rendering
        return


def draw_landmarks_only(frame: Any, landmarks: Sequence[Sequence[Any]]) -> None:
    """
    Draw MediaPipe landmarks and connections for every detected hand onto the frame.

    Parameters:
      - frame: OpenCV BGR image (numpy array).
      - landmarks: sequence of hand landmark sequences (each with landmarks having .x and .y).
    """
    if frame is None or not landmarks:
        return

    # If MediaPipe drawing helpers are available, use them for nicer visuals.
    if (
        _mp_drawing is not None
        and _mp_hands is not None
        and _mp_drawing_styles is not None
    ):
        for hand_landmarks in landmarks:
            try:
                _mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    _mp_hands.HAND_CONNECTIONS,
                    _mp_drawing_styles.get_default_hand_landmarks_style(),
                    _mp_drawing_styles.get_default_hand_connections_style(),
                )
            except Exception:
                # drawing should never break the application
                continue
    else:
        # Minimal fallback: draw small circles for each landmark if cv2 is available.
        if cv2 is None:
            return
        h, w = frame.shape[:2]
        for hand_landmarks in landmarks:
            try:
                for lm in hand_landmarks:
                    try:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                    except Exception:
                        continue
            except Exception:
                continue


def draw_labels_only(
    frame: Any,
    frame_shape: Tuple[int, int, int],
    landmarks: Sequence[Sequence[Any]],
    gestures: Optional[Sequence[Optional[str]]] = None,
    handedness: Optional[Sequence[Optional[str]]] = None,
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = (0, 255, 255),
) -> None:
    """
    Draw per-hand labels (handedness + gesture name) above the first landmark
    of each hand.

    Parameters:
      - frame: OpenCV BGR image
      - frame_shape: expected shape tuple (h, w, channels) or the frame itself will be used
      - landmarks: sequence of landmark sequences
      - gestures: optional sequence of gesture names aligned with landmarks
      - handedness: optional sequence of handedness names aligned with landmarks
      - font_scale: OpenCV font scale
      - color: text color (B,G,R)
    """
    if frame is None or not landmarks:
        return

    # allow frame_shape to be either a shape tuple or the frame; prefer reading frame.shape if available
    try:
        if hasattr(frame, "shape"):
            h, w = frame.shape[:2]
        else:
            h, w = frame_shape[0], frame_shape[1]
    except Exception:
        # fallback to safe defaults
        h, w = 480, 640

    for i, hand_landmarks in enumerate(landmarks):
        label = ""
        try:
            if handedness and i < len(handedness) and handedness[i]:
                label += f"{handedness[i]} "
            if gestures and i < len(gestures) and gestures[i]:
                label += f"{gestures[i]}"
        except Exception:
            # skip label composition if malformed
            label = ""

        if not label:
            continue

        # Compute location using landmark 0 (wrist) if available; else place near top-left of a hand bbox
        try:
            lm0 = hand_landmarks[0]
            x0 = int(lm0.x * w)
            y0 = int(lm0.y * h)
            # shift text slightly upward so it does not overlap landmarks
            text_pos = (max(0, x0), max(10, y0 - 10))
        except Exception:
            # fallback placement
            text_pos = (10 + i * 120, 30)

        _safe_put_text(
            frame, label, text_pos, font_scale=font_scale, color=color, thickness=2
        )


def draw_landmarks_and_labels(
    frame: Any,
    landmarks: Sequence[Sequence[Any]],
    gestures: Optional[Sequence[Optional[str]]] = None,
    handedness: Optional[Sequence[Optional[str]]] = None,
    label_font_scale: float = 1.0,
    label_color: Tuple[int, int, int] = (0, 255, 255),
) -> None:
    """
    Convenience wrapper that draws both landmarks+connections and labels.

    Parameters:
      - frame: OpenCV BGR image
      - landmarks: sequence of landmark sequences
      - gestures: optional sequence aligned with landmarks
      - handedness: optional sequence aligned with landmarks
      - label_font_scale: font size for labels
      - label_color: color for labels (B,G,R)
    """
    # Draw landmarks first (so labels are on top)
    try:
        draw_landmarks_only(frame, landmarks)
    except Exception:
        pass

    try:
        draw_labels_only(
            frame,
            getattr(frame, "shape", (480, 640, 3)),
            landmarks,
            gestures,
            handedness,
            font_scale=label_font_scale,
            color=label_color,
        )
    except Exception:
        pass


__all__ = [
    "draw_landmarks_and_labels",
    "draw_landmarks_only",
    "draw_labels_only",
]
