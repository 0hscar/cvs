"""
Recognizer wrapper for MediaPipe Tasks GestureRecognizer.

This module provides a small, testable wrapper around the MediaPipe
GestureRecognizer Tasks API. It converts the Tasks callback results into
simple Python lists and pushes them into a provided ResultStore instance.

Public API:
- RecognizerRunner: main class. Create with a model path and a ResultStore,
  then use `create_options()` to get options for creating a recognizer or
  call `create_recognizer()` as a context manager to obtain a recognizer
  instance. Use `recognize_async(mp_image, timestamp_ms)` to submit images.

This wrapper intentionally keeps the callback simple and avoids any
application-level logic (drawing, mouse tracking, action handling).
Those responsibilities belong to higher-level modules.
"""

import contextlib
from typing import Any, Callable, Optional, Sequence

# Try to import MediaPipe Tasks API. Keep imports localized to fail fast if MP is missing.
try:
    import mediapipe as mp  # used for mp.Image type hints and ImageFormat enum
    from mediapipe.tasks.python import BaseOptions, vision
except Exception as e:  # pragma: no cover - environment-dependent
    # Provide helpful import-time error while keeping module importable in static analysis.
    raise ImportError(
        "MediaPipe Tasks Python API is required by recognizer.py. "
        "Ensure `mediapipe` is installed and the Tasks API is available."
    ) from e

# Import the thread-safe result store defined in the package.
try:
    # Prefer relative import when used as a package
    from .result_store import ResultStore
except Exception:
    # Fall back to absolute import if module executed differently
    from handRecognition.result_store import ResultStore  # type: ignore

# Types from Tasks API
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
VisionRunningMode = vision.RunningMode

# Signature of the callback that the Tasks API expects:
# def result_callback(result, output_image: mp.Image, timestamp_ms: int): ...
ResultCallbackType = Callable[[Any, mp.Image, int], None]


class RecognizerRunner:
    """
    Wrapper that creates a GestureRecognizer and forwards callback results
    to a provided ResultStore.

    Example usage:
        rs = ResultStore()
        runner = RecognizerRunner(model_path="/path/to/gesture_recognizer.task", result_store=rs)
        options = runner.create_options()
        with GestureRecognizer.create_from_options(options) as recognizer:
            # create mp.Image and call:
            recognizer.recognize_async(mp_image, timestamp_ms)
            # ResultStore will be populated by the background callback.

    The callback used by the Tasks API runs on a background thread internal to
    MediaPipe. ResultStore is thread-safe and designed to be used from multiple
    threads.
    """

    def __init__(self, model_path: str, result_store: ResultStore, num_hands: int = 2):
        """
        Args:
            model_path: Path to a compiled MediaPipe Tasks model asset (.task).
            result_store: Instance of ResultStore where parsed results will be stored.
            num_hands: Number of hands to request from the recognizer (default 2).
        """
        self.model_path = model_path
        self.result_store = result_store
        self.num_hands = int(num_hands)
        # Keep a stable reference to the bound callback so it doesn't get GC'd
        self._bound_callback: Optional[ResultCallbackType] = None

    def _make_callback(self) -> ResultCallbackType:
        """
        Create and return a callback function suitable for GestureRecognizerOptions.
        The callback parses the Task API types into simple Python lists and stores
        them in the ResultStore.
        """

        def _callback(result: Any, output_image: mp.Image, timestamp_ms: int) -> None:
            gestures = []
            handedness = []
            landmarks = []

            # result.gestures is a sequence-of-sequences for each detected hand.
            try:
                if getattr(result, "gestures", None):
                    for hand_gestures in result.gestures:
                        if hand_gestures:
                            # hand_gestures[0] is the top category
                            try:
                                gestures.append(hand_gestures[0].category_name)
                            except Exception:
                                gestures.append(None)
                        else:
                            gestures.append(None)
                if getattr(result, "handedness", None):
                    for hand in result.handedness:
                        if hand:
                            try:
                                handedness.append(hand[0].category_name)
                            except Exception:
                                handedness.append(None)
                        else:
                            handedness.append(None)
                if getattr(result, "hand_landmarks", None):
                    for hand_landmarks in result.hand_landmarks:
                        landmarks.append(hand_landmarks)
            except Exception:
                # Defensive: never allow callback to raise into MediaPipe; ensure store receives something.
                try:
                    # ensure we at least set empty lists to avoid stale state
                    self.result_store.set([], [], [])
                except Exception:
                    # Last resort: swallow exceptions — printing would be noisy in library code.
                    pass
            else:
                # Update the result store atomically with the parsed snapshot.
                try:
                    self.result_store.set(gestures, handedness, landmarks)
                except Exception:
                    # If setting the store fails, do not raise into MediaPipe.
                    pass

        return _callback

    def create_options(self) -> GestureRecognizerOptions:
        """
        Create and return a GestureRecognizerOptions object that can be used to
        instantiate a GestureRecognizer via the Tasks API.

        The returned options object is safe to pass to `GestureRecognizer.create_from_options`.
        """
        if self._bound_callback is None:
            self._bound_callback = self._make_callback()

        return GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._bound_callback,
            num_hands=self.num_hands,
        )

    @contextlib.contextmanager
    def create_recognizer(self):
        """
        Context manager that yields a created GestureRecognizer instance.

        Usage:
            with runner.create_recognizer() as recognizer:
                recognizer.recognize_async(mp_image, timestamp_ms)

        The returned object is the same type you would get from
        `GestureRecognizer.create_from_options(options)`.
        """
        options = self.create_options()
        with GestureRecognizer.create_from_options(options) as recognizer:
            yield recognizer

    # Convenience wrappers -------------------------------------------------

    def recognize_async(
        self,
        recognizer: GestureRecognizer,
        mp_image: mp.Image,
        timestamp_ms: Optional[int] = None,
    ) -> None:
        """
        Submit an mp.Image for asynchronous recognition.

        This is a thin wrapper that computes a current timestamp_ms if none is provided
        and forwards to the recognizer's `recognize_async` method.

        Args:
            recognizer: An instance created by the Tasks API (GestureRecognizer).
            mp_image: A mediapipe mp.Image in SRGB format typically created from an RGB numpy array.
            timestamp_ms: Optional epoch milliseconds timestamp to associate with the image.
                          If omitted, the current time is used.
        """
        if timestamp_ms is None:
            import time

            timestamp_ms = int(time.time() * 1000)
        try:
            recognizer.recognize_async(mp_image, int(timestamp_ms))
        except Exception:
            # Defensive: do not raise from this helper; callers may want to continue the loop.
            pass

    def recognize(
        self,
        recognizer: GestureRecognizer,
        mp_image: mp.Image,
        timestamp_ms: Optional[int] = None,
    ) -> Any:
        """
        Synchronous recognition helper (wraps recognizer.recognize).
        Returns the raw result from the recognizer or None on failure.
        """
        if timestamp_ms is None:
            import time

            timestamp_ms = int(time.time() * 1000)
        try:
            return recognizer.recognize(mp_image, int(timestamp_ms))
        except Exception:
            return None


# Lightweight factory helper --------------------------------------------------
def make_recognizer_runner(
    model_path: str, result_store: ResultStore, num_hands: int = 2
) -> RecognizerRunner:
    """
    Convenience factory for creating a RecognizerRunner.
    """
    return RecognizerRunner(
        model_path=model_path, result_store=result_store, num_hands=num_hands
    )
