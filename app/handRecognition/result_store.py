"""
Thread-safe ResultStore for mediapipe recognizer callbacks.

This module provides a small, well-documented container to hold the latest
recognition results coming from a background callback thread. It is intentionally
minimal: callers can `set` a new snapshot of (gestures, handedness, landmarks)
and other threads can obtain a consistent `snapshot()` copy.

The API is deliberately simple and avoids exposing internal locks or mutable
references to internal lists.
"""

import time
from threading import Lock
from typing import Any, List, Optional, Tuple


class ResultStore:
    """
    Thread-safe store for recognizer results.

    Attributes:
        _lock: Protects access to internal data.
        _gestures: List of optional gesture names per detected hand.
        _handedness: List of optional handedness names per detected hand.
        _landmarks: List of landmark objects (left as Any because mediapipe types are runtime-only).
        _last_update_ts: Unix timestamp (float) of last update.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._gestures: List[Optional[str]] = []
        self._handedness: List[Optional[str]] = []
        self._landmarks: List[Any] = []
        self._last_update_ts: float = 0.0

    def set(
        self,
        gestures: List[Optional[str]],
        handedness: List[Optional[str]],
        landmarks: List[Any],
    ) -> None:
        """
        Atomically replace the stored results with the provided values.

        This method copies the provided lists (shallow copy) to avoid retaining
        references to caller-owned mutable lists.

        Parameters:
            gestures: list of optional gesture names (e.g. ["03_fist", None])
            handedness: list of optional handedness names (e.g. ["Left", "Right"])
            landmarks: list of landmark sequences/objects from mediapipe
        """
        ts = time.time()
        with self._lock:
            # store copies so external mutations won't affect internal state
            self._gestures = list(gestures)
            self._handedness = list(handedness)
            self._landmarks = list(landmarks)
            self._last_update_ts = ts

    # Provide an alias for clearer intent in some call-sites
    set_results = set

    def snapshot(
        self,
    ) -> Tuple[List[Optional[str]], List[Optional[str]], List[Any], float]:
        """
        Return a tuple (gestures, handedness, landmarks, last_update_ts) where
        each list is a shallow copy of the internal state.

        The returned lists are independent of the store's internal lists and can
        safely be inspected or mutated by the caller.
        """
        with self._lock:
            return (
                list(self._gestures),
                list(self._handedness),
                list(self._landmarks),
                float(self._last_update_ts),
            )

    def get_gestures(self) -> Tuple[List[Optional[str]], float]:
        """Return (gestures_list, last_update_ts)."""
        with self._lock:
            return (list(self._gestures), float(self._last_update_ts))

    def get_handedness(self) -> Tuple[List[Optional[str]], float]:
        """Return (handedness_list, last_update_ts)."""
        with self._lock:
            return (list(self._handedness), float(self._last_update_ts))

    def get_landmarks(self) -> Tuple[List[Any], float]:
        """Return (landmarks_list, last_update_ts)."""
        with self._lock:
            return (list(self._landmarks), float(self._last_update_ts))

    def clear(self) -> None:
        """Clear stored results and reset the timestamp."""
        with self._lock:
            self._gestures = []
            self._handedness = []
            self._landmarks = []
            self._last_update_ts = 0.0

    @property
    def last_update_ts(self) -> float:
        """Return the timestamp (unix epoch float) of the last update (0.0 if none)."""
        with self._lock:
            return float(self._last_update_ts)


__all__ = ["ResultStore"]
