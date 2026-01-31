"""
Tracker: encapsulates fingertip -> OS mouse movement mapping with smoothing.

This module provides a simple, testable `Tracker` class that:
 - accepts a callable `move_mouse_normalized(nx, ny, fallback_screen_size=None)`
   which performs the actual OS-level mouse movement and returns True on success.
 - optionally accepts a `detect_monitors()` callable used to provide debug
   information or a monitor fallback box if callers don't provide one.
 - maintains per-hand exponential moving average smoothing state so cursor
   movement is stable.
 - applies a sensitivity multiplier and optional Y inversion.

Design goals:
 - Keep methods small and side-effect free where possible.
 - Be defensive: failure to move the mouse should not raise exceptions.
 - Allow callers to inject dependencies (for easier testing).
"""

from typing import Any, Callable, Dict, Optional, Tuple

# Type aliases
MoveMouseFn = Callable[..., bool]
DetectMonitorsFn = Callable[
    [], Any
]  # returns a list-like of monitor tuples in the original code


class Tracker:
    """
    Tracker encapsulates mouse-tracking logic and smoothing.

    Args:
        move_mouse_normalized: callable with signature
            (nx: float, ny: float, fallback_screen_size: Optional[Tuple[int,int] or Tuple[int,int,int,int]]) -> bool
            If None, tracking will be a no-op and `track_hand` will return False.
        detect_monitors: optional callable that returns monitor info (used for debug/fallback).
        smooth_alpha: EMA alpha in (0,1] controlling smoothing. Larger alpha -> more immediate follow.
        sensitivity: multiplier applied to (nx-0.5) to amplify movements around center.
        invert_y: if True, flip the normalized Y coordinate before mapping to screen.
        track_gesture_name: gesture name which must be active to enable tracking when using gesture gating.
    """

    def __init__(
        self,
        move_mouse_normalized: Optional[MoveMouseFn],
        detect_monitors: Optional[DetectMonitorsFn] = None,
        smooth_alpha: float = 0.6,
        sensitivity: float = 1.2,
        invert_y: bool = False,
        track_gesture_name: str = "06_index",
    ) -> None:
        self.move_mouse_normalized = move_mouse_normalized
        self.detect_monitors = detect_monitors
        self.smooth_alpha = float(smooth_alpha)
        self.sensitivity = float(sensitivity)
        self.invert_y = bool(invert_y)
        self.track_gesture_name = track_gesture_name

        # per-hand last smoothed normalized coords: hand_index -> (sx, sy)
        self._last_smoothed: Dict[int, Tuple[float, float]] = {}

    # Public API ------------------------------------------------------------

    def set_smoothing(self, alpha: float) -> None:
        """Update smoothing alpha (0 < alpha <= 1)."""
        self.smooth_alpha = float(alpha)

    def set_sensitivity(self, sensitivity: float) -> None:
        """Update sensitivity multiplier."""
        self.sensitivity = float(sensitivity)

    def reset_hand(self, hand_index: int) -> None:
        """Clear smoothing state for a specific hand index."""
        self._last_smoothed.pop(hand_index, None)

    def reset_all(self) -> None:
        """Clear smoothing state for all hands."""
        self._last_smoothed.clear()

    def track_hand(
        self,
        hand_index: int,
        hand_landmarks: Any,
        gesture_name: Optional[str] = None,
        monitor_box: Optional[Tuple[int, int, int, int]] = None,
        require_gesture: bool = True,
    ) -> bool:
        """
        Attempt to move the mouse based on the hand's index-fingertip landmark.

        Parameters:
            hand_index: index of the detected hand (0..N-1).
            hand_landmarks: sequence-like object containing landmarks with .x and .y attributes.
                            The index fingertip is expected at index 8 (MediaPipe convention).
            gesture_name: optional current gesture name for this hand. If provided and
                          require_gesture=True, movement only occurs when it equals the
                          configured `track_gesture_name`.
            monitor_box: optional explicit monitor geometry tuple (w, h) or (w, h, ox, oy).
                         If provided it will be forwarded to the move function as fallback.
            require_gesture: if True and gesture_name is provided, only track when gesture_name==track_gesture_name.

        Returns:
            True if a move was attempted and the underlying move function reported success.
            False otherwise (including when movement is disabled or failed).
        """
        # Guard: must have a move backend
        if not callable(self.move_mouse_normalized):
            # tracking not available
            return False

        # If a gesture name was provided and we require it, gate tracking on it.
        if require_gesture and gesture_name is not None:
            if gesture_name != self.track_gesture_name:
                return False

        # Ensure landmarks contain the expected fingertip
        try:
            if not hand_landmarks or len(hand_landmarks) <= 8:
                return False
            lm = hand_landmarks[8]
            nx = float(lm.x)
            ny = float(lm.y)
        except Exception:
            # malformed landmarks
            return False

        # Optional Y inversion
        if self.invert_y:
            ny = 1.0 - ny

        # Apply sensitivity: scale movement around center 0.5
        nx = 0.5 + (nx - 0.5) * self.sensitivity
        ny = 0.5 + (ny - 0.5) * self.sensitivity

        # Clamp to [0,1]
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))

        # Smooth coordinates using exponential moving average per-hand
        prev = self._last_smoothed.get(hand_index, (nx, ny))
        alpha = float(self.smooth_alpha)
        sx = prev[0] * (1.0 - alpha) + nx * alpha
        sy = prev[1] * (1.0 - alpha) + ny * alpha
        self._last_smoothed[hand_index] = (sx, sy)

        # Debug: provide a best-effort computed absolute coords when monitor_box provided
        try:
            if monitor_box and len(monitor_box) in (2, 4):
                if len(monitor_box) == 4:
                    w_mon, h_mon, ox, oy = monitor_box
                else:
                    w_mon, h_mon = monitor_box
                    ox = oy = 0
                abs_x = int(ox + sx * max(0, (int(w_mon) - 1)))
                abs_y = int(oy + sy * max(0, (int(h_mon) - 1)))
                # Best-effort console debug; avoid raising on any printing issues.
                try:
                    print(
                        f"[Tracker] hand {hand_index} normalized (sx,sy)=({sx:.4f},{sy:.4f}) -> abs=({abs_x},{abs_y}) monitor={w_mon}x{h_mon}+{ox}+{oy}"
                    )
                except Exception:
                    pass

        except Exception:
            # ignore any monitor_box parsing errors; we still attempt to move with normalized coords
            pass

        # If monitor_box not supplied and detect_monitors is available, try to pick a fallback monitor_box
        fallback = None
        if monitor_box:
            fallback = monitor_box
        else:
            try:
                if callable(self.detect_monitors):
                    mons = self.detect_monitors()
                    if mons:
                        # choose DP-1 if present else primary else first
                        chosen = None
                        for m in mons:
                            if m and len(m) >= 6 and m[0] == "DP-1":
                                chosen = m
                                break
                        if chosen is None:
                            for m in mons:
                                if m and len(m) >= 6 and m[5]:
                                    chosen = m
                                    break
                        if chosen is None and mons:
                            chosen = mons[0]
                        if chosen and len(chosen) >= 4:
                            # expect (name, w, h, ox, oy, is_primary) in original helper
                            # Map to (w,h,ox,oy)
                            try:
                                name = chosen[0]
                                w_mon = int(chosen[1])
                                h_mon = int(chosen[2])
                                ox = int(chosen[3])
                                oy = int(chosen[4]) if len(chosen) > 4 else 0
                                fallback = (w_mon, h_mon, ox, oy)
                            except Exception:
                                fallback = None
            except Exception:
                fallback = None

        # Finally call the injected move function. Return its boolean result.
        try:
            if fallback:
                moved = bool(
                    self.move_mouse_normalized(sx, sy, fallback_screen_size=fallback)
                )
            else:
                moved = bool(self.move_mouse_normalized(sx, sy))
        except Exception:
            moved = False

        if not moved:
            # One-time diagnostic if movement failed; avoid spamming upstream loops.
            try:
                print("[Tracker] move_mouse_normalized failed")
            except Exception:
                pass

        return moved

    # Helper for debugging / unit tests -----------------------------------

    def last_smoothed_for_hand(self, hand_index: int) -> Optional[Tuple[float, float]]:
        """Return the last smoothed (sx, sy) for a given hand index, or None if unknown."""
        return self._last_smoothed.get(hand_index)
