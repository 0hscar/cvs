"""
Hold manager for detecting gesture transitions that should start/stop a held mouse action.

This module provides a small, well-tested-friendly `HoldManager` class that
encapsulates the logic of detecting transitions into and out of a configured
hold gesture (default: "03_fist") and invoking provided `start_hold` / `stop_hold`
callables.

Behavior:
 - Call `update(gestures, handedness=None, landmarks=None)` frequently (e.g. each frame).
 - For each hand index it will detect transitions:
     not-hold -> hold  : attempts to call start_hold(handIndex=hi, handedness=handedness)
     hold -> not-hold  : attempts to call stop_hold()
 - Keeps per-hand previous gesture state so callers do not need to manage it.
 - Throttles repeated start attempts using `restart_cooldown` per-hand.
 - Tracks an internal `holding` map to allow `release_all()` to try to clear any ongoing holds.

The module is defensive: missing backends or exceptions in start/stop calls are
caught and logged to stdout, and they won't raise to the caller.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

# Try to import default start/stop hooks from the gestures module if available.
try:
    from gestureActions.gestures import start_hold as _default_start_hold
    from gestureActions.gestures import stop_hold as _default_stop_hold
except Exception:
    try:
        from app.gestureActions.gestures import start_hold as _default_start_hold
        from app.gestureActions.gestures import stop_hold as _default_stop_hold
    except Exception:
        _default_start_hold = None  # type: ignore
        _default_stop_hold = None  # type: ignore


class HoldManager:
    """
    Encapsulate detection of hold gesture transitions and invocation of start/stop hooks.

    Args:
        start_hold: optional callable to invoke when a hold starts.
                    Expected signature: start_hold(handedness=None, handIndex=None) but any callable is accepted.
        stop_hold: optional callable to invoke when a hold ends.
                   Expected signature: stop_hold() but any callable is accepted.
        hold_gesture_name: gesture name that represents the 'hold' (defaults to '03_fist').
        restart_cooldown: minimum seconds between repeated start attempts per-hand to avoid spamming.
    """

    def __init__(
        self,
        start_hold: Optional[Callable[..., bool]] = None,
        stop_hold: Optional[Callable[..., bool]] = None,
        hold_gesture_name: str = "03_fist",
        restart_cooldown: float = 0.1,
    ) -> None:
        self.start_hold = start_hold if start_hold is not None else _default_start_hold
        self.stop_hold = stop_hold if stop_hold is not None else _default_stop_hold
        self.hold_gesture_name = hold_gesture_name
        self.restart_cooldown = float(restart_cooldown)

        # Per-hand previous gesture name (or None)
        self._prev_gestures: Dict[int, Optional[str]] = {}
        # Per-hand last start timestamp to rate-limit attempts
        self._last_start_ts: Dict[int, float] = {}
        # Per-hand holding state according to what this manager invoked successfully
        self._holding: Dict[int, bool] = {}

    def update(
        self,
        gestures: Optional[Sequence[Optional[str]]],
        handedness: Optional[Sequence[Optional[str]]] = None,
        landmarks: Optional[Sequence[Any]] = None,
    ) -> None:
        """
        Process the latest gestures and perform start/stop transitions.

        Parameters:
            gestures: sequence of gesture name strings (or None) indexed by hand.
            handedness: optional sequence of handedness strings aligned with gestures.
                        If provided, it will be forwarded to start_hold when available.
            landmarks: optional sequence of landmarks; used only to determine how many hands
                       to iterate when gestures is shorter/longer (keeps parity with original logic).

        This function does not return a value; it invokes the configured hooks as side effects.
        """
        # Normalize inputs
        gestures = list(gestures) if gestures is not None else []
        handedness = list(handedness) if handedness is not None else []
        landmarks = list(landmarks) if landmarks is not None else []

        max_hands = max(len(gestures), len(landmarks))
        # If there are zero detected hands but prev_gestures has entries, we still want to iterate
        # over the previously seen indices to potentially stop holds.
        if max_hands == 0 and self._prev_gestures:
            # consider indices present in prev state
            indices = sorted(self._prev_gestures.keys())
        else:
            indices = list(range(max_hands))

        now = time.time()
        for hi in indices:
            prev = self._prev_gestures.get(hi)
            curr = gestures[hi] if hi < len(gestures) else None
            hand_name = handedness[hi] if hi < len(handedness) else None

            # Entering hold
            if prev != self.hold_gesture_name and curr == self.hold_gesture_name:
                # Rate-limit attempts per-hand
                last = self._last_start_ts.get(hi, 0.0)
                if now - last >= self.restart_cooldown:
                    if callable(self.start_hold):
                        try:
                            # many start_hold implementations accept (handedness, handIndex) or kwargs.
                            # Try calling with keyword args first for clarity, fall back to positional.
                            try:
                                result = self.start_hold(
                                    handedness=hand_name, handIndex=hi
                                )
                            except TypeError:
                                # fallback: positional (handedness, handIndex) then (handIndex,)
                                try:
                                    result = self.start_hold(hand_name, hi)
                                except Exception:
                                    try:
                                        result = self.start_hold(hi)
                                    except Exception:
                                        result = None
                            # If the backend reports success truthily, mark holding
                            if result:
                                self._holding[hi] = True
                            else:
                                # Some backends return None even on success; optimistically set True
                                # only if we didn't have a holding state already.
                                self._holding.setdefault(hi, True)
                            print(
                                f"[HoldManager] start_hold invoked for hand {hi} (gesture={curr})"
                            )
                        except Exception as e:
                            print(f"[HoldManager] start_hold error for hand {hi}: {e}")
                    else:
                        print(
                            f"[HoldManager] start_hold backend not available for hand {hi}"
                        )
                    self._last_start_ts[hi] = now
                else:
                    # Too soon since last attempt; skip
                    pass

            # Exiting hold
            if prev == self.hold_gesture_name and curr != self.hold_gesture_name:
                if callable(self.stop_hold):
                    try:
                        res = self.stop_hold()
                        # Regardless of result, mark as not holding
                        self._holding[hi] = False
                        print(
                            f"[HoldManager] stop_hold invoked for hand {hi} (gesture -> {curr})"
                        )
                    except Exception as e:
                        print(f"[HoldManager] stop_hold error for hand {hi}: {e}")
                else:
                    print(
                        f"[HoldManager] stop_hold backend not available for hand {hi}"
                    )
                # clear last start timestamp so re-start attempts can happen immediately later if needed
                self._last_start_ts.pop(hi, None)

            # Update prev tracking
            if curr is not None:
                self._prev_gestures[hi] = curr
            else:
                # No gesture detected for this hand in the frame -> clear prev to avoid stale transitions
                self._prev_gestures.pop(hi, None)

    def is_holding(self, hand_index: int) -> bool:
        """Return whether this manager believes a given hand index currently holds the button."""
        return bool(self._holding.get(hand_index, False))

    def any_holding(self) -> bool:
        """Return True if any hand is currently marked as holding."""
        return any(self._holding.values())

    def release_all(self) -> None:
        """
        Attempt to release any holds this manager believes it started.

        Calls stop_hold() if available and clears internal holding state.
        """
        if not callable(self.stop_hold):
            print("[HoldManager] stop_hold backend not available; cannot release holds")
            # still clear internal state to avoid dangling assumptions
            self._holding.clear()
            return

        # Attempt to call stop once; the underlying implementations are expected to be idempotent.
        try:
            self.stop_hold()
            print("[HoldManager] stop_hold invoked to release all holds")
        except Exception as e:
            print(f"[HoldManager] stop_hold error when releasing all holds: {e}")
        finally:
            self._holding.clear()
            self._last_start_ts.clear()
            self._prev_gestures.clear()


__all__ = ["HoldManager"]
