import os
import time

from gestureActions.gestures import GESTURE_ACTIONS, GESTURE_COOLDOWNS

last_media_action = 0
_last_trigger_time = {}


def handleGesture(gesture, handedness=None, handIndex=None):
    now = time.time()
    cooldown = GESTURE_COOLDOWNS.get(gesture, 0)
    last_time = _last_trigger_time.get(gesture, 0)

    if handedness and handIndex is not None:
        print(f"Hand {handIndex} ({handedness}): {gesture}")

        if gesture in GESTURE_ACTIONS and (now - last_time > cooldown):
            result = GESTURE_ACTIONS[gesture](handedness, handIndex)
            _last_trigger_time[gesture] = now
            return result
    else:
        print(f"Gesture: {gesture}")
    return False
