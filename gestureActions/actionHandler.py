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


# def handleGesture(gesture, handedness=None, handIndex=None):
#     """
#     Handle the recognized gesture.

#     Args:
#         gesture (str): The name of the recognized gesture.
#         handedness (str, optional): The handedness of the hand ('Left' or 'Right').
#         handIndex (int, optional): The index of the hand if multiple hands are detected.
#     """
#     global last_media_action
#     cooldown = 1.0  # Cooldown period in seconds
#     if handedness and handIndex is not None:
#         print(f"Hand {handIndex} ({handedness}): {gesture}")
#         match gesture:
#             case "Open_Palm":
#                 print("Action: Open Palm detected.")
#             case "Closed_Fist":
#                 print("Action: Closed Fist detected.")
#                 now = time.time()
#                 if now - last_media_action > cooldown:
#                     print("Action: Thumbs Up detected.")
#                     print("Pause / Unpause media playback")
#                     # os.system("playerctl play-pause")
#                     os.system("playerctl next")

#                     print("Done")
#                     last_media_action = now

#             case "Thumb_Down":
#                 print("Action: Thumbs Down detected.")
#                 return True

#     else:
#         print(f"Gesture: {gesture}")
