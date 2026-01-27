import os
import time

last_media_action = 0


def handleGesture(gesture, handedness=None, handIndex=None):
    """
    Handle the recognized gesture.

    Args:
        gesture (str): The name of the recognized gesture.
        handedness (str, optional): The handedness of the hand ('Left' or 'Right').
        handIndex (int, optional): The index of the hand if multiple hands are detected.
    """
    global last_media_action
    cooldown = 1.0  # Cooldown period in seconds
    if handedness and handIndex is not None:
        print(f"Hand {handIndex} ({handedness}): {gesture}")
        match gesture:
            case "Open_Palm":
                print("Action: Open Palm detected.")
            case "Closed_Fist":
                print("Action: Closed Fist detected.")
                now = time.time()
                if now - last_media_action > cooldown:
                    print("Action: Thumbs Up detected.")
                    print("Pause / Unpause media playback")
                    os.system("playerctl play-pause")
                    print("Done")
                    last_media_action = now
            case "Thumb_Down":
                print("Action: Thumbs Down detected.")
                return True

    else:
        print(f"Gesture: {gesture}")
