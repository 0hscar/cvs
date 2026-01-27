import os
import time

_last_trigger_time = {}

GESTURE_COOLDOWNS = {
    "Open_Palm": 1.0,
    "Closed_Fist": 1.0,
    "Thumb_Up": 0,
    "Thumb_Down": 0,
    "Peace": 0,
}


def action_open_palm(handedness, handIndex):
    print(f"Action: Open Palm detected on Hand {handIndex} ({handedness}).")
    return False


def action_thumb_up(handedness, handIndex):
    print(f"Action: Thumbs Up detected on Hand {handIndex} ({handedness}).")
    print("Volume Up")
    os.system("pactl set-sink-volume @DEFAULT_SINK@ +5%")


def action_thumb_down(handedness, handIndex):
    print(f"Action: Thumbs Down detected on Hand {handIndex} ({handedness}).")
    print("Volume Down")
    os.system("pactl set-sink-volume @DEFAULT_SINK@ -5%")


def action_closed_fist(handedness, handIndex):
    global _last_trigger_time
    now = time.time()
    if now - _last_trigger_time.get("Closed_Fist", 0) > GESTURE_COOLDOWNS.get(
        "Closed_Fist", 1.0
    ):
        print(f"Action: Closed Fist detected on Hand {handIndex} ({handedness}).")
        print("Pause / Unpause media playback")
        os.system("playerctl play-pause")
        print("Done")
        _last_trigger_time["Closed_Fist"] = now


def action_victory(handedness, handIndex):
    print(f"Action: Victory Detected on Hand {handIndex} ({handedness}).")
    print("Turn off")
    return True


GESTURE_ACTIONS = {
    "Open_Palm": action_open_palm,
    "Closed_Fist": action_closed_fist,
    "Thumb_Up": action_thumb_up,
    "Thumb_Down": action_thumb_down,
    "Victory": action_victory,
}
