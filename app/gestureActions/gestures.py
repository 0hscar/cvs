import os
import shutil
import subprocess
import time

# Try to use a Python-level mouse API if available (more portable when possible)
try:
    import pyautogui  # type: ignore

    _HAS_PYAUTOGUI = True
except Exception:
    _HAS_PYAUTOGUI = False

_last_trigger_time = {}

# State for an ongoing held left-button
_hold_state = {
    "holding": False,
    # optional: which hand index started the hold (for future use)
    "hand_index": None,
    "handedness": None,
}


def _run_cmd(cmd: list) -> bool:
    """
    Helper to run a subprocess command, returning True on success.
    """
    try:
        subprocess.check_call(cmd)
        return True
    except Exception:
        return False


def start_hold(handedness=None, handIndex=None):
    """
    Begin holding the left mouse button (mouse down). No-op if already holding.
    Tries backends in this order:
      1. pyautogui.mouseDown
      2. xdotool 'mousedown 1'
      3. ydotool 'click 0x40' (press)
    Returns True if the press was emitted, False if not.
    """
    global _hold_state
    if _hold_state["holding"]:
        return True

    # 1) Python-level
    if _HAS_PYAUTOGUI:
        try:
            pyautogui.mouseDown(button="left")
            _hold_state.update(
                {"holding": True, "hand_index": handIndex, "handedness": handedness}
            )
            return True
        except Exception:
            pass

    # 2) xdotool
    xdotool = shutil.which("xdotool")
    if xdotool:
        if _run_cmd([xdotool, "mousedown", "1"]):
            _hold_state.update(
                {"holding": True, "hand_index": handIndex, "handedness": handedness}
            )
            return True

    # 3) ydotool (some builds accept 'click 0x40' as press)
    ydotool = shutil.which("ydotool")
    if ydotool:
        # Try common forms that represent a button press on ydotool builds.
        # Historically some users used 'ydotool click 0x40' for press.
        if _run_cmd([ydotool, "click", "0x40"]):
            _hold_state.update(
                {"holding": True, "hand_index": handIndex, "handedness": handedness}
            )
            return True

    # If none worked, report failure (no change to state)
    return False


def stop_hold():
    """
    Release the held left mouse button (mouse up). No-op if not holding.
    Tries backends in this order:
      1. pyautogui.mouseUp
      2. xdotool 'mouseup 1'
      3. ydotool 'click 0x80' (release)
    Returns True if the release was emitted, False if not.
    """
    global _hold_state
    if not _hold_state["holding"]:
        return True

    # 1) Python-level
    if _HAS_PYAUTOGUI:
        try:
            pyautogui.mouseUp(button="left")
            _hold_state.update(
                {"holding": False, "hand_index": None, "handedness": None}
            )
            return True
        except Exception:
            pass

    # 2) xdotool
    xdotool = shutil.which("xdotool")
    if xdotool:
        if _run_cmd([xdotool, "mouseup", "1"]):
            _hold_state.update(
                {"holding": False, "hand_index": None, "handedness": None}
            )
            return True

    # 3) ydotool
    ydotool = shutil.which("ydotool")
    if ydotool:
        # Try release code commonly used by ydotool users
        if _run_cmd([ydotool, "click", "0x80"]):
            _hold_state.update(
                {"holding": False, "hand_index": None, "handedness": None}
            )
            return True

    # If none worked, leave state unchanged (report False)
    return False


GESTURE_COOLDOWNS = {
    "Open_Palm": 1.0,
    "Closed_Fist": 1.0,
    "Thumb_Up": 0,
    "Thumb_Down": 0,
    "Peace": 0,
    "02_l": 0,
    # allow more frequent checks for the fist so we can start hold quickly
    "03_fist": 0.1,
}


# old model start
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
        # pyautogui.press("down")
        # os.system("ydotool key 108:1 108:0") # Works, ydotool.
        print("Done")
        _last_trigger_time["Closed_Fist"] = now


def action_victory(handedness, handIndex):
    print(f"Action: Victory Detected on Hand {handIndex} ({handedness}).")
    print("Turn off")
    return True


# Old model end

# New model gestures start


def action_fist(handedness, handIndex):
    """
    Start a held left-click when this action is invoked. The actual release should
    be triggered when the gesture stops (see note below). This function itself
    is non-blocking and only issues a mouse-down once when the hold is not active.
    """
    print(f"Action: Fist Gesture detected on Hand {handIndex} ({handedness}).")
    started = start_hold(handedness=handedness, handIndex=handIndex)
    if started:
        print("Started hold (mouse down)")
    else:
        print("Failed to start hold (no backend available)")

    # Update trigger time to rate-limit how often we attempt to (re)start the hold.
    global _last_trigger_time
    _last_trigger_time["03_fist"] = time.time()
    return False


def action_ok(handedness, handIndex):
    print(f"Action: OK Gesture detected on Hand {handIndex} ({handedness}).")

    # print("Turn off")
    # return True


def action_l(handedness, handIndex):
    print(f"Action: L Gesture detected on Hand {handIndex} ({handedness}).")
    # print("Turn off")
    # return True


GESTURE_ACTIONS = {
    "Open_Palm": action_open_palm,
    "Closed_Fist": action_closed_fist,
    "Thumb_Up": action_thumb_up,
    "Thumb_Down": action_thumb_down,
    "Victory": action_victory,
    "02_l": action_l,
    "03_fist": action_fist,
    "OK": action_ok,
}
