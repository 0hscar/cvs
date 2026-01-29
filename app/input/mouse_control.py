"""
Helper utilities to move the OS mouse from normalized coordinates.

Behavior:
- Try to use a Python-level library if available.
- Fall back to common command-line utilities if the Python option is unavailable.
- Provide two public functions:
    - move_mouse_abs(x, y)         -> move to absolute screen coordinates
    - move_mouse_normalized(nx, ny [, fallback_screen_size=(w,h)])
                                      -> move to normalized coords (0..1)

The functions return True on success, False on failure. They do not raise on
routine failures so callers can attempt other handling paths if desired.
"""

import shutil
import subprocess
import sys
from typing import Optional, Tuple

# Try to import a Python-level mouse controller first (if available).
# This provides the most portable, high-level control.
try:
    import pyautogui  # type: ignore

    _HAS_PYAUTOGUI = True
except Exception:
    _HAS_PYAUTOGUI = False


# Whether to prefer the 'primary' monitor when mapping normalized coords.
PREFER_PRIMARY_MONITOR = True
# If you want to force mapping to a specific monitor, set its name here (e.g. "DP-1").
# If None, the code will prefer primary / (0,0) / largest monitor.
TARGET_MONITOR_NAME: Optional[str] = "DP-1"


def detect_monitors() -> List[Tuple[str, int, int, int, int, bool]]:
    """
    Public helper that parses `xrandr --query` and returns a list of monitors:

    [(name, width, height, offset_x, offset_y, is_primary), ...]

    Returns an empty list if detection fails or xrandr is not available.
    This is the public wrapper other modules should import and call.
    """
    xr = shutil.which("xrandr")
    if not xr:
        return []
    try:
        out = subprocess.check_output(
            [xr, "--query"], stderr=subprocess.DEVNULL, text=True
        )
    except Exception:
        return []
    monitors: List[Tuple[str, int, int, int, int, bool]] = []
    import re

    for line in out.splitlines():
        if " connected" in line:
            text = line.strip()
            parts = text.split()
            name = parts[0]
            rest = " ".join(parts[1:])
            m = re.search(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", rest)
            if m:
                w = int(m.group(1))
                h = int(m.group(2))
                ox = int(m.group(3))
                oy = int(m.group(4))
                is_primary = "primary" in rest
                monitors.append((name, w, h, ox, oy, is_primary))
    return monitors


def _get_monitor_geometry() -> Optional[Tuple[int, int, int, int]]:
    """
    Attempt to determine monitor geometry to map normalized coordinates to a
    specific monitor area. Returns (width, height, offset_x, offset_y) for the
    selected monitor (prefer primary when available). If detection fails, returns None.
    """
    # Try xrandr --current to find monitor lines and geometry patterns
    try:
        out = subprocess.check_output(
            ["xrandr", "--current"], stderr=subprocess.DEVNULL, text=True
        )
        lines = out.splitlines()
        monitors = []
        for line in lines:
            text = line.strip()
            # match lines like: HDMI-1 connected primary 1920x1080+0+0 ...
            if " connected " in text:
                parts = text.split()
                name = parts[0]
                rest = " ".join(parts[1:])
                # search for geometry pattern WxH+X+Y
                import re

                m = re.search(r"(\d+)x(\d+)\+(\-?\d+)\+(\-?\d+)", rest)
                if m:
                    w = int(m.group(1))
                    h = int(m.group(2))
                    ox = int(m.group(3))
                    oy = int(m.group(4))
                    is_primary = "primary" in rest
                    monitors.append((name, w, h, ox, oy, is_primary))
        if monitors:
            # If a TARGET_MONITOR_NAME is set, prefer that monitor explicitly.
            if TARGET_MONITOR_NAME:
                for mon in monitors:
                    if mon[0] == TARGET_MONITOR_NAME:
                        return (mon[1], mon[2], mon[3], mon[4])
            # Prefer primary monitor if configured and available.
            if PREFER_PRIMARY_MONITOR:
                for mon in monitors:
                    if mon[5]:
                        return (mon[1], mon[2], mon[3], mon[4])
            # Fallback: prefer the monitor at offset (0,0) (commonly the main display)
            for mon in monitors:
                if mon[3] == 0 and mon[4] == 0:
                    return (mon[1], mon[2], mon[3], mon[4])
            # As a last resort, choose the monitor with the largest area.
            monitors.sort(key=lambda m: m[1] * m[2], reverse=True)
            name, w, h, ox, oy, _ = monitors[0]
            return (w, h, ox, oy)
    except Exception:
        pass

    # Fallback to xdpyinfo for overall screen size (no offsets)
    try:
        out = subprocess.check_output(
            ["xdpyinfo"], stderr=subprocess.DEVNULL, text=True
        )
        for line in out.splitlines():
            if "dimensions:" in line:
                parts = line.strip().split()
                if len(parts) >= 2:
                    dims = parts[1]
                    if "x" in dims:
                        w_str, h_str = dims.split("x", 1)
                        if w_str.isdigit() and h_str.isdigit():
                            return (int(w_str), int(h_str), 0, 0)
    except Exception:
        pass

    # If nothing worked, return None
    return None


def move_mouse_abs(x: int, y: int) -> bool:
    """
    Move the mouse pointer to absolute screen coordinates (x, y).

    Tries, in order:
      1. The Python-level library (if available)
      2. A standard command-line utility A (if installed)
      3. A standard command-line utility B (if installed)

    Returns True if the move succeeded, False otherwise.
    """
    # Debug: print requested coordinates and available backends
    print(
        f"[mouse_control] move_mouse_abs called with ({x},{y}). backends pyautogui={_HAS_PYAUTOGUI}, xdotool={shutil.which('xdotool') is not None}, ydotool={shutil.which('ydotool') is not None}"
    )
    # 1) Python-level API
    if _HAS_PYAUTOGUI:
        try:
            pyautogui.moveTo(x, y)
            return True
        except Exception as e:
            # Fall through to command line fallbacks
            print(f"[mouse_control] python-level move failed: {e}", file=sys.stderr)

    # 2) First CLI fallback
    cli_a = shutil.which("xdotool")
    if cli_a:
        try:
            subprocess.check_call([cli_a, "mousemove", str(x), str(y)])
            return True
        except Exception as e:
            print(f"[mouse_control] cli-a move failed: {e}", file=sys.stderr)

    # 3) Second CLI fallback (ydotool)
    cli_b = shutil.which("ydotool")
    if cli_b:
        # Try to detect which mouse-related subcommand this ydotool binary supports
        # by parsing `ydotool help`. Prefer the single-word `mousemove` form if present
        # (this matches the 'Available commands' output you pasted).
        def _detect_ydotool_mouse_form(cli_path):
            try:
                out = subprocess.check_output(
                    [cli_path, "help"], stderr=subprocess.DEVNULL, text=True
                )
                txt = out.lower()
                # Prefer exact 'mousemove'
                if "mousemove" in txt:
                    return [cli_path, "mousemove", str(x), str(y)]
                # If help mentions 'mouse' and 'move' separately, assume two-word form
                if "mouse" in txt and "move" in txt:
                    return [cli_path, "mouse", "move", str(x), str(y)]
                # Fallback: try single-letter shorthand if help shows it
                if "\n m " in txt or "\nm " in txt:
                    return [cli_path, "m", str(x), str(y)]
            except Exception:
                # If help fails, we'll fall back to trying common forms below
                pass
            return None

        # If detection yields a specific form, try it first
        detected = _detect_ydotool_mouse_form(cli_b)
        if detected:
            try:
                subprocess.check_call(detected)
                return True
            except Exception as e:
                print(
                    f"[mouse_control] ydotool selected form {detected} failed: {e}",
                    file=sys.stderr,
                )

        # As a fallback, try common invocation forms in a preferred order that puts
        # the single-word `mousemove` first (your build indicates that's the supported form).
        tried_forms = [
            [cli_b, "mousemove", str(x), str(y)],
            [cli_b, "mouse", "move", str(x), str(y)],
            [cli_b, "move", str(x), str(y)],
            [cli_b, "m", str(x), str(y)],
        ]
        for form in tried_forms:
            try:
                subprocess.check_call(form)
                return True
            except Exception as e:
                print(
                    f"[mouse_control] ydotool invocation {form} failed: {e}",
                    file=sys.stderr,
                )

        # As a last attempt, try sending a simple command to ydotool via stdin
        # if the binary supports a stdin command interface (some builds do).
        try:
            p = subprocess.Popen(
                [cli_b],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            cmd = f"mousemove {x} {y}\n"
            out, err = p.communicate(cmd, timeout=1)
            if p.returncode == 0:
                return True
            else:
                print(
                    f"[mouse_control] ydotool stdin attempt returned code {p.returncode}. stdout={out!r} stderr={err!r}",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"[mouse_control] ydotool stdin attempt failed: {e}", file=sys.stderr)

    # None of the methods worked
    print("[mouse_control] No available method to move the mouse.", file=sys.stderr)
    return False


def move_mouse_normalized(
    nx: float, ny: float, fallback_screen_size: Optional[Tuple[int, int]] = None
) -> bool:
    """
    Move the mouse using normalized coordinates in [0.0, 1.0].

    Parameters:
      - nx, ny: normalized coordinates (0.0 = left/top, 1.0 = right/bottom).
                 Values slightly outside [0,1] will be clamped.
      - fallback_screen_size: optional (width, height) OR (w,h,ox,oy) to use if screen size
                              cannot be discovered automatically.

    Behavior:
      Maps normalized coords to a single monitor area (prefer primary monitor if available).
      The returned absolute coordinates are computed as: offset_x + nx * monitor_width
    Returns True on success, False on failure.
    """
    # Clamp normalized coords to [0,1]
    nx = max(0.0, min(1.0, float(nx)))
    ny = max(0.0, min(1.0, float(ny)))

    monitor_w = monitor_h = None
    offset_x = offset_y = 0

    # If pyautogui is available, we can get total screen size, but prefer per-monitor geometry
    total_w = total_h = None
    if _HAS_PYAUTOGUI:
        try:
            size = pyautogui.size()
            total_w, total_h = int(size.width), int(size.height)
        except Exception:
            total_w = total_h = None

    # If a fallback with explicit geometry (w,h,ox,oy) is provided, use it
    if fallback_screen_size:
        if isinstance(fallback_screen_size, tuple) and len(fallback_screen_size) == 4:
            monitor_w, monitor_h, offset_x, offset_y = fallback_screen_size
        elif isinstance(fallback_screen_size, tuple) and len(fallback_screen_size) == 2:
            monitor_w, monitor_h = fallback_screen_size

    # Prefer a Hyprland-specific mapping when Hyprland's control utility is present.
    # Hyprland exposes monitor positions in its own coordinate space (see `hyprctl -j monitors`).
    # If `hyprctl` is available, compute absolute coords using hyprctl monitor x/y offsets
    # and dispatch a movecursor via hyprctl (this works reliably on Hyprland Wayland).
    hyprctl_path = shutil.which("hyprctl")
    if hyprctl_path:
        try:
            # Query hyprland monitor information and pick the configured target monitor name
            out = subprocess.check_output(
                [hyprctl_path, "-j", "monitors"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1,
            )
            import json

            mons = json.loads(out)
            # mons is a list of monitor dicts. Try to pick TARGET_MONITOR_NAME first.
            chosen = None
            if isinstance(mons, list):
                for m in mons:
                    if TARGET_MONITOR_NAME and m.get("name") == TARGET_MONITOR_NAME:
                        chosen = m
                        break
                # fallback: pick focused monitor if available
                if chosen is None:
                    for m in mons:
                        if m.get("focused"):
                            chosen = m
                            break
                # fallback: pick first monitor
                if chosen is None and len(mons) > 0:
                    chosen = mons[0]
            if chosen:
                hy_x = int(chosen.get("x", 0))
                hy_y = int(chosen.get("y", 0))
                hy_w = int(chosen.get("width", chosen.get("width", 0)))
                hy_h = int(chosen.get("height", chosen.get("height", 0)))
                # Compute absolute coordinates in Hyprland space
                abs_x = hy_x + int(nx * max(0, (hy_w - 1)))
                abs_y = hy_y + int(ny * max(0, (hy_h - 1)))
                try:
                    # Use hyprctl dispatch movecursor which expects global Hyprland coords
                    subprocess.check_call(
                        [hyprctl_path, "dispatch", "movecursor", str(abs_x), str(abs_y)]
                    )
                    return True
                except Exception as e:
                    # If hyprctl dispatch failed, fall back to other methods below
                    print(
                        f"[mouse_control] hyprctl movecursor failed: {e}",
                        file=sys.stderr,
                    )
        except Exception as e:
            # Parsing or hyprctl failure — fall back to other detection methods
            print(f"[mouse_control] hyprctl detection failed: {e}", file=sys.stderr)

    # Try to detect a primary monitor geometry via xrandr (legacy fallback)
    geo = _get_monitor_geometry()
    if geo:
        monitor_w, monitor_h, offset_x, offset_y = geo

    # If still unknown but we have total screen size, map normalized coords to the whole area
    if monitor_w is None or monitor_h is None:
        if total_w and total_h:
            monitor_w, monitor_h = total_w, total_h
            offset_x, offset_y = 0, 0

    if monitor_w is None or monitor_h is None:
        print(
            "[mouse_control] Could not determine monitor geometry; skipping move.",
            file=sys.stderr,
        )
        return False

    # Compute absolute pixel coordinates mapped to the chosen monitor area
    abs_x = offset_x + int(nx * max(0, (monitor_w - 1)))
    abs_y = offset_y + int(ny * max(0, (monitor_h - 1)))

    # Clamp to non-negative coordinates to avoid strange multi-monitor behavior
    abs_x = max(0, abs_x)
    abs_y = max(0, abs_y)

    return move_mouse_abs(abs_x, abs_y)


if __name__ == "__main__":
    # Simple interactive test when executed directly.
    # Moves cursor to the center of the screen as a quick smoke test.
    ok = move_mouse_normalized(0.5, 0.5)
    if ok:
        print("Mouse moved to center.")
    else:
        print("Mouse move failed.", file=sys.stderr)
