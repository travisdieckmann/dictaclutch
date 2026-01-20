"""
Linux-specific keyboard handling for DictaClutch.

Uses xdotool for keyboard simulation.
"""

import subprocess
import sys
import time

if sys.platform == "win32":
    raise ImportError("This module is only available on Linux")


def send_ctrl_v() -> None:
    """Send Ctrl+V using xdotool."""
    subprocess.run(["xdotool", "key", "ctrl+v"], check=False)


def type_text(text: str) -> None:
    """Type text using xdotool."""
    subprocess.run(["xdotool", "type", "--", text], check=False)


def send_backspace(count: int) -> None:
    """Send backspace key N times."""
    if count <= 0:
        return

    for _ in range(count):
        subprocess.run(["xdotool", "key", "BackSpace"], check=False)
        time.sleep(0.01)


def send_ctrl_left(count: int) -> None:
    """Send Ctrl+Left arrow N times to jump back by words."""
    if count <= 0:
        return

    for _ in range(count):
        subprocess.run(["xdotool", "key", "ctrl+Left"], check=False)
        time.sleep(0.005)


def send_ctrl_backspace(count: int) -> None:
    """Send Ctrl+Backspace N times to delete words."""
    if count <= 0:
        return

    for _ in range(count):
        subprocess.run(["xdotool", "key", "ctrl+BackSpace"], check=False)
        time.sleep(0.01)


def send_end_key() -> None:
    """Send End key to go to end of line."""
    subprocess.run(["xdotool", "key", "End"], check=False)


def get_foreground_window() -> int:
    """Get ID of currently focused window (not implemented, returns 0)."""
    # xdotool can get window ID but implementation varies
    # For now, return 0 (focus tracking not implemented)
    return 0
