"""
Windows-specific keyboard handling for DictaClutch.

Uses ctypes to interface with Windows user32.dll for keyboard simulation.
"""

import sys
import time
import ctypes
from ctypes import wintypes

if sys.platform != "win32":
    raise ImportError("This module is only available on Windows")


def send_ctrl_v() -> None:
    """Send Ctrl+V using Windows API."""
    VK_CONTROL = 0x11
    VK_V = 0x56
    KEYEVENTF_KEYUP = 0x0002

    user32 = ctypes.windll.user32

    # Press Ctrl
    user32.keybd_event(VK_CONTROL, 0, 0, 0)
    # Press V
    user32.keybd_event(VK_V, 0, 0, 0)
    # Release V
    user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
    # Release Ctrl
    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


def type_char(char: str) -> None:
    """Type a single character on Windows using Unicode input."""
    user32 = ctypes.windll.user32

    # Use Unicode input
    INPUT_KEYBOARD = 1
    KEYEVENTF_UNICODE = 0x0004
    KEYEVENTF_KEYUP = 0x0002

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class INPUT(ctypes.Structure):
        _fields_ = [
            ("type", wintypes.DWORD),
            ("ki", KEYBDINPUT),
            ("padding", ctypes.c_ubyte * 8),
        ]

    # Key down
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.ki.wVk = 0
    inp.ki.wScan = ord(char)
    inp.ki.dwFlags = KEYEVENTF_UNICODE
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    # Key up
    inp.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def send_backspace(count: int) -> None:
    """Send backspace key N times to delete characters."""
    if count <= 0:
        return

    VK_BACK = 0x08
    KEYEVENTF_KEYUP = 0x0002
    user32 = ctypes.windll.user32

    for _ in range(count):
        user32.keybd_event(VK_BACK, 0, 0, 0)
        user32.keybd_event(VK_BACK, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.01)


def send_key(vk_code: int, count: int = 1) -> None:
    """Send a virtual key N times."""
    if count <= 0:
        return

    KEYEVENTF_KEYUP = 0x0002
    user32 = ctypes.windll.user32

    for _ in range(count):
        user32.keybd_event(vk_code, 0, 0, 0)
        user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.005)


def send_left_arrow(count: int) -> None:
    """Send left arrow key N times."""
    VK_LEFT = 0x25
    send_key(VK_LEFT, count)


def send_ctrl_left(count: int) -> None:
    """Send Ctrl+Left arrow N times to jump back by words."""
    if count <= 0:
        return

    VK_LEFT = 0x25
    VK_CONTROL = 0x11
    KEYEVENTF_KEYUP = 0x0002
    user32 = ctypes.windll.user32

    for _ in range(count):
        user32.keybd_event(VK_CONTROL, 0, 0, 0)
        user32.keybd_event(VK_LEFT, 0, 0, 0)
        user32.keybd_event(VK_LEFT, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.005)


def send_ctrl_backspace(count: int) -> None:
    """Send Ctrl+Backspace N times to delete words."""
    if count <= 0:
        return

    VK_BACK = 0x08
    VK_CONTROL = 0x11
    KEYEVENTF_KEYUP = 0x0002
    user32 = ctypes.windll.user32

    for _ in range(count):
        user32.keybd_event(VK_CONTROL, 0, 0, 0)
        user32.keybd_event(VK_BACK, 0, 0, 0)
        user32.keybd_event(VK_BACK, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.01)


def send_end_key() -> None:
    """Send End key to go to end of line."""
    VK_END = 0x23
    send_key(VK_END, 1)


def get_foreground_window() -> int:
    """Get handle of currently focused window."""
    return ctypes.windll.user32.GetForegroundWindow()
