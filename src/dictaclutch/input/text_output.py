"""
Platform-agnostic text output for DictaClutch.

Provides functions for pasting and typing text that work on both
Windows and Linux.
"""

import sys
import time

import pyperclip

from dictaclutch.config import DEFAULT_CONFIG


# Import platform-specific keyboard functions
if sys.platform == "win32":
    from dictaclutch.input.keyboard_win import (
        send_ctrl_v,
        type_char,
        send_backspace,
        send_ctrl_left,
        send_ctrl_backspace,
        send_end_key,
        get_foreground_window,
    )

    def _type_text_platform(text: str) -> None:
        """Type text character by character on Windows."""
        for char in text:
            type_char(char)
            time.sleep(0.01)
else:
    from dictaclutch.input.keyboard_linux import (
        send_ctrl_v,
        type_text as _type_text_linux,
        send_backspace,
        send_ctrl_left,
        send_ctrl_backspace,
        send_end_key,
        get_foreground_window,
    )

    def _type_text_platform(text: str) -> None:
        """Type text on Linux using xdotool."""
        _type_text_linux(text)


def paste_text(text: str) -> None:
    """
    Paste text into the active window using configured method.

    Args:
        text: Text to paste
    """
    if not text.strip():
        return

    if DEFAULT_CONFIG["paste_method"] == "clipboard":
        paste_via_clipboard(text)
    else:
        type_text(text)


def paste_via_clipboard(text: str) -> None:
    """
    Copy to clipboard and simulate Ctrl+V.

    Args:
        text: Text to paste
    """
    pyperclip.copy(text)
    time.sleep(0.05)  # Small delay to ensure clipboard is ready
    send_ctrl_v()


def type_text(text: str) -> None:
    """
    Type text character by character (cross-platform).

    Args:
        text: Text to type
    """
    _type_text_platform(text)


# Re-export platform functions for direct use
__all__ = [
    "paste_text",
    "paste_via_clipboard",
    "type_text",
    "send_backspace",
    "send_ctrl_left",
    "send_ctrl_backspace",
    "send_end_key",
    "get_foreground_window",
]
