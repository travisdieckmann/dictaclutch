"""Hotkey handling and diagnostic modules."""

from dictaclutch.hotkeys.handler import MultiHotkeyHandler
from dictaclutch.hotkeys.diagnostic import run_diagnostic, HotkeyDiagnostic

__all__ = [
    "MultiHotkeyHandler",
    "run_diagnostic",
    "HotkeyDiagnostic",
]
