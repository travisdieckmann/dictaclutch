#!/usr/bin/env python3
"""
Hotkey Testing Script
Tests how different key combinations are registered by pynput.
Logs all key events to hotkey_test.log to avoid terminal interference.
Press ESC to exit.
"""

from pynput import keyboard
from datetime import datetime
import sys

LOG_FILE = "hotkey_test.log"

# Track currently pressed keys by VK code (same approach as the fix)
pressed_vk_codes: set[int] = set()

# VK codes for modifiers
CTRL_VK_CODES = {162, 163}  # VK_LCONTROL, VK_RCONTROL
SHIFT_VK_CODES = {160, 161}  # VK_LSHIFT, VK_RSHIFT

def get_vk_code(key) -> int | None:
    """Extract VK code from a key."""
    if hasattr(key, "vk") and key.vk is not None:
        return key.vk
    if hasattr(key, "value") and hasattr(key.value, "vk"):
        return key.value.vk
    return None

def log(msg: str, also_print: bool = False):
    """Write to log file with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] {msg}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if also_print:
        print(line)

def on_press(key):
    """Handle key press - log what's detected."""
    vk = get_vk_code(key)
    if vk is not None:
        pressed_vk_codes.add(vk)

    log("=" * 60)
    log(f"KEY PRESSED: {key}")
    log("-" * 60)

    # Show key details
    log("Key attributes:")
    if hasattr(key, 'char'):
        log(f"  .char = {repr(key.char)}")
    if hasattr(key, 'vk'):
        log(f"  .vk = {key.vk} (0x{key.vk:02X})")
    if hasattr(key, 'name'):
        log(f"  .name = {repr(key.name)}")
    log(f"  type = {type(key).__name__}")
    log(f"  repr = {repr(key)}")

    # Show all currently pressed VK codes
    log(f"\nCurrently pressed VK codes: {sorted(pressed_vk_codes)}")

    # Check for specific combos using VK codes (same logic as voice_input.py)
    log("\nCombo detection (VK-based):")

    has_ctrl = bool(pressed_vk_codes & CTRL_VK_CODES)
    has_shift = bool(pressed_vk_codes & SHIFT_VK_CODES)
    has_j = 74 in pressed_vk_codes  # J = VK 74
    has_k = 75 in pressed_vk_codes  # K = VK 75

    log(f"  Ctrl pressed (VK 162/163): {has_ctrl}")
    log(f"  Shift pressed (VK 160/161): {has_shift}")
    log(f"  J pressed (VK 74): {has_j}")
    log(f"  K pressed (VK 75): {has_k}")

    # Hotkey detection
    log("\nHotkey matches:")
    if has_ctrl and has_shift and has_j:
        log("  >>> Ctrl+Shift+J DETECTED <<<", also_print=True)
    if has_ctrl and has_shift and has_k:
        log("  >>> Ctrl+Shift+K DETECTED <<<", also_print=True)

    if has_ctrl and has_shift:
        # Show what letter keys are pressed
        letter_vks = [vk for vk in pressed_vk_codes if 65 <= vk <= 90]
        if letter_vks:
            letters = [f"{chr(vk)} (VK={vk})" for vk in letter_vks]
            log(f"  Letter keys with Ctrl+Shift: {', '.join(letters)}")

def on_release(key):
    """Handle key release."""
    vk = get_vk_code(key)
    if vk is not None:
        pressed_vk_codes.discard(vk)
        log(f"KEY RELEASED: {key} (VK={vk})")
        log(f"  Remaining VK codes: {sorted(pressed_vk_codes)}")

    if key == keyboard.Key.esc:
        log("\nESC pressed - exiting...", also_print=True)
        return False  # Stop listener

def main():
    # Clear/create log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"Hotkey Test Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

    print("=" * 60)
    print("HOTKEY TESTING SCRIPT")
    print("=" * 60)
    print()
    print(f"Logging all key events to: {LOG_FILE}")
    print()
    print("Press Ctrl+Shift+J and Ctrl+Shift+K separately.")
    print("Hotkey detections will be printed here.")
    print("Full details are written to the log file.")
    print()
    print("Press ESC to exit")
    print()
    print("Expected VK codes:")
    print("  Ctrl_L = 162, Ctrl_R = 163")
    print("  Shift_L = 160, Shift_R = 161")
    print("  J = 74, K = 75")
    print()
    print("Waiting for key presses...")
    print()

    log("Script started - waiting for key presses")
    log(f"CTRL_VK_CODES = {CTRL_VK_CODES}")
    log(f"SHIFT_VK_CODES = {SHIFT_VK_CODES}")
    log("")

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            log("\nCtrl+C - exiting...", also_print=True)

    print(f"\nLog saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
