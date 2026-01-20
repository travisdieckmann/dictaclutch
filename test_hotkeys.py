#!/usr/bin/env python3
"""
Hotkey Testing Script
Tests how different key combinations are registered by pynput.
Press keys to see what the system detects.
Press Ctrl+C to exit.
"""

from pynput import keyboard
import sys

# Track currently pressed keys
current_keys = set()

def on_press(key):
    """Handle key press - show what's detected."""
    current_keys.add(key)

    print("\n" + "=" * 60)
    print(f"KEY PRESSED: {key}")
    print("-" * 60)

    # Show key details
    print("Key attributes:")
    if hasattr(key, 'char'):
        print(f"  .char = {repr(key.char)}")
    if hasattr(key, 'vk'):
        print(f"  .vk = {key.vk} (0x{key.vk:02X})")
    if hasattr(key, 'name'):
        print(f"  .name = {repr(key.name)}")
    print(f"  type = {type(key).__name__}")
    print(f"  repr = {repr(key)}")

    # Show all currently pressed keys
    print("\nCurrently pressed keys:")
    for k in current_keys:
        desc = []
        if hasattr(k, 'name'):
            desc.append(f"name={k.name}")
        if hasattr(k, 'char') and k.char:
            desc.append(f"char={repr(k.char)}")
        if hasattr(k, 'vk'):
            desc.append(f"vk={k.vk}")
        print(f"  - {k} ({', '.join(desc)})")

    # Check for specific combos
    print("\nCombo detection:")

    # Check Ctrl
    has_ctrl = any(
        k in [keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r]
        for k in current_keys
    )
    print(f"  Ctrl pressed: {has_ctrl}")

    # Check Shift
    has_shift = any(
        k in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]
        for k in current_keys
    )
    print(f"  Shift pressed: {has_shift}")

    # Check for J (VK code 74)
    has_j_by_vk = any(hasattr(k, 'vk') and k.vk == 74 for k in current_keys)
    has_j_by_char = any(hasattr(k, 'char') and k.char and k.char.lower() == 'j' for k in current_keys)
    print(f"  J pressed (by VK=74): {has_j_by_vk}")
    print(f"  J pressed (by char): {has_j_by_char}")

    # Check for K (VK code 75)
    has_k_by_vk = any(hasattr(k, 'vk') and k.vk == 75 for k in current_keys)
    has_k_by_char = any(hasattr(k, 'char') and k.char and k.char.lower() == 'k' for k in current_keys)
    print(f"  K pressed (by VK=75): {has_k_by_vk}")
    print(f"  K pressed (by char): {has_k_by_char}")

    # Combo results
    print("\nHotkey matches:")
    if has_ctrl and has_shift and (has_j_by_vk or has_j_by_char):
        print("  ✅ Ctrl+Shift+J detected!")
    if has_ctrl and has_shift and (has_k_by_vk or has_k_by_char):
        print("  ✅ Ctrl+Shift+K detected!")

    if has_ctrl and has_shift:
        # Show what letter key if any
        letter_keys = []
        for k in current_keys:
            if hasattr(k, 'vk') and 65 <= k.vk <= 90:  # A-Z
                letter = chr(k.vk)
                letter_keys.append(f"{letter} (vk={k.vk})")
            elif hasattr(k, 'char') and k.char and k.char.isalpha():
                letter_keys.append(f"{k.char} (char)")
        if letter_keys:
            print(f"  Letter keys with Ctrl+Shift: {', '.join(letter_keys)}")

def on_release(key):
    """Handle key release."""
    current_keys.discard(key)

    if key == keyboard.Key.esc:
        print("\nESC pressed - exiting...")
        return False  # Stop listener

def main():
    print("=" * 60)
    print("HOTKEY TESTING SCRIPT")
    print("=" * 60)
    print()
    print("Press any keys to see how they're registered.")
    print("Try pressing Ctrl+Shift+J and Ctrl+Shift+K separately.")
    print()
    print("Press ESC to exit (or Ctrl+C)")
    print()
    print("Expected VK codes:")
    print("  J = 74 (0x4A)")
    print("  K = 75 (0x4B)")
    print()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            print("\nCtrl+C - exiting...")

if __name__ == "__main__":
    main()
