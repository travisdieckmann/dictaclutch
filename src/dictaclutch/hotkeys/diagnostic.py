"""
Hotkey diagnostic tool for DictaClutch.

Allows users to test and validate their hotkey configurations by
displaying key events and detecting hotkey combinations.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from pynput import keyboard

from dictaclutch.config import DEFAULT_CONFIG


LOG_FILE = "hotkey_diagnostic.log"


class HotkeyDiagnostic:
    """
    Diagnostic tool for testing hotkey detection.

    Logs all key events to a file and prints detected hotkey combinations
    to the console.
    """

    # VK codes for modifier keys
    CTRL_VK_CODES = {162, 163}  # VK_LCONTROL, VK_RCONTROL
    SHIFT_VK_CODES = {160, 161}  # VK_LSHIFT, VK_RSHIFT
    ALT_VK_CODES = {164, 165}  # VK_LMENU, VK_RMENU (Alt keys)

    def __init__(
        self,
        hotkeys_to_test: dict[str, set] | None = None,
        log_file: str | Path = LOG_FILE,
        verbose: bool = False,
    ):
        """
        Initialize the diagnostic tool.

        Args:
            hotkeys_to_test: Dict mapping hotkey names to key sets.
                Defaults to the configured batch and streaming hotkeys.
            log_file: Path to log file for key events
            verbose: If True, print all key events to console
        """
        self.hotkeys = hotkeys_to_test or {
            "batch": DEFAULT_CONFIG["hotkey_batch"],
            "streaming": DEFAULT_CONFIG["hotkey_streaming"],
        }
        self.log_file = Path(log_file)
        self.verbose = verbose
        self.pressed_vk_codes: set[int] = set()
        self.listener: keyboard.Listener | None = None

    def _get_vk_code(self, key: Any) -> int | None:
        """Extract VK code from a key."""
        if hasattr(key, "vk") and key.vk is not None:
            return key.vk
        if hasattr(key, "value") and hasattr(key.value, "vk"):
            return key.value.vk
        return None

    def _get_letter_vk(self, hotkey_set: set) -> int | None:
        """Extract the VK code for the letter key from a hotkey configuration."""
        for k in hotkey_set:
            if hasattr(k, "char") and k.char:
                return ord(k.char.upper())
        return None

    def _log(self, msg: str, also_print: bool = False) -> None:
        """Write to log file with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{timestamp}] {msg}"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if also_print or self.verbose:
            print(line)

    def _requires_modifier(self, hotkey_set: set, modifier_key: keyboard.Key) -> bool:
        """Check if a hotkey config requires a specific modifier key."""
        for k in hotkey_set:
            if k == modifier_key:
                return True
            if hasattr(k, "name"):
                if modifier_key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                    if k.name in ("ctrl", "ctrl_l", "ctrl_r"):
                        return True
                if modifier_key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                    if k.name in ("shift", "shift_l", "shift_r"):
                        return True
                if modifier_key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
                    if k.name in ("alt", "alt_l", "alt_r", "alt_gr"):
                        return True
        return False

    def _check_hotkeys(self) -> None:
        """Check if any configured hotkeys are currently pressed."""
        has_ctrl = bool(self.pressed_vk_codes & self.CTRL_VK_CODES)
        has_shift = bool(self.pressed_vk_codes & self.SHIFT_VK_CODES)
        has_alt = bool(self.pressed_vk_codes & self.ALT_VK_CODES)

        for name, hotkey_set in self.hotkeys.items():
            # Dynamically check which modifiers are required
            needs_ctrl = self._requires_modifier(hotkey_set, keyboard.Key.ctrl)
            needs_shift = self._requires_modifier(hotkey_set, keyboard.Key.shift)
            needs_alt = self._requires_modifier(hotkey_set, keyboard.Key.alt)

            # Check if required modifiers are pressed
            if needs_ctrl and not has_ctrl:
                continue
            if needs_shift and not has_shift:
                continue
            if needs_alt and not has_alt:
                continue

            letter_vk = self._get_letter_vk(hotkey_set)
            if letter_vk and letter_vk in self.pressed_vk_codes:
                self._log(f">>> {name.upper()} HOTKEY DETECTED <<<", also_print=True)

    def _on_press(self, key: Any) -> None:
        """Handle key press - log and detect hotkeys."""
        vk = self._get_vk_code(key)
        if vk is not None:
            self.pressed_vk_codes.add(vk)

        self._log("=" * 60)
        self._log(f"KEY PRESSED: {key}")
        self._log("-" * 60)

        # Show key details
        self._log("Key attributes:")
        if hasattr(key, 'char'):
            self._log(f"  .char = {repr(key.char)}")
        if hasattr(key, 'vk'):
            self._log(f"  .vk = {key.vk} (0x{key.vk:02X})")
        if hasattr(key, 'name'):
            self._log(f"  .name = {repr(key.name)}")
        self._log(f"  type = {type(key).__name__}")
        self._log(f"  repr = {repr(key)}")

        # Show all currently pressed VK codes
        self._log(f"\nCurrently pressed VK codes: {sorted(self.pressed_vk_codes)}")

        # Check for hotkeys
        self._log("\nHotkey detection (VK-based):")
        has_ctrl = bool(self.pressed_vk_codes & self.CTRL_VK_CODES)
        has_shift = bool(self.pressed_vk_codes & self.SHIFT_VK_CODES)
        has_alt = bool(self.pressed_vk_codes & self.ALT_VK_CODES)
        self._log(f"  Ctrl pressed (VK 162/163): {has_ctrl}")
        self._log(f"  Shift pressed (VK 160/161): {has_shift}")
        self._log(f"  Alt pressed (VK 164/165): {has_alt}")

        # Show letter keys
        letter_vks = [v for v in self.pressed_vk_codes if 65 <= v <= 90]
        if letter_vks:
            letters = [f"{chr(v)} (VK={v})" for v in letter_vks]
            self._log(f"  Letter keys: {', '.join(letters)}")

        self._check_hotkeys()

    def _on_release(self, key: Any) -> bool | None:
        """Handle key release."""
        vk = self._get_vk_code(key)
        if vk is not None:
            self.pressed_vk_codes.discard(vk)
            self._log(f"KEY RELEASED: {key} (VK={vk})")
            self._log(f"  Remaining VK codes: {sorted(self.pressed_vk_codes)}")

        if key == keyboard.Key.esc:
            self._log("\nESC pressed - exiting...", also_print=True)
            return False  # Stop listener

        return None

    def run(self) -> None:
        """Run the diagnostic tool."""
        # Clear/create log file
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"Hotkey Diagnostic Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

        print("=" * 60)
        print("DICTACLUTCH HOTKEY DIAGNOSTIC")
        print("=" * 60)
        print()
        print(f"Logging all key events to: {self.log_file}")
        print()
        print("Configured hotkeys to test:")
        for name, hotkey_set in self.hotkeys.items():
            keys = []
            for k in hotkey_set:
                if hasattr(k, 'name'):
                    keys.append(k.name)
                elif hasattr(k, 'char'):
                    keys.append(k.char.upper())
            print(f"  {name}: {' + '.join(keys)}")
        print()
        print("Hotkey detections will be printed here.")
        if self.verbose:
            print("Verbose mode: All key events will be printed.")
        else:
            print("Full details are written to the log file.")
        print()
        print("Press ESC to exit")
        print()
        print("Expected VK codes:")
        print("  Ctrl_L = 162, Ctrl_R = 163")
        print("  Shift_L = 160, Shift_R = 161")
        print("  Alt_L = 164, Alt_R = 165")
        print("  J = 74, K = 75")
        print()
        print("Waiting for key presses...")
        print()

        self._log("Diagnostic started - waiting for key presses")
        self._log(f"CTRL_VK_CODES = {self.CTRL_VK_CODES}")
        self._log(f"SHIFT_VK_CODES = {self.SHIFT_VK_CODES}")
        self._log(f"ALT_VK_CODES = {self.ALT_VK_CODES}")
        self._log("")

        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )

        with self.listener:
            try:
                self.listener.join()
            except KeyboardInterrupt:
                self._log("\nCtrl+C - exiting...", also_print=True)

        print(f"\nLog saved to: {self.log_file}")


def run_diagnostic(verbose: bool = False) -> None:
    """
    Run the hotkey diagnostic tool.

    Args:
        verbose: If True, print all key events to console
    """
    diagnostic = HotkeyDiagnostic(verbose=verbose)
    diagnostic.run()
