"""
Hotkey handling for DictaClutch.

Provides the MultiHotkeyHandler class for managing multiple global
hotkey combinations with separate callbacks.
"""

from typing import Any, Callable

from pynput import keyboard


class MultiHotkeyHandler:
    """
    Handles multiple global hotkey combinations with separate callbacks.

    Uses VK codes for consistent key tracking across different keyboard
    states and layouts.
    """

    # VK codes for modifier keys (used for consistent tracking)
    CTRL_VK_CODES = {162, 163}  # VK_LCONTROL, VK_RCONTROL
    SHIFT_VK_CODES = {160, 161}  # VK_LSHIFT, VK_RSHIFT
    ALT_VK_CODES = {164, 165}  # VK_LMENU, VK_RMENU (Alt keys)

    def __init__(
        self,
        hotkey_configs: dict[str, set],
        listener_factory: Callable[..., keyboard.Listener] | None = None,
    ):
        """
        Initialize with hotkey configurations.

        Args:
            hotkey_configs: Dict mapping mode names to key sets, e.g.:
                {
                    "batch": {keyboard.Key.ctrl, keyboard.Key.shift, keyboard.KeyCode.from_char("j")},
                    "streaming": {keyboard.Key.ctrl, keyboard.Key.shift, keyboard.KeyCode.from_char("k")},
                }
            listener_factory: Optional factory for creating keyboard listeners (for testing)
        """
        self.hotkey_configs = hotkey_configs
        self._listener_factory = listener_factory or keyboard.Listener
        self.callbacks: dict[str, Callable[[], None]] = {}
        self.exit_callback: Callable[[], None] | None = None
        # Track keys by VK code to avoid object equality issues
        # When Ctrl+Shift is held, pynput may report different key objects
        # for press vs release, causing set.discard() to fail
        self.pressed_vk_codes: set[int] = set()
        self.triggered: dict[str, bool] = {}
        self.listener: keyboard.Listener | None = None

        for mode in hotkey_configs:
            self.triggered[mode] = False

    def register_callback(self, mode: str, callback: Callable[[], None]) -> None:
        """Register callback for a hotkey mode."""
        self.callbacks[mode] = callback

    def register_exit_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for ESC key exit."""
        self.exit_callback = callback

    def _get_vk_code(self, key: Any) -> int | None:
        """Extract VK code from a key, handling both Key enums and KeyCodes."""
        # KeyCode objects have a .vk attribute
        if hasattr(key, "vk") and key.vk is not None:
            return key.vk
        # Key enums (ctrl, shift, etc.) have a .value attribute with vk
        if hasattr(key, "value") and hasattr(key.value, "vk"):
            return key.value.vk
        return None

    def _get_letter_vk(self, hotkey_set: set) -> int | None:
        """Extract the VK code for the letter key from a hotkey configuration."""
        for k in hotkey_set:
            if hasattr(k, "char") and k.char:
                # Convert letter to VK code (A=65, B=66, ..., J=74, K=75, ...)
                return ord(k.char.upper())
        return None

    def _requires_modifier(self, hotkey_set: set, modifier_key: keyboard.Key) -> bool:
        """Check if a hotkey config requires a specific modifier key."""
        for k in hotkey_set:
            if k == modifier_key:
                return True
            # Also check for generic ctrl/shift/alt that match specific variants
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

    def _is_hotkey_pressed(self, mode: str) -> bool:
        """Check if specific hotkey combo is pressed using VK codes."""
        required = self.hotkey_configs[mode]

        # Dynamically check which modifiers are required by this hotkey
        needs_ctrl = self._requires_modifier(required, keyboard.Key.ctrl)
        needs_shift = self._requires_modifier(required, keyboard.Key.shift)
        needs_alt = self._requires_modifier(required, keyboard.Key.alt)

        # Check if required modifiers are pressed
        has_ctrl = bool(self.pressed_vk_codes & self.CTRL_VK_CODES)
        has_shift = bool(self.pressed_vk_codes & self.SHIFT_VK_CODES)
        has_alt = bool(self.pressed_vk_codes & self.ALT_VK_CODES)

        # Verify all required modifiers are pressed
        if needs_ctrl and not has_ctrl:
            return False
        if needs_shift and not has_shift:
            return False
        if needs_alt and not has_alt:
            return False

        # Get the letter VK code from the config
        letter_vk = self._get_letter_vk(required)

        # Check if the letter key is pressed by VK code
        has_letter = letter_vk is not None and letter_vk in self.pressed_vk_codes

        return has_letter

    def _on_press(self, key: Any) -> None:
        """Handle key press - track by VK code."""
        vk = self._get_vk_code(key)
        if vk is not None:
            self.pressed_vk_codes.add(vk)

        for mode in self.hotkey_configs:
            if not self.triggered[mode] and self._is_hotkey_pressed(mode):
                self.triggered[mode] = True
                if mode in self.callbacks:
                    self.callbacks[mode]()

    def _on_release(self, key: Any) -> None:
        """Handle key release - track by VK code."""
        # Check for ESC key to exit
        if key == keyboard.Key.esc:
            if self.exit_callback:
                self.exit_callback()
            return  # Exit callback handles shutdown

        vk = self._get_vk_code(key)
        if vk is not None:
            self.pressed_vk_codes.discard(vk)

        for mode in self.hotkey_configs:
            if not self._is_hotkey_pressed(mode):
                self.triggered[mode] = False

    def start(self) -> None:
        """Start listening for hotkeys."""
        self.listener = self._listener_factory(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.start()

    def stop(self) -> None:
        """Stop listening."""
        if self.listener:
            self.listener.stop()
