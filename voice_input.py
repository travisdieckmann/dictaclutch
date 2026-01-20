#!/usr/bin/env python3
"""
Voice-to-Text Input Tool
Capture audio with a hotkey and transcribe using local Whisper, then paste into active window.
Works on Windows and Linux.
"""

import sys
import threading
import queue
import time
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from pynput import keyboard
from faster_whisper import WhisperModel
import pyperclip

# Platform-specific imports
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
else:
    # Linux - we'll use xdotool for typing
    import subprocess

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Batch mode hotkey (existing behavior): Ctrl+Shift+J
    "hotkey_batch": {keyboard.Key.ctrl, keyboard.Key.shift, keyboard.KeyCode.from_char("j")},
    # Streaming mode hotkey (new): Ctrl+Shift+K
    "hotkey_streaming": {keyboard.Key.ctrl, keyboard.Key.shift, keyboard.KeyCode.from_char("k")},
    # Whisper model: tiny, base, small, medium, large-v2, large-v3
    # Recommended for 4GB VRAM: small
    "model_size": "small",
    # Device: "cuda" for GPU, "cpu" for CPU-only
    "device": "cuda",
    # Compute type: float16 for GPU (faster), int8 for CPU, float32 for compatibility
    "compute_type": "float16",
    # Audio settings
    "sample_rate": 16000,
    "channels": 1,
    # Language: None for auto-detect, or "en", "es", "fr", etc.
    "language": "en",
    # Paste method: "clipboard" (Ctrl+V) or "type" (simulate keystrokes - slower but more compatible)
    "paste_method": "type",
    # Audio feedback
    "beep_on_start": True,
    "beep_on_stop": True,
    # Streaming mode settings
    "streaming_min_chunk": 1.0,        # Seconds of audio before transcription attempt
    "streaming_agreement_n": 2,         # Number of consecutive agreements for confirmation
    "streaming_buffer_max": 30.0,       # Maximum buffer size (seconds) before trimming
    "streaming_use_vad": True,          # Use Voice Activity Detection
}

# ============================================================================
# AUDIO FEEDBACK
# ============================================================================


def generate_beep(
    frequency: float, duration: float, sample_rate: int = 44100
) -> np.ndarray:
    """Generate a simple sine wave beep."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t) * 0.3
    # Apply fade in/out to avoid clicks
    fade_samples = int(sample_rate * 0.01)
    wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
    wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return wave.astype(np.float32)


def play_beep(frequency: float = 440, duration: float = 0.1):
    """Play a beep sound."""
    try:
        beep = generate_beep(frequency, duration)
        sd.play(beep, samplerate=44100)
    except Exception as e:
        print(f"Could not play beep: {e}")


def beep_start():
    """Beep indicating recording started (ascending tone)."""
    if CONFIG["beep_on_start"]:
        play_beep(440, 0.08)
        time.sleep(0.1)
        play_beep(880, 0.08)


def beep_stop():
    """Beep indicating recording stopped (descending tone)."""
    if CONFIG["beep_on_stop"]:
        play_beep(880, 0.08)
        time.sleep(0.1)
        play_beep(440, 0.08)


def beep_error():
    """Beep indicating an error."""
    play_beep(220, 0.3)


# ============================================================================
# TEXT INPUT (PASTE/TYPE)
# ============================================================================


def paste_text(text: str):
    """Paste text into the active window."""
    if not text.strip():
        return

    if CONFIG["paste_method"] == "clipboard":
        paste_via_clipboard(text)
    else:
        type_text(text)


def paste_via_clipboard(text: str):
    """Copy to clipboard and simulate Ctrl+V."""
    pyperclip.copy(text)
    time.sleep(0.05)  # Small delay to ensure clipboard is ready

    if sys.platform == "win32":
        # Windows: use SendInput
        send_ctrl_v_windows()
    else:
        # Linux: use xdotool
        subprocess.run(["xdotool", "key", "ctrl+v"], check=False)


def send_ctrl_v_windows():
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


def type_text(text: str):
    """Type text character by character (fallback method)."""
    if sys.platform == "win32":
        # Use SendInput for each character
        for char in text:
            type_char_windows(char)
            time.sleep(0.01)
    else:
        # Linux: use xdotool
        subprocess.run(["xdotool", "type", "--", text], check=False)


def type_char_windows(char: str):
    """Type a single character on Windows."""
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


# ============================================================================
# WINDOW FOCUS DETECTION
# ============================================================================


def get_foreground_window() -> int:
    """Get handle of currently focused window (Windows only)."""
    if sys.platform == "win32":
        return ctypes.windll.user32.GetForegroundWindow()
    else:
        # Linux: return 0 (focus tracking not implemented)
        return 0


# ============================================================================
# AUDIO RECORDING
# ============================================================================


class AudioRecorder:
    """Records audio from the default microphone."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.audio_data = []
        self.stream = None

        # Log the audio input device
        try:
            default_input = sd.query_devices(kind="input")
            print(f"Audio input device: {default_input['name']}")
        except Exception as e:
            print(f"Could not query audio device: {e}")

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")
        if self.recording:
            self.audio_data.append(indata.copy())

    def start(self):
        """Start recording."""
        self.audio_data = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            callback=self._audio_callback,
        )
        self.stream.start()

    def stop(self) -> np.ndarray:
        """Stop recording and return audio data."""
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.audio_data:
            return np.array([], dtype=np.float32)

        return np.concatenate(self.audio_data, axis=0).flatten()

    def save_wav(self, audio_data: np.ndarray, filepath: str):
        """Save audio data to a WAV file."""
        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())


# ============================================================================
# TRANSCRIPTION
# ============================================================================


class Transcriber:
    """Transcribes audio using faster-whisper."""

    def __init__(self, model_size: str, device: str, compute_type: str):
        print(f"Loading Whisper model '{model_size}' on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Model loaded!")

    def transcribe(self, audio_data: np.ndarray, language: str = None) -> str:
        """Transcribe audio data to text."""
        if len(audio_data) == 0:
            return ""

        # faster-whisper can accept numpy array directly
        segments, info = self.model.transcribe(
            audio_data,
            language=language,
            beam_size=5,
            vad_filter=True,  # Filter out silence
        )

        # Collect all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)

        return " ".join(text_parts).strip()


# ============================================================================
# HOTKEY HANDLER
# ============================================================================


class MultiHotkeyHandler:
    """Handles multiple global hotkey combinations with separate callbacks."""

    def __init__(self, hotkey_configs: dict):
        """
        Initialize with hotkey configurations.

        Args:
            hotkey_configs: Dict mapping mode names to key sets, e.g.:
                {
                    "batch": {keyboard.Key.ctrl, keyboard.Key.shift, keyboard.KeyCode.from_char("j")},
                    "streaming": {keyboard.Key.ctrl, keyboard.Key.shift, keyboard.KeyCode.from_char("k")},
                }
        """
        self.hotkey_configs = hotkey_configs
        self.callbacks = {}  # mode -> callback
        self.current_keys = set()
        self.triggered = {}  # mode -> bool (prevent repeat)
        self.listener = None

        for mode in hotkey_configs:
            self.triggered[mode] = False

    def register_callback(self, mode: str, callback):
        """Register callback for a hotkey mode."""
        self.callbacks[mode] = callback

    def _get_letter_key(self, hotkey_set: set) -> str | None:
        """Extract the letter key from a hotkey configuration."""
        for k in hotkey_set:
            if hasattr(k, "char") and k.char:
                return k.char.lower()
        return None

    def _is_hotkey_pressed(self, mode: str) -> bool:
        """Check if specific hotkey combo is pressed."""
        required = self.hotkey_configs[mode]

        # Check for Ctrl (any variant)
        has_ctrl = any(
            k in [keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r]
            for k in self.current_keys
        )

        # Check for Shift (any variant)
        has_shift = any(
            k in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]
            for k in self.current_keys
        )

        # Get the letter key from the config
        letter_key = self._get_letter_key(required)

        # Check for the letter key by char or virtual key code
        has_letter = False
        if letter_key:
            # Map letter to VK code (A=65, B=66, ..., J=74, K=75, ...)
            vk_code = ord(letter_key.upper())
            has_letter = any(
                (hasattr(k, "vk") and k.vk == vk_code)
                or (hasattr(k, "char") and k.char and k.char.lower() == letter_key)
                for k in self.current_keys
            )

        return has_ctrl and has_shift and has_letter

    def _on_press(self, key):
        """Handle key press."""
        self.current_keys.add(key)

        for mode in self.hotkey_configs:
            if not self.triggered[mode] and self._is_hotkey_pressed(mode):
                self.triggered[mode] = True
                if mode in self.callbacks:
                    self.callbacks[mode]()

    def _on_release(self, key):
        """Handle key release."""
        self.current_keys.discard(key)

        for mode in self.hotkey_configs:
            if not self._is_hotkey_pressed(mode):
                self.triggered[mode] = False

    def start(self):
        """Start listening for hotkeys."""
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.start()

    def stop(self):
        """Stop listening."""
        if self.listener:
            self.listener.stop()


# ============================================================================
# MAIN APPLICATION
# ============================================================================


class VoiceInputApp:
    """Main application coordinating recording, transcription, and input."""

    def __init__(self):
        self.recorder = AudioRecorder(
            sample_rate=CONFIG["sample_rate"],
            channels=CONFIG["channels"],
        )
        self.transcriber = None  # Lazy load

        # Mode tracking: None, "batch", or "streaming"
        self.mode = None
        self.target_window = None  # Window handle for focus tracking

        # Setup multi-hotkey handler
        hotkey_configs = {
            "batch": CONFIG["hotkey_batch"],
            "streaming": CONFIG["hotkey_streaming"],
        }
        self.hotkey_handler = MultiHotkeyHandler(hotkey_configs)
        self.hotkey_handler.register_callback("batch", self.toggle_batch)
        self.hotkey_handler.register_callback("streaming", self.toggle_streaming)

        # Batch mode state
        self.is_recording = False
        self.processing = False
        self.lock = threading.Lock()

    def load_model(self):
        """Load the Whisper model."""
        self.transcriber = Transcriber(
            model_size=CONFIG["model_size"],
            device=CONFIG["device"],
            compute_type=CONFIG["compute_type"],
        )

    def toggle_batch(self):
        """Toggle batch recording mode (Ctrl+Shift+J)."""
        with self.lock:
            # If streaming is active, stop it first
            if self.mode == "streaming":
                self._stop_streaming()

            if self.processing:
                return  # Ignore if still processing previous recording

            if self.mode != "batch":
                self._start_batch()
            else:
                self._stop_batch()

    def toggle_streaming(self):
        """Toggle streaming mode (Ctrl+Shift+K)."""
        with self.lock:
            # If batch is active, stop it first
            if self.mode == "batch":
                self._stop_batch()

            if self.mode != "streaming":
                self._start_streaming()
            else:
                self._stop_streaming()

    def _start_batch(self):
        """Start batch recording mode."""
        print("\n🎤 [BATCH] Recording... (press Ctrl+Shift+J to stop)")
        self.mode = "batch"
        self.is_recording = True
        self.target_window = get_foreground_window()
        beep_start()
        self.recorder.start()

    def _stop_batch(self):
        """Stop batch recording and process."""
        print("⏹️  [BATCH] Stopped recording, transcribing...")
        self.is_recording = False
        self.processing = True
        self.mode = None
        beep_stop()

        # Get audio data
        audio_data = self.recorder.stop()

        # Process in background thread
        thread = threading.Thread(target=self._process_batch_audio, args=(audio_data,))
        thread.start()

    def _start_streaming(self):
        """Start streaming mode (placeholder for Phase 2)."""
        print("\n🎙️  [STREAMING] Mode started (press Ctrl+Shift+K to stop)")
        print("    ⚠️  Streaming transcription not yet implemented - coming in Phase 2")
        self.mode = "streaming"
        self.target_window = get_foreground_window()
        beep_start()

    def _stop_streaming(self):
        """Stop streaming mode (placeholder for Phase 2)."""
        print("⏹️  [STREAMING] Mode stopped")
        self.mode = None
        beep_stop()

    def _process_batch_audio(self, audio_data: np.ndarray):
        """Process recorded audio for batch mode (runs in background thread)."""
        try:
            if len(audio_data) < CONFIG["sample_rate"] * 0.3:  # Less than 0.3 seconds
                print("Recording too short, ignoring.")
                return

            # Transcribe
            start_time = time.time()
            text = self.transcriber.transcribe(audio_data, CONFIG["language"])
            elapsed = time.time() - start_time

            if text:
                # Add space after ending punctuation for better flow
                if text and text[-1] in ".!?":
                    text += " "

                print(f"📝 Transcribed ({elapsed:.2f}s): {text}")

                # Small delay to ensure hotkey is fully released
                time.sleep(0.15)

                # Check if target window is still focused
                if self.target_window and get_foreground_window() != self.target_window:
                    print("⚠️  Window focus changed, skipping paste")
                else:
                    # Paste text
                    paste_text(text)
            else:
                print("No speech detected.")

        except Exception as e:
            print(f"Error during transcription: {e}")
            beep_error()

        finally:
            with self.lock:
                self.processing = False

    def run(self):
        """Run the application."""
        print("=" * 60)
        print("Voice-to-Text Input Tool (Dual Mode)")
        print("=" * 60)
        print()

        # Load model
        self.load_model()

        # Format hotkeys for display
        def format_hotkey(hotkey_set):
            return " + ".join(
                k.name if hasattr(k, "name") else (k.char if hasattr(k, "char") else str(k))
                for k in hotkey_set
            )

        batch_hotkey = format_hotkey(CONFIG["hotkey_batch"])
        streaming_hotkey = format_hotkey(CONFIG["hotkey_streaming"])

        print(f"\n✅ Ready!")
        print(f"   [{batch_hotkey}] - Batch mode (record → transcribe → paste)")
        print(f"   [{streaming_hotkey}] - Streaming mode (real-time transcription)")
        print("Press Ctrl+C to exit.\n")

        # Start hotkey listener
        self.hotkey_handler.start()

        try:
            # Keep main thread alive
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nExiting...")
            self.hotkey_handler.stop()


# ============================================================================
# ENTRY POINT
# ============================================================================


def main():
    # Check for CUDA availability
    try:
        import torch

        if CONFIG["device"] == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            CONFIG["device"] = "cpu"
            CONFIG["compute_type"] = "int8"
        elif CONFIG["device"] == "cuda":
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    except ImportError:
        if CONFIG["device"] == "cuda":
            print("WARNING: PyTorch not found, falling back to CPU")
            CONFIG["device"] = "cpu"
            CONFIG["compute_type"] = "int8"

    app = VoiceInputApp()
    app.run()


if __name__ == "__main__":
    main()
