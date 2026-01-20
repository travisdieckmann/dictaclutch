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
    "streaming_min_chunk": 0.5,         # Seconds between transcription attempts (lower = more real-time)
    "streaming_buffer_max": 30.0,       # Maximum buffer size (seconds) before trimming
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


def generate_tone_sequence(frequencies: list[float], tone_duration: float = 0.10, gap_duration: float = 0.03) -> np.ndarray:
    """Generate a sequence of tones with gaps between them."""
    sample_rate = 44100
    tones = []
    gap_samples = int(sample_rate * gap_duration)
    gap = np.zeros(gap_samples, dtype=np.float32)

    for i, freq in enumerate(frequencies):
        tone = generate_beep(freq, tone_duration, sample_rate)
        tones.append(tone)
        if i < len(frequencies) - 1:  # Add gap between tones, not after last
            tones.append(gap)

    return np.concatenate(tones)


def play_tone_sequence(frequencies: list[float]):
    """Play a sequence of tones."""
    try:
        sequence = generate_tone_sequence(frequencies)
        sd.play(sequence, samplerate=44100)
        sd.wait()  # Block until playback finishes
    except Exception as e:
        print(f"Could not play beep sequence: {e}")


def beep_batch_start():
    """Beep indicating batch recording started (4-tone ascending, lower range)."""
    if CONFIG["beep_on_start"]:
        # E4 → G4 → A4 → C5 (warm, steady progression)
        play_tone_sequence([330, 392, 440, 523])


def beep_batch_stop():
    """Beep indicating batch recording stopped (4-tone descending, lower range)."""
    if CONFIG["beep_on_stop"]:
        # C5 → A4 → G4 → E4
        play_tone_sequence([523, 440, 392, 330])


def beep_streaming_start():
    """Beep indicating streaming started (4-tone ascending, higher range)."""
    if CONFIG["beep_on_start"]:
        # C5 → D5 → E5 → G5 (bright, dynamic progression)
        play_tone_sequence([523, 587, 659, 784])


def beep_streaming_stop():
    """Beep indicating streaming stopped (4-tone descending, higher range)."""
    if CONFIG["beep_on_stop"]:
        # G5 → E5 → D5 → C5
        play_tone_sequence([784, 659, 587, 523])


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


def send_backspace_windows(count: int):
    """Send backspace key N times to delete characters (Windows only)."""
    if sys.platform != "win32" or count <= 0:
        return

    VK_BACK = 0x08
    KEYEVENTF_KEYUP = 0x0002
    user32 = ctypes.windll.user32

    for _ in range(count):
        user32.keybd_event(VK_BACK, 0, 0, 0)
        user32.keybd_event(VK_BACK, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.01)


def send_key_windows(vk_code: int, count: int = 1):
    """Send a virtual key N times (Windows only)."""
    if sys.platform != "win32" or count <= 0:
        return

    KEYEVENTF_KEYUP = 0x0002
    user32 = ctypes.windll.user32

    for _ in range(count):
        user32.keybd_event(vk_code, 0, 0, 0)
        user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.005)  # Faster than backspace


def send_left_arrow_windows(count: int):
    """Send left arrow key N times (Windows only)."""
    VK_LEFT = 0x25
    send_key_windows(VK_LEFT, count)


def send_ctrl_left(count: int):
    """Send Ctrl+Left arrow N times to jump back by words."""
    if count <= 0:
        return

    if sys.platform == "win32":
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
    else:
        # Linux: use xdotool
        for _ in range(count):
            subprocess.run(["xdotool", "key", "ctrl+Left"], check=False)
            time.sleep(0.005)


def send_ctrl_backspace(count: int):
    """Send Ctrl+Backspace N times to delete words."""
    if count <= 0:
        return

    if sys.platform == "win32":
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
    else:
        # Linux: use xdotool
        for _ in range(count):
            subprocess.run(["xdotool", "key", "ctrl+BackSpace"], check=False)
            time.sleep(0.01)


def send_end_key():
    """Send End key to go to end of line."""
    if sys.platform == "win32":
        VK_END = 0x23
        send_key_windows(VK_END, 1)
    else:
        # Linux: use xdotool
        subprocess.run(["xdotool", "key", "End"], check=False)


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


class StreamingAudioRecorder:
    """Queue-based audio recorder for real-time streaming."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_queue = queue.Queue()  # Thread-safe delivery
        self.stream = None
        self.recording = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream - adds chunks to queue."""
        if status:
            print(f"Audio status: {status}")
        if self.recording:
            self.chunk_queue.put(indata.copy())

    def start(self):
        """Start recording."""
        self.recording = True
        # Clear any old chunks
        while not self.chunk_queue.empty():
            try:
                self.chunk_queue.get_nowait()
            except queue.Empty:
                break

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            callback=self._audio_callback,
        )
        self.stream.start()

    def stop(self):
        """Stop recording."""
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_chunk(self, timeout: float = 0.1) -> np.ndarray | None:
        """Get next audio chunk (non-blocking with timeout)."""
        try:
            return self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ============================================================================
# INCREMENTAL BUFFER (Smart Diff Corrections)
# ============================================================================


class IncrementalBuffer:
    """
    Tracks typed text and computes minimal corrections using character-level diff.
    Types text as it's spoken and makes minimal corrections when needed.
    """

    def __init__(self):
        self.typed_text: str = ""  # What's actually been typed to the screen
        self.transcript_history: list[str] = []  # Ring buffer of recent transcripts

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split()) if text.strip() else 0

    def _get_word_at_end(self, text: str, word_count: int) -> str:
        """Get the last N words from text."""
        words = text.split()
        if word_count >= len(words):
            return text
        return " ".join(words[-word_count:])

    def update(self, new_transcript: str) -> dict:
        """
        Compare new transcript with typed text and compute minimal edit.
        Uses word-based Ctrl+Arrow navigation for efficient middle-text edits.

        Strategies:
        - "append": Just add text at end (no changes needed)
        - "backspace": Delete from end, retype (fallback)
        - "word_jump": Use Ctrl+Left to jump words, Ctrl+Backspace to delete words

        Returns: {
            "strategy": "append" | "backspace" | "word_jump",
            "backspace": N,              # chars to delete (backspace strategy)
            "append": "text",            # text to add
            "word_jumps": N,             # Ctrl+Left count (word_jump strategy)
            "word_deletes": N,           # Ctrl+Backspace count (word_jump strategy)
            "word_insert": "text",       # text to insert after deletion
        }
        """
        if not new_transcript:
            return {"strategy": "append", "backspace": 0, "append": ""}

        # Normalize: strip and collapse whitespace
        new_transcript = " ".join(new_transcript.split())

        # Track history for debugging
        self.transcript_history.append(new_transcript)
        if len(self.transcript_history) > 10:
            self.transcript_history.pop(0)

        # If nothing typed yet, just type the new transcript
        if not self.typed_text:
            self.typed_text = new_transcript
            return {"strategy": "append", "backspace": 0, "append": new_transcript}

        old = self.typed_text
        new = new_transcript

        # Word-based analysis
        old_words = old.split()
        new_words = new.split()

        # Find common word prefix
        common_prefix_words = 0
        for i in range(min(len(old_words), len(new_words))):
            if old_words[i] == new_words[i]:
                common_prefix_words = i + 1
            else:
                break

        # Find common word suffix (but don't overlap with prefix)
        common_suffix_words = 0
        old_remaining = len(old_words) - common_prefix_words
        new_remaining = len(new_words) - common_prefix_words
        max_suffix = min(old_remaining, new_remaining)

        for i in range(1, max_suffix + 1):
            if old_words[-i] == new_words[-i]:
                common_suffix_words = i
            else:
                break

        # Calculate word-level changes
        old_middle_words = len(old_words) - common_prefix_words - common_suffix_words
        new_middle_words = new_words[common_prefix_words:len(new_words) - common_suffix_words] if common_suffix_words > 0 else new_words[common_prefix_words:]
        new_middle_text = " ".join(new_middle_words)

        # Also do character-level for fallback
        prefix_len = 0
        min_len = min(len(old), len(new))
        for i in range(min_len):
            if old[i] == new[i]:
                prefix_len = i + 1
            else:
                break

        suffix_len = 0
        old_char_remaining = len(old) - prefix_len
        new_char_remaining = len(new) - prefix_len
        max_char_suffix = min(old_char_remaining, new_char_remaining)

        for i in range(1, max_char_suffix + 1):
            if old[-i] == new[-i]:
                suffix_len = i
            else:
                break

        old_middle_len = len(old) - prefix_len - suffix_len
        new_middle_char = new[prefix_len:len(new) - suffix_len] if suffix_len > 0 else new[prefix_len:]
        suffix_text = new[-suffix_len:] if suffix_len > 0 else ""

        # Update what we consider "typed"
        self.typed_text = new_transcript

        # Calculate operation costs
        # Word jump: Ctrl+Left × (suffix_words) + Ctrl+Backspace × (old_middle_words) + type + End
        word_jump_ops = common_suffix_words + old_middle_words + len(new_middle_text) + 1

        # Backspace: backspace × (old_middle_chars + suffix_chars) + type
        backspace_ops = (old_middle_len + suffix_len) + len(new_middle_char) + suffix_len

        # Decide strategy
        # Use word_jump if:
        # 1. There are suffix words to preserve (common_suffix_words >= 1)
        # 2. There are middle words to delete (old_middle_words >= 1)
        # 3. Word approach is more efficient
        use_word_jump = (
            common_suffix_words >= 1 and
            old_middle_words >= 1 and
            word_jump_ops < backspace_ops
        )

        if use_word_jump:
            # Need to add trailing space if inserting before more words
            insert_text = new_middle_text
            if new_middle_text and common_suffix_words > 0:
                insert_text = new_middle_text + " "

            return {
                "strategy": "word_jump",
                "word_jumps": common_suffix_words,
                "word_deletes": old_middle_words,
                "word_insert": insert_text,
                "backspace": 0,
                "append": "",
            }
        else:
            return {
                "strategy": "backspace",
                "backspace": old_middle_len + suffix_len,
                "append": new_middle_char + suffix_text,
                "word_jumps": 0,
                "word_deletes": 0,
                "word_insert": "",
            }

    def get_stable_prefix(self) -> str:
        """
        Find text that's been stable across recent transcripts.
        Useful for knowing what we're confident about.
        """
        if len(self.transcript_history) < 2:
            return ""

        # Find common prefix across last few transcripts
        texts = self.transcript_history[-3:]  # Last 3
        if not texts:
            return ""

        prefix = texts[0]
        for text in texts[1:]:
            # Find common prefix
            common_len = 0
            for i in range(min(len(prefix), len(text))):
                if prefix[i] == text[i]:
                    common_len = i + 1
                else:
                    break
            prefix = prefix[:common_len]

        return prefix

    def reset(self):
        """Clear all state (on recording start/stop)."""
        self.typed_text = ""
        self.transcript_history.clear()


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


class StreamingTranscriber:
    """
    Real-time streaming transcription with incremental output.
    Transcribes at fixed intervals and uses smart diff for minimal corrections.
    """

    def __init__(self, model: WhisperModel, config: dict):
        self.model = model
        self.sample_rate = config["sample_rate"]
        self.min_chunk_size = config["streaming_min_chunk"]
        self.max_buffer_size = config["streaming_buffer_max"]
        self.language = config["language"]

        # Audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)

        # Incremental buffer for smart diff corrections
        self.incremental_buffer = IncrementalBuffer()

        # Tracking
        self.last_transcript = ""

    def add_chunk(self, chunk: np.ndarray):
        """Add audio chunk to rolling buffer."""
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk.flatten()])

    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds."""
        return len(self.audio_buffer) / self.sample_rate

    def should_transcribe(self) -> bool:
        """Check if we have enough audio to transcribe (time-based, no VAD wait)."""
        return self.get_buffer_duration() >= self.min_chunk_size

    def transcribe_buffer(self) -> dict:
        """
        Transcribe current buffer and return incremental edit operations.

        Returns: {
            "backspace": N,  # chars to delete
            "append": "text to add"
        }
        """
        if len(self.audio_buffer) == 0:
            return {"backspace": 0, "append": ""}

        # Transcribe
        segments, _ = self.model.transcribe(
            self.audio_buffer,
            language=self.language,
            beam_size=5,
            vad_filter=False,  # Don't filter - we want real-time output
        )

        # Collect transcript
        transcript = " ".join(seg.text for seg in segments).strip()
        self.last_transcript = transcript

        # Get incremental edit (smart diff)
        return self.incremental_buffer.update(transcript)

    def trim_buffer(self):
        """Trim buffer to max size, keeping recent audio."""
        max_samples = int(self.max_buffer_size * self.sample_rate)
        if len(self.audio_buffer) > max_samples:
            # Keep the most recent audio
            self.audio_buffer = self.audio_buffer[-max_samples:]

    def reset(self):
        """Clear all buffers."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.incremental_buffer.reset()
        self.last_transcript = ""


# ============================================================================
# HOTKEY HANDLER
# ============================================================================


class MultiHotkeyHandler:
    """Handles multiple global hotkey combinations with separate callbacks."""

    # VK codes for modifier keys (used for consistent tracking)
    CTRL_VK_CODES = {162, 163}  # VK_LCONTROL, VK_RCONTROL
    SHIFT_VK_CODES = {160, 161}  # VK_LSHIFT, VK_RSHIFT

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
        self.exit_callback = None  # Called when ESC is pressed
        # Track keys by VK code to avoid object equality issues
        # When Ctrl+Shift is held, pynput may report different key objects
        # for press vs release, causing set.discard() to fail
        self.pressed_vk_codes: set[int] = set()
        self.triggered = {}  # mode -> bool (prevent repeat)
        self.listener = None

        for mode in hotkey_configs:
            self.triggered[mode] = False

    def register_callback(self, mode: str, callback):
        """Register callback for a hotkey mode."""
        self.callbacks[mode] = callback

    def register_exit_callback(self, callback):
        """Register callback for ESC key exit."""
        self.exit_callback = callback

    def _get_vk_code(self, key) -> int | None:
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

    def _is_hotkey_pressed(self, mode: str) -> bool:
        """Check if specific hotkey combo is pressed using VK codes."""
        required = self.hotkey_configs[mode]

        # Check for Ctrl (any variant) by VK code
        has_ctrl = bool(self.pressed_vk_codes & self.CTRL_VK_CODES)

        # Check for Shift (any variant) by VK code
        has_shift = bool(self.pressed_vk_codes & self.SHIFT_VK_CODES)

        # Get the letter VK code from the config
        letter_vk = self._get_letter_vk(required)

        # Check if the letter key is pressed by VK code
        has_letter = letter_vk is not None and letter_vk in self.pressed_vk_codes

        return has_ctrl and has_shift and has_letter

    def _on_press(self, key):
        """Handle key press - track by VK code."""
        vk = self._get_vk_code(key)
        if vk is not None:
            self.pressed_vk_codes.add(vk)

        for mode in self.hotkey_configs:
            if not self.triggered[mode] and self._is_hotkey_pressed(mode):
                self.triggered[mode] = True
                if mode in self.callbacks:
                    self.callbacks[mode]()

    def _on_release(self, key):
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
        # Batch mode recorder
        self.recorder = AudioRecorder(
            sample_rate=CONFIG["sample_rate"],
            channels=CONFIG["channels"],
        )
        self.transcriber = None  # Lazy load (shared model)

        # Streaming mode components
        self.streaming_recorder = StreamingAudioRecorder(
            sample_rate=CONFIG["sample_rate"],
            channels=CONFIG["channels"],
        )
        self.streaming_transcriber = None  # Lazy load
        self.stop_event = threading.Event()  # For clean shutdown

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
        self.hotkey_handler.register_exit_callback(self._on_exit)

        # Batch mode state
        self.is_recording = False
        self.processing = False
        self.lock = threading.Lock()

        # Exit flag
        self.exit_requested = False

    def _on_exit(self):
        """Handle ESC key press - clean shutdown."""
        print("\nESC pressed - exiting...")
        self.exit_requested = True
        # Stop any active mode
        if self.mode == "streaming":
            self._stop_streaming()
        elif self.mode == "batch":
            self.recorder.stop()
        self.hotkey_handler.stop()

    def load_model(self):
        """Load the Whisper model."""
        self.transcriber = Transcriber(
            model_size=CONFIG["model_size"],
            device=CONFIG["device"],
            compute_type=CONFIG["compute_type"],
        )
        # Initialize streaming transcriber with the same model
        self.streaming_transcriber = StreamingTranscriber(
            model=self.transcriber.model,
            config=CONFIG,
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

        # Play beep BEFORE starting recorder (so beep isn't captured)
        beep_batch_start()
        time.sleep(0.1)  # Let echo die down

        self.recorder.start()

    def _stop_batch(self):
        """Stop batch recording and process."""
        print("⏹️  [BATCH] Stopped recording, transcribing...")
        self.is_recording = False
        self.processing = True
        self.mode = None

        # Get audio data BEFORE playing beep (so beep isn't captured)
        audio_data = self.recorder.stop()

        beep_batch_stop()

        # Process in background thread
        thread = threading.Thread(target=self._process_batch_audio, args=(audio_data,))
        thread.start()

    def _start_streaming(self):
        """Start streaming mode with real-time transcription."""
        # Ensure model is loaded
        if self.streaming_transcriber is None:
            print("❌ Model not loaded yet!")
            return

        print("\n🎙️  [STREAMING] Mode started (press Ctrl+Shift+K to stop)")
        self.mode = "streaming"
        self.target_window = get_foreground_window()

        # Reset streaming components
        self.stop_event.clear()
        self.streaming_transcriber.reset()

        # Play beep BEFORE starting recorder (so beep isn't captured)
        beep_streaming_start()
        time.sleep(0.1)  # Let echo die down

        # Start audio recording
        self.streaming_recorder.start()

        # Start worker thread
        thread = threading.Thread(target=self._streaming_worker, daemon=True)
        thread.start()

    def _stop_streaming(self):
        """Stop streaming mode."""
        print("⏹️  [STREAMING] Mode stopped")

        # Signal worker to stop
        self.stop_event.set()

        # Stop recording
        self.streaming_recorder.stop()

        self.mode = None
        beep_streaming_stop()

    def _streaming_worker(self):
        """Background worker for real-time streaming transcription with smart diff corrections."""
        last_transcribe_time = time.time()

        while not self.stop_event.is_set():
            # Get audio chunks from recorder
            chunk = self.streaming_recorder.get_chunk(timeout=0.1)

            if chunk is not None:
                self.streaming_transcriber.add_chunk(chunk)

            # Check if we should transcribe (time-based, no VAD wait)
            current_time = time.time()
            time_since_last = current_time - last_transcribe_time

            if (self.streaming_transcriber.should_transcribe() and
                    time_since_last >= CONFIG["streaming_min_chunk"]):
                try:
                    # Transcribe and get incremental edit
                    result = self.streaming_transcriber.transcribe_buffer()
                    strategy = result.get("strategy", "backspace")

                    # Check if target window is still focused
                    if self.target_window and get_foreground_window() != self.target_window:
                        has_changes = (result.get("backspace", 0) or result.get("append", "") or
                                       result.get("word_deletes", 0) or result.get("word_insert", ""))
                        if has_changes:
                            print(f"    [paused - window not focused]")
                    else:
                        if strategy == "word_jump":
                            # Word-jump strategy: Ctrl+Left to jump words, Ctrl+Backspace to delete
                            word_jumps = result.get("word_jumps", 0)
                            word_deletes = result.get("word_deletes", 0)
                            word_insert = result.get("word_insert", "")

                            if word_jumps > 0 or word_deletes > 0 or word_insert:
                                # Jump backward by words
                                send_ctrl_left(word_jumps)
                                # Delete words
                                send_ctrl_backspace(word_deletes)
                                # Insert replacement
                                if word_insert:
                                    type_text(word_insert)
                                # Return to end
                                send_end_key()
                                print(f"    🎯 [Ctrl←×{word_jumps} Ctrl⌫×{word_deletes} +'{word_insert.strip()}' End]")

                        else:
                            # Backspace strategy: delete from end, retype
                            backspace_count = result.get("backspace", 0)
                            text_to_append = result.get("append", "")

                            if backspace_count > 0:
                                send_backspace_windows(backspace_count)
                                print(f"    🔄 [-{backspace_count}]", end="")

                            if text_to_append:
                                type_text(text_to_append)
                                display = text_to_append if len(text_to_append) <= 30 else text_to_append[:27] + "..."
                                print(f" +'{display}'")
                            elif backspace_count > 0:
                                print()  # Newline after correction-only

                    last_transcribe_time = current_time

                    # Trim buffer if getting too large
                    self.streaming_transcriber.trim_buffer()

                except Exception as e:
                    print(f"    ❌ Streaming error: {e}")
                    beep_error()
                    # Stop streaming on error
                    self.stop_event.set()
                    break

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
        print("Press ESC or Ctrl+C to exit.\n")

        # Start hotkey listener
        self.hotkey_handler.start()

        try:
            # Keep main thread alive until exit requested
            while not self.exit_requested:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nCtrl+C - exiting...")
            self._on_exit()


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
