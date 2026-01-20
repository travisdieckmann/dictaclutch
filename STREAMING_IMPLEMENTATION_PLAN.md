# Implementation Plan: Dual-Mode Voice Transcription

## Overview

Add two distinct transcription modes to voice_input.py:
1. **Batch Mode** (Current): Toggle record → transcribe complete audio → paste result
2. **Streaming Mode** (New): Continuous recording → real-time partial results → handle corrections

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Correction visibility | **Visible** | Backspace + retype for lower latency |
| Buffer management | **Conservative (30s)** | Better context for corrections |
| Window focus | **Pause on focus loss** | Prevent typing into wrong window |
| Error handling | **Stop and notify** | Clean failure, user restarts manually |
| Word matching | **Word-boundary** | Avoid jarring mid-word corrections |
| VAD source | **faster-whisper built-in** | No extra dependency needed |

---

## Mode Configuration

### Hotkey Assignments

```python
CONFIG = {
    # Batch mode (existing behavior)
    "hotkey_batch": {keyboard.Key.ctrl, keyboard.Key.shift, keyboard.KeyCode.from_char("j")},

    # Streaming mode (new)
    "hotkey_streaming": {keyboard.Key.ctrl, keyboard.Key.shift, keyboard.KeyCode.from_char("k")},

    # Streaming settings
    "streaming_min_chunk": 1.0,        # Seconds of audio before transcription attempt
    "streaming_agreement_n": 2,         # Number of consecutive agreements for confirmation
    "streaming_buffer_max": 30.0,       # Maximum buffer size (seconds) before trimming
    "streaming_use_vad": True,          # Use Voice Activity Detection
}
```

---

## Architecture Design

### Class Structure

```
VoiceInputApp
├─ AudioRecorder (existing - batch mode)
├─ StreamingAudioRecorder (NEW - queue-based delivery)
├─ Transcriber (existing - batch mode)
├─ StreamingTranscriber (NEW)
│   ├─ HypothesisBuffer (NEW - word-boundary matching)
│   └─ VAD (from faster_whisper.vad)
├─ MultiHotkeyHandler (NEW - replaces HotkeyHandler)
├─ mode: "batch" | "streaming" | None
└─ target_window: int (for focus tracking)
```

### New Classes Required

#### 1. HypothesisBuffer
Tracks transcription hypotheses using **word-boundary matching** to detect stable text.

**Key difference from original plan:** Uses word-level comparison instead of character-level to avoid mid-word corrections.

```python
class HypothesisBuffer:
    def __init__(self, agreement_n: int = 2):
        self.history: list[str] = []           # Last N transcriptions
        self.committed_words: list[str] = []   # Confirmed stable words
        self.typed_char_count: int = 0         # Actual chars sent to keyboard
        self.agreement_n: int = agreement_n

    def update(self, new_transcript: str) -> dict:
        """
        Add new transcript and determine what's stable using word-boundary matching.
        Returns: {
            "text_to_type": "new words to type",
            "correction": {
                "backspace_count": 5,
                "replacement": "corrected words"
            } or None
        }
        """
        self.history.append(new_transcript)
        if len(self.history) > self.agreement_n + 1:
            self.history.pop(0)

        if len(self.history) >= self.agreement_n:
            # Find common word prefix across last N transcripts
            common_words = self._find_common_word_prefix(self.history[-self.agreement_n:])

            if len(common_words) > len(self.committed_words):
                # New confirmed words
                new_words = common_words[len(self.committed_words):]
                text_to_type = " ".join(new_words)
                if self.committed_words:  # Add leading space if not first words
                    text_to_type = " " + text_to_type
                self.committed_words = common_words
                self.typed_char_count += len(text_to_type)
                return {"text_to_type": text_to_type, "correction": None}

            elif len(common_words) < len(self.committed_words):
                # CORRECTION: Previous words were wrong
                wrong_words = self.committed_words[len(common_words):]
                wrong_text = " ".join(wrong_words)
                if common_words:  # Account for space before wrong words
                    wrong_text = " " + wrong_text

                backspace_count = len(wrong_text)
                self.committed_words = common_words
                self.typed_char_count -= backspace_count

                return {
                    "text_to_type": "",
                    "correction": {
                        "backspace_count": backspace_count,
                        "replacement": ""
                    }
                }

        return {"text_to_type": "", "correction": None}

    def _find_common_word_prefix(self, transcripts: list[str]) -> list[str]:
        """Find longest common prefix at word boundaries."""
        words_lists = [t.split() for t in transcripts]
        if not words_lists or not all(words_lists):
            return []

        common_words = []
        for word_tuple in zip(*words_lists):
            if len(set(word_tuple)) == 1:
                common_words.append(word_tuple[0])
            else:
                break
        return common_words

    def reset(self):
        """Clear all buffers (on recording start/stop)"""
        self.history.clear()
        self.committed_words.clear()
        self.typed_char_count = 0
```

#### 2. StreamingAudioRecorder
Queue-based audio delivery for real-time streaming (separate from batch AudioRecorder).

```python
class StreamingAudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_queue = queue.Queue()  # Thread-safe delivery
        self.stream = None
        self.recording = False

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        if self.recording:
            self.chunk_queue.put(indata.copy())

    def start(self):
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
```

#### 3. StreamingTranscriber
Manages continuous audio buffering and periodic transcription.

```python
class StreamingTranscriber:
    def __init__(self, model: WhisperModel, config: dict):
        self.model = model
        self.min_chunk_size = config["streaming_min_chunk"]
        self.max_buffer_size = config["streaming_buffer_max"]
        self.use_vad = config["streaming_use_vad"]

        self.audio_buffer = np.array([], dtype=np.float32)
        self.hypothesis_buffer = HypothesisBuffer(config["streaming_agreement_n"])
        self.last_transcribe_time = 0

        # VAD from faster-whisper (no extra dependency!)
        if self.use_vad:
            from faster_whisper.vad import get_speech_timestamps, VadOptions
            self.get_speech_timestamps = get_speech_timestamps
            self.vad_options = VadOptions()

    def add_chunk(self, chunk: np.ndarray):
        """Add audio chunk to rolling buffer."""
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk.flatten()])

    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds."""
        return len(self.audio_buffer) / 16000  # Assuming 16kHz

    def should_transcribe(self) -> bool:
        """Check if we should transcribe now."""
        duration = self.get_buffer_duration()

        if duration < self.min_chunk_size:
            return False

        if self.use_vad:
            # Check for speech end (silence detected)
            try:
                timestamps = self.get_speech_timestamps(self.audio_buffer, self.vad_options)
                if timestamps:
                    last_end = timestamps[-1]["end"] / 16000
                    buffer_end = duration
                    # If last speech ended > 0.5s ago, transcribe now
                    if buffer_end - last_end > 0.5:
                        return True
            except Exception:
                pass  # Fall back to time-based

            # Force transcribe if buffer too large
            if duration > self.max_buffer_size:
                return True

            return False

        # No VAD - transcribe at fixed intervals
        return duration >= self.min_chunk_size

    def transcribe_buffer(self) -> dict:
        """
        Transcribe current buffer and update hypothesis.
        Returns result from HypothesisBuffer.update()
        """
        if len(self.audio_buffer) == 0:
            return {"text_to_type": "", "correction": None}

        # Transcribe
        segments, _ = self.model.transcribe(
            self.audio_buffer,
            language="en",
            beam_size=5,
            vad_filter=False,  # We handle VAD ourselves
        )

        transcript = " ".join(seg.text for seg in segments).strip()
        self.last_transcribe_time = time.time()

        # Update hypothesis buffer
        return self.hypothesis_buffer.update(transcript)

    def trim_buffer(self):
        """Trim buffer to max size, keeping recent audio."""
        max_samples = int(self.max_buffer_size * 16000)
        if len(self.audio_buffer) > max_samples:
            # Keep the most recent audio
            self.audio_buffer = self.audio_buffer[-max_samples:]

    def reset(self):
        """Clear all buffers."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.hypothesis_buffer.reset()
        self.last_transcribe_time = 0
```

#### 4. MultiHotkeyHandler
Supports multiple hotkey combinations with separate callbacks.

```python
class MultiHotkeyHandler:
    def __init__(self, hotkey_configs: dict):
        """
        hotkey_configs = {
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

    def _is_hotkey_pressed(self, mode: str) -> bool:
        """Check if specific hotkey combo is pressed."""
        # Similar logic to existing HotkeyHandler but generalized
        required = self.hotkey_configs[mode]

        has_ctrl = any(k in [keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r]
                       for k in self.current_keys)
        has_shift = any(k in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]
                        for k in self.current_keys)

        # Get the letter key from the config
        letter_key = None
        for k in required:
            if hasattr(k, 'char'):
                letter_key = k.char.lower()
                break

        has_letter = any(
            (hasattr(k, 'char') and k.char and k.char.lower() == letter_key)
            for k in self.current_keys
        ) if letter_key else False

        return has_ctrl and has_shift and has_letter

    def _on_press(self, key):
        self.current_keys.add(key)

        for mode in self.hotkey_configs:
            if not self.triggered[mode] and self._is_hotkey_pressed(mode):
                self.triggered[mode] = True
                if mode in self.callbacks:
                    self.callbacks[mode]()

    def _on_release(self, key):
        self.current_keys.discard(key)

        for mode in self.hotkey_configs:
            if not self._is_hotkey_pressed(mode):
                self.triggered[mode] = False

    def start(self):
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.start()

    def stop(self):
        if self.listener:
            self.listener.stop()
```

---

## Window Focus Detection

Track which window was active when streaming started and pause output if focus changes.

```python
def get_foreground_window() -> int:
    """Get handle of currently focused window (Windows only)."""
    return ctypes.windll.user32.GetForegroundWindow()

# In VoiceInputApp:
def start_streaming(self):
    self.target_window = get_foreground_window()
    # ... start streaming ...

def should_output(self) -> bool:
    """Check if we should type (target window still focused)."""
    return get_foreground_window() == self.target_window
```

---

## Backspace Functionality

```python
def send_backspace_windows(count: int):
    """Send backspace key N times to delete characters."""
    VK_BACK = 0x08
    KEYEVENTF_KEYUP = 0x0002
    user32 = ctypes.windll.user32

    for _ in range(count):
        user32.keybd_event(VK_BACK, 0, 0, 0)
        user32.keybd_event(VK_BACK, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.01)

def apply_correction(correction: dict):
    """Apply correction by backspacing and typing replacement."""
    if correction:
        send_backspace_windows(correction["backspace_count"])
        if correction["replacement"]:
            type_text(correction["replacement"])
```

---

## Implementation Phases

### Phase 1: Dual Hotkey Foundation

**Goal:** Two separate hotkeys trigger two different recording modes.

**Changes:**

1. Add streaming config keys to `CONFIG`
2. Implement `MultiHotkeyHandler` class
3. Add `get_foreground_window()` function
4. Update `VoiceInputApp`:
   - Add `mode` state tracking ("batch" | "streaming" | None)
   - Add `target_window` for focus tracking
   - Register both hotkeys with callbacks
   - Implement mutual exclusion between modes

**Deliverable:** User can press Ctrl+Shift+J for batch mode, Ctrl+Shift+K for streaming mode (placeholder message).

---

### Phase 2: Basic Streaming Without Corrections

**Goal:** Streaming mode records continuously and types partial results every N seconds.

**Changes:**

1. Implement `StreamingAudioRecorder` with queue-based delivery
2. Implement `StreamingTranscriber` (without HypothesisBuffer):
   - Rolling audio buffer
   - Transcribe every N seconds
   - Type incremental text (naively append new words)
3. Streaming worker thread with `threading.Event` for clean shutdown:
   ```python
   self.stop_event = threading.Event()

   def streaming_worker():
       while not self.stop_event.is_set():
           # Get chunks from queue
           # Add to buffer
           # Transcribe if ready
           # Type new text (if window focused)
   ```
4. Window focus checking before typing

**Limitations at this phase:**
- No correction handling - may type duplicate words
- Text only grows, never corrects

**Deliverable:** Working streaming mode that types words as they're spoken (with rough edges).

---

### Phase 3: LocalAgreement & Correction Handling

**Goal:** Implement intelligent correction detection using word-boundary matching.

**Changes:**

1. Implement `HypothesisBuffer` with word-level matching:
   - `_find_common_word_prefix()` for stable word detection
   - Track `typed_char_count` for accurate backspace counts
   - Return corrections at word boundaries only
2. Add `send_backspace_windows()` function
3. Add `apply_correction()` function
4. Integrate hypothesis buffer into `StreamingTranscriber`
5. Error handling: catch transcription errors, stop streaming, play error beep

**Deliverable:** Streaming mode that types words and backspaces/corrects when Whisper changes its mind.

---

### Phase 4: VAD Integration & Optimization

**Goal:** Use Voice Activity Detection to minimize latency and unnecessary transcriptions.

**Changes:**

1. Use faster-whisper's built-in VAD:
   ```python
   from faster_whisper.vad import get_speech_timestamps, VadOptions
   ```
2. Enhanced `should_transcribe()` logic:
   - Detect speech end (silence > 0.5s) to trigger transcription
   - Force transcribe if buffer exceeds max size
3. Smart buffer trimming:
   - Keep last 30 seconds (conservative)
   - Trim at word boundaries when possible
4. Memory optimization for long sessions

**Deliverable:** Streaming mode that transcribes at natural sentence boundaries, reducing latency and corrections.

---

## Testing Strategy

### Test Cases

1. **Batch Mode (Regression Testing):**
   - Press Ctrl+Shift+J → speak → press again → verify text pastes correctly
   - Test with punctuation (period should add space)
   - Test very short audio (< 0.3s) → should ignore
   - Test while processing → should ignore new trigger

2. **Streaming Mode - Basic:**
   - Press Ctrl+Shift+K → speak continuously → verify text appears incrementally
   - Press Ctrl+Shift+K again → verify recording stops
   - Verify can't start batch while streaming active (mutual exclusion)

3. **Streaming Mode - Corrections:**
   - Speak ambiguous phrase (e.g., "their" vs "there")
   - Observe if correction happens (backspace + retype)
   - Count correction frequency (should be low with agreement_n=2)

4. **Streaming Mode - Window Focus:**
   - Start streaming → switch windows → verify typing pauses
   - Switch back → verify typing resumes

5. **Streaming Mode - VAD:**
   - Speak sentence → pause → verify transcription triggers on pause
   - Speak without pauses → verify doesn't wait too long
   - Test in noisy environment → verify VAD filters noise

6. **Streaming Mode - Errors:**
   - Simulate GPU error → verify clean shutdown with error beep

7. **Mode Switching:**
   - Start batch → press streaming hotkey → verify batch cancels
   - Start streaming → press batch hotkey → verify streaming stops
   - Rapid mode switching → verify no crashes

### Performance Benchmarks

Track these metrics:

- **Batch mode:**
  - Transcription time for 5s, 10s, 30s audio
  - Accuracy on test phrases

- **Streaming mode:**
  - Latency (time from speech to text appearance)
  - Correction frequency (backspaces per minute)
  - CPU/GPU usage
  - Buffer memory usage over 5-minute session

---

## Configuration Tuning Guide

### For Low Latency (Fastest Response)

```python
"streaming_min_chunk": 0.5,      # Transcribe after just 0.5s
"streaming_agreement_n": 1,      # Don't wait for confirmation
"streaming_use_vad": True,       # Detect speech end quickly
```

**Trade-off:** More corrections, potential flicker

### For High Accuracy (Fewest Corrections)

```python
"streaming_min_chunk": 2.0,      # Wait for more context
"streaming_agreement_n": 3,      # Need 3 confirmations
"streaming_use_vad": True,       # Wait for sentence boundaries
```

**Trade-off:** Higher latency (1-3s delay)

### Balanced (Recommended)

```python
"streaming_min_chunk": 1.0,
"streaming_agreement_n": 2,
"streaming_use_vad": True,
```

---

## Known Limitations & Future Enhancements

### Limitations

1. **Reprocessing Overhead:**
   - Each transcription processes entire buffer from start
   - Same audio transcribed multiple times
   - GPU must recompute for overlapping audio

2. **No True Incremental Decoding:**
   - Whisper architecture doesn't support incremental decoding
   - Must process full audio context each time
   - Cannot pause/resume decoder state

3. **Correction Flicker:**
   - Backspace + retype is visible to user
   - Can be distracting in some applications
   - Some apps may not handle backspace correctly

4. **Buffer Growth:**
   - Long monologues (5+ min) cause buffer to grow
   - Must trim carefully to avoid splitting words
   - Trimming too aggressively loses context

### Future Enhancements

1. **Multi-Language Support:**
   - Auto-detect language in streaming mode
   - Switch models on the fly

2. **Punctuation Prediction:**
   - Add commas, periods, capitalization in real-time
   - Use separate punctuation model (FullStop, etc.)

3. **Command Mode:**
   - Voice commands ("new paragraph", "delete last sentence")
   - Control paste_method via voice

4. **Visual Feedback:**
   - Show transcript in overlay window
   - Distinguish committed vs tentative text with colors
   - Display confidence scores

5. **Multiple Hotkeys:**
   - Different keys for different languages
   - Per-application hotkey profiles

---

## Dependencies

### Required Packages

```bash
# Existing (no changes)
pip install faster-whisper
pip install sounddevice
pip install pynput
pip install pyperclip
pip install numpy

# NOTE: silero-vad is NOT needed - faster-whisper includes it!
# torch is already installed for CUDA support
```

### Hardware Requirements

- **Batch Mode:** Works on CPU or GPU
- **Streaming Mode:**
  - **GPU highly recommended** (NVIDIA with 4GB+ VRAM)
  - CPU streaming will be too slow (>5s latency)
  - Minimum: 8GB RAM, 4-core CPU

---

## Implementation Checklist

- [x] Phase 1: Multi-hotkey support ✅ COMPLETED
  - [x] Update CONFIG with dual hotkeys
  - [x] Implement MultiHotkeyHandler
  - [x] Add get_foreground_window()
  - [x] Update VoiceInputApp with mode tracking
  - [x] Test: Both hotkeys print different messages

- [ ] Phase 2: Basic streaming (no corrections)
  - [ ] Implement StreamingAudioRecorder (queue-based)
  - [ ] Implement StreamingTranscriber (simple version)
  - [ ] Add streaming worker thread with Event shutdown
  - [ ] Add window focus checking
  - [ ] Test: Words appear as spoken (duplicates OK)

- [ ] Phase 3: LocalAgreement & corrections
  - [ ] Implement HypothesisBuffer with word-boundary matching
  - [ ] Add send_backspace_windows()
  - [ ] Add apply_correction()
  - [ ] Integrate hypothesis buffer into streaming
  - [ ] Add error handling (stop + error beep)
  - [ ] Test: Corrections handled gracefully

- [ ] Phase 4: VAD & optimization
  - [ ] Use faster_whisper.vad (built-in)
  - [ ] Implement smart buffer trimming (30s max)
  - [ ] Add sentence boundary detection
  - [ ] Test: Latency improved, corrections reduced

- [ ] Testing & Documentation
  - [ ] Run all test cases
  - [ ] Benchmark performance
  - [ ] Update README with new features
  - [ ] Add configuration guide

---

## References

- [whisper_streaming](https://github.com/ufal/whisper_streaming) - Production-ready streaming implementation
- [WhisperLive](https://github.com/collabora/WhisperLive) - Alternative streaming approach
- [Silero VAD](https://github.com/snakers4/silero-vad) - Fast voice activity detection
- [LocalAgreement Policy Paper](https://aclanthology.org/2023.ijcnlp-demo.3.pdf) - Theoretical foundation

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-20 | Use LocalAgreement-N policy | Proven effective in whisper_streaming |
| 2026-01-20 | agreement_n=2 as default | Balance between latency and accuracy |
| 2026-01-20 | Separate classes for batch/streaming | Keeps code maintainable |
| 2026-01-20 | Optional VAD integration | Allows CPU-only users to skip dependency |
| 2026-01-20 | Backspace-based corrections | Most compatible across applications |
| 2026-01-20 | Word-boundary matching | Avoids jarring mid-word corrections |
| 2026-01-20 | Use faster-whisper built-in VAD | No extra dependency needed |
| 2026-01-20 | Queue-based audio delivery | Thread-safe for streaming |
| 2026-01-20 | Track typed_char_count | Accurate backspace counts |
| 2026-01-20 | Pause on window focus loss | Prevent typing into wrong window |
| 2026-01-20 | Stop and notify on errors | Clean failure, user restarts manually |
| 2026-01-20 | Conservative 30s buffer | Better context for corrections |
