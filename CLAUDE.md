# CLAUDE.md - Project Instructions for Claude

## Project Overview

**DictaClutch** is a voice-to-text input tool that provides real-time voice transcription using faster-whisper. The application runs locally and supports two modes:

1. **Batch Mode** (Ctrl+Shift+J): Record audio, then transcribe and paste
2. **Streaming Mode** (Ctrl+Shift+K): Real-time transcription as you speak

## Package Structure

```
dictaclutch/
├── pyproject.toml              # Modern Python packaging (PEP 517/518/621)
├── README.md                   # User documentation
├── LICENSE                     # MIT license
├── CLAUDE.md                   # This file
│
├── src/dictaclutch/            # Main package (src layout)
│   ├── __init__.py             # Package exports, lazy imports
│   ├── __main__.py             # CLI entry point (dictaclutch command)
│   ├── config.py               # Configuration (DEFAULT_CONFIG, Config dataclass)
│   │
│   ├── audio/
│   │   ├── recorder.py         # AudioRecorder, StreamingAudioRecorder
│   │   ├── feedback.py         # Beep generation and playback
│   │   └── noise.py            # NoiseReducer (optional noisereduce)
│   │
│   ├── transcription/
│   │   ├── transcriber.py      # Transcriber (batch mode)
│   │   ├── streaming.py        # StreamingTranscriber (real-time)
│   │   └── buffer.py           # IncrementalBuffer (smart diff)
│   │
│   ├── input/
│   │   ├── text_output.py      # Platform-agnostic facade
│   │   ├── keyboard_win.py     # Windows ctypes implementation
│   │   └── keyboard_linux.py   # Linux xdotool implementation
│   │
│   ├── hotkeys/
│   │   ├── handler.py          # MultiHotkeyHandler
│   │   └── diagnostic.py       # HotkeyDiagnostic tool
│   │
│   └── app.py                  # DictaClutchApp orchestrator
│
├── tests/
│   ├── conftest.py             # Shared fixtures
│   └── ...                     # Unit tests
│
├── scripts/
│   ├── run.sh / run.bat        # Run scripts
│   └── setup.sh / setup.bat    # Setup scripts
│
└── voice_input.py              # DEPRECATED shim for backwards compat
```

## Core Classes

| Module | Class | Purpose |
|--------|-------|---------|
| `app.py` | `DictaClutchApp` | Main orchestrator, coordinates all components |
| `audio/recorder.py` | `AudioRecorder` | Batch audio recording (list-based) |
| `audio/recorder.py` | `StreamingAudioRecorder` | Queue-based streaming audio |
| `audio/feedback.py` | - | Beep functions (start/stop/error) |
| `audio/noise.py` | `NoiseReducer` | Optional noise reduction |
| `transcription/transcriber.py` | `Transcriber` | WhisperModel wrapper |
| `transcription/streaming.py` | `StreamingTranscriber` | Real-time transcription |
| `transcription/buffer.py` | `IncrementalBuffer` | Smart diff for corrections |
| `hotkeys/handler.py` | `MultiHotkeyHandler` | Global hotkey detection |
| `hotkeys/diagnostic.py` | `HotkeyDiagnostic` | Hotkey testing tool |
| `config.py` | `Config` | Configuration dataclass |

## Key Design Decisions

| Feature | Implementation |
|---------|---------------|
| Package layout | `src/` layout for proper isolation |
| Dependency injection | All classes accept factory parameters for testing |
| Correction visibility | Visible (backspace + retype) |
| Buffer management | Conservative (30s max) |
| Window focus | Pause output on focus loss |
| Error handling | Stop and notify with error beep |
| Word matching | Word-boundary diffing |
| Hotkey detection | VK code-based (not key object equality) |

## Development Commands

### Running the Application

```bash
# Recommended (after pip install -e .)
dictaclutch

# Or module invocation
python -m dictaclutch

# Legacy (deprecated, shows warning)
python voice_input.py
```

### Hotkey Diagnostic

```bash
dictaclutch --diagnose          # Test default hotkeys
dictaclutch --diagnose -v       # Verbose (all key events)
```

### Testing

```bash
pip install -e ".[dev]"
pytest tests/
pytest tests/ -v --cov=dictaclutch
```

### Type Checking

```bash
mypy src/dictaclutch/
```

## Configuration

Settings are in `src/dictaclutch/config.py`:

```python
DEFAULT_CONFIG = {
    "hotkey_batch": {keyboard.Key.ctrl, keyboard.Key.shift, keyboard.KeyCode.from_char("j")},
    "hotkey_streaming": {keyboard.Key.ctrl, keyboard.Key.shift, keyboard.KeyCode.from_char("k")},
    "model_size": "small",
    "device": "cuda",
    "compute_type": "float16",
    "language": "en",
    "paste_method": "type",
    "streaming_min_chunk": 0.5,
    "streaming_buffer_max": 30.0,
    "noise_reduction_enabled": True,
    "noise_reduction_strength": 0.75,
}
```

## Dependency Injection Pattern

All major classes support dependency injection for testability:

```python
# Example: AudioRecorder accepts stream_factory
class AudioRecorder:
    def __init__(self, stream_factory=None, ...):
        self._stream_factory = stream_factory or sd.InputStream

# In tests:
mock_stream = Mock()
recorder = AudioRecorder(stream_factory=lambda **kw: mock_stream)
```

## Testing Guidelines

1. **Run syntax check:** `python -m py_compile src/dictaclutch/app.py`
2. **Run unit tests:** `pytest tests/`
3. **Test batch mode:** Press Ctrl+Shift+J, speak, press again
4. **Test streaming mode:** Press Ctrl+Shift+K, speak continuously
5. **Test hotkeys:** `dictaclutch --diagnose`
6. **Test window focus:** Switch windows while streaming

## Platform-Specific Code

- **Windows:** Uses ctypes for keyboard simulation (`keyboard_win.py`)
- **Linux:** Uses xdotool subprocess calls (`keyboard_linux.py`)
- **Detection:** `sys.platform` checks in `text_output.py`

## Dependencies

### Required
- `faster-whisper` - GPU-optimized Whisper transcription
- `sounddevice` - Audio capture
- `pynput` - Global hotkey detection
- `pyperclip` - Clipboard operations
- `numpy` - Array operations

### Optional
- `noisereduce` - Noise reduction (`pip install -e ".[noise]"`)
- `torch` - CUDA support (separate install)

### Development
- `pytest`, `pytest-cov`, `pytest-mock` - Testing
- `mypy` - Type checking
- `ruff` - Linting

## Known Limitations

1. Whisper doesn't support true incremental decoding
2. Corrections may cause visible flicker in some applications
3. Long recordings require buffer trimming which may lose context
4. Some applications may not handle programmatic backspaces well
5. VK code detection required due to pynput key object inequality issues
