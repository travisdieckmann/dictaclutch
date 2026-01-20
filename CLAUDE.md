# CLAUDE.md - Project Instructions for Claude

## Project Overview

This is a **Voice-to-Text Input Tool** that provides real-time voice transcription using faster-whisper. The application runs locally and supports two modes:

1. **Batch Mode** (Ctrl+Shift+J): Record audio, then transcribe and paste
2. **Streaming Mode** (Ctrl+Shift+K): Real-time transcription as you speak

## Key Files

- `voice_input.py` - Main application with all functionality
- `STREAMING_IMPLEMENTATION_PLAN.md` - Detailed implementation plan and architecture
- `requirements.txt` - Python dependencies
- `setup.bat` / `setup.sh` - Environment setup scripts
- `run.bat` / `run.sh` - Application launch scripts

## Architecture

### Core Classes

- **AudioRecorder**: Batch audio recording (list-based accumulation)
- **StreamingAudioRecorder**: Queue-based audio delivery for streaming
- **Transcriber**: Wrapper around WhisperModel for batch transcription
- **StreamingTranscriber**: Continuous transcription with VAD and buffer management
- **HypothesisBuffer**: LocalAgreement-N policy for stable text detection
- **MultiHotkeyHandler**: Handles multiple global hotkey combinations
- **VoiceInputApp**: Main orchestrator coordinating all components

### Key Design Decisions

| Feature | Implementation |
|---------|---------------|
| Correction visibility | Visible (backspace + retype) |
| Buffer management | Conservative (30s max) |
| Window focus | Pause output on focus loss |
| Error handling | Stop and notify with error beep |
| Word matching | Word-boundary (not character) |
| VAD | faster-whisper built-in (no extra dependency) |

## Development Guidelines

### Running the Application

```bash
# Windows
.\venv\Scripts\activate
python voice_input.py

# Linux
source venv/bin/activate
python voice_input.py
```

### Configuration

All settings are in the `CONFIG` dict at the top of `voice_input.py`:

```python
CONFIG = {
    "hotkey_batch": {...},           # Ctrl+Shift+J
    "hotkey_streaming": {...},       # Ctrl+Shift+K
    "model_size": "small",           # tiny, base, small, medium, large-v3
    "device": "cuda",                # cuda or cpu
    "compute_type": "float16",       # float16, int8, float32
    "streaming_min_chunk": 1.0,      # Seconds before transcription
    "streaming_agreement_n": 2,      # Confirmations needed
    "streaming_buffer_max": 30.0,    # Max buffer seconds
    "streaming_use_vad": True,       # Enable VAD
}
```

### Testing Changes

1. Run syntax check: `python -m py_compile voice_input.py`
2. Test batch mode: Press Ctrl+Shift+J, speak, press again
3. Test streaming mode: Press Ctrl+Shift+K, speak continuously
4. Test corrections: Speak ambiguous phrases, watch for backspace corrections
5. Test window focus: Switch windows while streaming

### Code Style

- No external linting required (plain Python)
- Use type hints where helpful
- Keep functions focused and documented
- Platform-specific code guarded with `sys.platform` checks

## Dependencies

- `faster-whisper` - GPU-optimized Whisper transcription
- `sounddevice` - Audio capture
- `pynput` - Global hotkey detection
- `pyperclip` - Clipboard operations
- `numpy` - Array operations
- `torch` (optional) - CUDA support

## Hardware Requirements

- **Batch Mode**: Works on CPU or GPU
- **Streaming Mode**: GPU highly recommended (NVIDIA 4GB+ VRAM)
- Minimum: 8GB RAM, 4-core CPU

## Known Limitations

1. Whisper doesn't support true incremental decoding
2. Corrections may cause visible flicker in some applications
3. Long recordings require buffer trimming which may lose context
4. Some applications may not handle programmatic backspaces well
