# DictaClutch

A lightweight voice-to-text input tool that captures audio with global hotkeys and transcribes it using GPU-accelerated Whisper, then outputs text into the active window.

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub issues](https://img.shields.io/github/issues/travisdieckmann/dictaclutch)](https://github.com/travisdieckmann/dictaclutch/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/travisdieckmann/dictaclutch)](https://github.com/travisdieckmann/dictaclutch/pulls)
[![GitHub](https://img.shields.io/github/license/travisdieckmann/dictaclutch)](/LICENSE)

</div>

## Features

- **Two modes of operation:**
  - **Batch Mode** (Ctrl+Shift+J) - Record, transcribe, paste
  - **Streaming Mode** (Ctrl+Shift+K) - Real-time transcription as you speak
- **GPU-accelerated** transcription using faster-whisper with CUDA
- **Smart corrections** - Streaming mode intelligently corrects itself using word-level diffs
- **Noise reduction** - Optional background noise filtering (fans, AC, hum)
- **Audio feedback** - Distinct beep sequences for start/stop of each mode
- **Cross-platform** - Windows and Linux support
- **Hotkey diagnostic tool** - Built-in tool to validate hotkey detection

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended for streaming mode)
- ~2GB disk space for the Whisper model
- Microphone

## Installation

### Quick Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/dictaclutch.git
cd dictaclutch

# Run the setup script
# Windows:
scripts\setup.bat

# Linux:
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Manual Installation

#### 1. Install CUDA (for GPU acceleration)

**Windows:**
1. Download CUDA Toolkit 12.1+ from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
2. Install with default options
3. Verify: `nvcc --version`

**Linux:**
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit
```

#### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux
source venv/bin/activate
```

#### 3. Install PyTorch with CUDA

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Install DictaClutch

```bash
# With noise reduction support (recommended)
pip install -e ".[noise]"

# For development (includes testing tools)
pip install -e ".[noise,dev]"
```

#### 5. Linux Only: Install xdotool

```bash
sudo apt install xdotool
```

## Usage

### Running the Application

```bash
# After activating your virtual environment
dictaclutch

# Or using Python module
python -m dictaclutch
```

### Hotkey Diagnostic Tool

If hotkeys aren't working as expected, use the built-in diagnostic tool:

```bash
dictaclutch --diagnose

# Verbose mode (shows all key events)
dictaclutch --diagnose --verbose
```

This logs all key events to `hotkey_diagnostic.log` and prints detected hotkey combinations.

### Controls

| Action | Hotkey |
|--------|--------|
| Start/Stop Batch Mode | `Ctrl + Shift + J` |
| Start/Stop Streaming Mode | `Ctrl + Shift + K` |
| Exit | `ESC` or `Ctrl + C` |

### Workflow

#### Batch Mode (Ctrl+Shift+J)
1. Press hotkey to start recording (ascending beep)
2. Speak your message
3. Press hotkey again to stop (descending beep)
4. Text is transcribed and pasted into the active window

#### Streaming Mode (Ctrl+Shift+K)
1. Press hotkey to start (two-tone beep)
2. Speak continuously - text appears in real-time
3. Watch as the system corrects itself using smart word-level diffs
4. Press hotkey again to stop (descending two-tone beep)

## Configuration

Configuration is defined in `src/dictaclutch/config.py`. Key settings:

```python
DEFAULT_CONFIG = {
    # Hotkeys
    "hotkey_batch": {keyboard.Key.ctrl, keyboard.Key.shift, keyboard.KeyCode.from_char("j")},
    "hotkey_streaming": {keyboard.Key.ctrl, keyboard.Key.shift, keyboard.KeyCode.from_char("k")},

    # Model: tiny, base, small, medium, large-v2, large-v3
    "model_size": "small",

    # Device: "cuda" for GPU, "cpu" for CPU-only
    "device": "cuda",
    "compute_type": "float16",

    # Language: "en", "es", "de", etc. (or None for auto-detect)
    "language": "en",

    # Streaming settings
    "streaming_min_chunk": 0.5,      # Seconds between transcriptions
    "streaming_buffer_max": 30.0,    # Maximum audio buffer (seconds)

    # Noise reduction
    "noise_reduction_enabled": True,
    "noise_reduction_strength": 0.75,
}
```

### Model Size Comparison

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | ~1GB | Fastest | Good for clear speech |
| base | ~1GB | Fast | Better |
| small | ~2GB | Balanced | Recommended |
| medium | ~5GB | Slower | High accuracy |
| large-v3 | ~10GB | Slowest | Best accuracy |

For 4GB VRAM GPUs (like RTX 3050 Ti), `small` is recommended.

## Project Structure

```
dictaclutch/
├── pyproject.toml              # Modern Python packaging
├── README.md                   # This file
├── LICENSE                     # MIT license
├── src/dictaclutch/            # Main package
│   ├── __init__.py             # Package exports
│   ├── __main__.py             # CLI entry point
│   ├── config.py               # Configuration
│   ├── app.py                  # Main application
│   ├── audio/                  # Audio handling
│   │   ├── recorder.py         # Audio recording
│   │   ├── feedback.py         # Beep generation
│   │   └── noise.py            # Noise reduction
│   ├── transcription/          # Transcription
│   │   ├── transcriber.py      # Batch transcription
│   │   ├── streaming.py        # Streaming transcription
│   │   └── buffer.py           # Incremental buffer
│   ├── input/                  # Text output
│   │   ├── text_output.py      # Platform facade
│   │   ├── keyboard_win.py     # Windows implementation
│   │   └── keyboard_linux.py   # Linux implementation
│   └── hotkeys/                # Hotkey handling
│       ├── handler.py          # Multi-hotkey handler
│       └── diagnostic.py       # Diagnostic tool
├── tests/                      # Unit tests
└── scripts/                    # Shell scripts
    ├── run.sh / run.bat
    └── setup.sh / setup.bat
```

## Troubleshooting

### "CUDA not available"
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Ensure PyTorch was installed with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Hotkeys don't work
- Run `dictaclutch --diagnose` to test hotkey detection
- Some apps intercept global hotkeys - try a different combination
- On Linux, ensure you have permissions for `/dev/input` devices:
  ```bash
  sudo usermod -aG input $USER
  # Log out and back in
  ```

### No audio detected
- Check your default microphone in system settings
- Run: `python -c "import sounddevice; print(sounddevice.query_devices())"`

### Text doesn't appear in application
- Try changing `paste_method` to `"clipboard"` in config
- On Linux, ensure `xdotool` is installed

### Noise reduction not working
- Install the noise reduction dependency: `pip install noisereduce`
- Or reinstall with: `pip install -e ".[noise]"`

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Style

```bash
# Type checking
mypy src/dictaclutch/

# Linting
ruff check src/dictaclutch/
```

## License

MIT - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Acknowledgments

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Fast Whisper implementation
- [pynput](https://github.com/moses-palmer/pynput) - Global hotkey detection
- [noisereduce](https://github.com/timsainb/noisereduce) - Noise reduction
