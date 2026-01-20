# Voice-to-Text Input Tool

A lightweight tool that captures audio with a global hotkey and transcribes it using GPU-accelerated Whisper, then pastes the text into the active window (like Claude Desktop).

## Features

- **Global hotkey** (Ctrl+Shift+Space) - works from any application
- **GPU-accelerated** transcription using faster-whisper with CUDA
- **Sub-second transcription** for typical voice clips
- **Auto-paste** into the active window
- **Audio feedback** (beeps on start/stop)
- **Cross-platform** (Windows and Linux)

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (for GPU acceleration)
- ~2GB disk space for the Whisper model
- Microphone

## Installation

### 1. Install CUDA (if using GPU)

**Windows:**
1. Download CUDA Toolkit 12.1+ from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
2. Install with default options
3. Verify: `nvcc --version`

**Linux:**
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit
```

### 2. Create Virtual Environment

```bash
# Create project directory
cd voice-to-claude

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux
source venv/bin/activate
```

### 3. Install PyTorch with CUDA

```bash
# For CUDA 12.1 (recommended for RTX 30/40 series)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 5. Linux Only: Install xdotool

```bash
sudo apt install xdotool
```

## Usage

```bash
# Activate virtual environment first
# Windows: venv\Scripts\activate
# Linux: source venv/bin/activate

python voice_input.py
```

On first run, it will download the Whisper model (~1GB for "small").

### Controls

| Action | Hotkey |
|--------|--------|
| Start recording | `Ctrl + Shift + Space` |
| Stop recording & transcribe | `Ctrl + Shift + Space` |
| Exit | `Ctrl + C` (in terminal) |

### Workflow

1. Start the tool (`python voice_input.py`)
2. Click into Claude Desktop (or any text field)
3. Press `Ctrl+Shift+Space` to start recording (ascending beep)
4. Speak your message
5. Press `Ctrl+Shift+Space` to stop (descending beep)
6. Text is automatically pasted into the active window

## Configuration

Edit the `CONFIG` dictionary at the top of `voice_input.py`:

```python
CONFIG = {
    # Change hotkey (example: Ctrl+Alt+R)
    "hotkey": {keyboard.Key.ctrl, keyboard.Key.alt, keyboard.KeyCode.from_char('r')},
    
    # Model size: tiny (fastest), base, small (recommended), medium, large-v3 (most accurate)
    "model_size": "small",
    
    # Use CPU instead of GPU
    "device": "cpu",
    "compute_type": "int8",  # Use int8 for CPU
    
    # Language: None for auto-detect, or specific like "en", "es", "de"
    "language": "en",
    
    # Disable beeps
    "beep_on_start": False,
    "beep_on_stop": False,
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

For your RTX 3050 Ti (4GB VRAM), `small` is the sweet spot.

## Troubleshooting

### "CUDA not available"
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Ensure PyTorch was installed with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Hotkey doesn't work
- Some apps intercept global hotkeys. Try a different combination.
- On Linux, ensure you have permissions for `/dev/input` devices:
  ```bash
  sudo usermod -aG input $USER
  # Log out and back in
  ```

### No audio detected
- Check your default microphone in system settings
- Run: `python -c "import sounddevice; print(sounddevice.query_devices())"`

### Paste doesn't work
- Try changing `paste_method` to `"type"` in config (slower but more compatible)
- On Linux, ensure `xdotool` is installed

## Running on Startup (Optional)

### Windows
1. Create a shortcut to `voice_input.py`
2. Press `Win+R`, type `shell:startup`, press Enter
3. Move the shortcut to the Startup folder

### Linux (systemd)
```bash
# Create service file
cat > ~/.config/systemd/user/voice-input.service << EOF
[Unit]
Description=Voice to Text Input Tool

[Service]
ExecStart=/path/to/venv/bin/python /path/to/voice_input.py
Restart=on-failure

[Install]
WantedBy=default.target
EOF

# Enable and start
systemctl --user enable voice-input.service
systemctl --user start voice-input.service
```

## License

MIT - Use freely!
