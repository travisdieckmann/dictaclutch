#!/bin/bash
set -e

echo "============================================================"
echo "DictaClutch - Setup"
echo "============================================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.10+"
    exit 1
fi

cd "$(dirname "$0")/.."

echo "[1/5] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo
echo "[2/5] Installing PyTorch with CUDA 12.1 support..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

echo
echo "[3/5] Verifying CUDA..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo
echo "[4/5] Installing DictaClutch..."
pip install -e ".[noise,dev]"

echo
echo "[5/5] Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    echo "Installing xdotool (requires sudo)..."
    sudo apt-get install -y xdotool
elif command -v dnf &> /dev/null; then
    echo "Installing xdotool (requires sudo)..."
    sudo dnf install -y xdotool
elif command -v pacman &> /dev/null; then
    echo "Installing xdotool (requires sudo)..."
    sudo pacman -S --noconfirm xdotool
else
    echo "WARNING: Could not detect package manager. Please install xdotool manually."
fi

echo
echo "============================================================"
echo "Setup complete!"
echo
echo "To run DictaClutch:"
echo "  source venv/bin/activate"
echo "  dictaclutch"
echo
echo "Or: ./scripts/run.sh"
echo
echo "To test hotkey detection:"
echo "  dictaclutch --diagnose"
echo "============================================================"
