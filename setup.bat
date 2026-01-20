@echo off
echo ============================================================
echo Voice-to-Text Input Tool - Setup
echo ============================================================
echo.

echo Checking for Python...
call python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+ from python.org
    pause
    exit /b 1
)
echo Python found!
echo.

echo [1/4] Creating virtual environment...
call python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo [2/4] Installing PyTorch with CUDA 12.1 support...
call venv\Scripts\activate.bat && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo [3/4] Verifying CUDA...
call venv\Scripts\activate.bat && call python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo [4/4] Installing other dependencies...
call venv\Scripts\activate.bat && pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Setup complete!
echo.
echo To run the tool:
echo   1. Open a terminal in this folder
echo   2. Run: venv\Scripts\activate
echo   3. Run: python voice_input.py
echo.
echo Or just double-click: run.bat
echo ============================================================
pause
