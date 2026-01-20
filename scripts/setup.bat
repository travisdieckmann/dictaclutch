@echo off
echo ============================================================
echo DictaClutch - Setup
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

cd /d "%~dp0\.."

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
echo [4/4] Installing DictaClutch...
call venv\Scripts\activate.bat && pip install -e ".[noise,dev]"
if errorlevel 1 (
    echo ERROR: Failed to install DictaClutch
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Setup complete!
echo.
echo To run DictaClutch:
echo   1. Open a terminal in this folder
echo   2. Run: venv\Scripts\activate
echo   3. Run: dictaclutch
echo.
echo Or just double-click: scripts\run.bat
echo.
echo To test hotkey detection:
echo   dictaclutch --diagnose
echo ============================================================
pause
