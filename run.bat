@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
call python voice_input.py
pause
