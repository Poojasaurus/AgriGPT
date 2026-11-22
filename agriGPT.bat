@echo off
title agriGPT
cd /d "%~dp0"

:: Check if virtual environment exists
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" desktop_app.py
) else if exist "..\venv\Scripts\python.exe" (
    "..\venv\Scripts\python.exe" desktop_app.py
) else (
    python desktop_app.py
)
