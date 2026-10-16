@echo off
title Driver Behaviour Analysis - Web Demo
echo =====================================================================
echo   🚗 Driver Behaviour Analysis - Real-Time Perception System
echo =====================================================================
echo.
cd /d "%~dp0"

REM Activate virtual environment
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment .venv not found!
    pause
    exit /b 1
)

echo [INFO] Starting Gradio application on http://127.0.0.1:7860 ...
echo [INFO] Your browser will open automatically in 5 seconds.
echo.

REM Open browser automatically
start "" http://127.0.0.1:7860

REM Run the main application
python app_demo.py

pause
