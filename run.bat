@echo off
setlocal enabledelayedexpansion

REM Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in your PATH.
    echo Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if the virtual environment exists
if not exist venv\Scripts\activate (
    echo Virtual environment not found.
    echo Please run setup.bat first to create the virtual environment.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo.
echo Starting application...

REM Create a simple launcher script
echo import os, sys > launcher.py
echo # Add app_files to Python path >> launcher.py
echo app_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_files") >> launcher.py
echo sys.path.insert(0, app_files_dir) >> launcher.py
echo. >> launcher.py
echo # Import and run the app >> launcher.py
echo from app_files.app import app >> launcher.py
echo from shiny import run_app >> launcher.py
echo. >> launcher.py
echo # Disable warnings >> launcher.py
echo import warnings >> launcher.py
echo warnings.filterwarnings("ignore", module="watchfiles") >> launcher.py
echo. >> launcher.py
echo # Launch the app >> launcher.py
echo run_app(app, reload=False, launch_browser=True, port=8000) >> launcher.py

REM Run the launcher script
python launcher.py
set APP_ERROR=%ERRORLEVEL%

REM Delete the launcher script
del launcher.py

if %APP_ERROR% NEQ 0 (
    echo.
    echo The application exited with an error (code %APP_ERROR%).
    echo.
    echo Possible solutions:
    echo 1. Make sure all dependencies are installed by running setup.bat
    echo 2. Check if port 8000 is already in use by another application
    echo 3. Look for error messages above for more details
    echo.
    pause
)