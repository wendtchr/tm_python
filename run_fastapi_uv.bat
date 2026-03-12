@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo Checking for uv...
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo uv is not installed or not in PATH.
    echo Install uv from https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

if not exist .venv312\Scripts\python.exe (
    echo Creating Python 3.12 virtual environment at .venv312...
    uv venv .venv312 --python 3.12
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create .venv312 environment.
        pause
        exit /b 1
    )
)

echo Installing/updating dependencies...
uv pip install --python .venv312\Scripts\python.exe -r fastapi_app/requirements.txt -r app_files/requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Starting FastAPI app on http://127.0.0.1:8000 ...
echo Press Ctrl+C to stop.
echo.

.venv312\Scripts\python.exe -m uvicorn fastapi_app.main:app --reload --reload-dir fastapi_app --reload-dir app_files --host 127.0.0.1 --port 8000
set APP_ERROR=%ERRORLEVEL%

if %APP_ERROR% NEQ 0 (
    echo.
    echo The FastAPI app exited with error code %APP_ERROR%.
    echo Common fixes:
    echo 1. Ensure port 8000 is not already in use
    echo 2. Re-run this script to refresh dependencies
    echo 3. Check traceback above for module errors
    echo.
    pause
)

exit /b %APP_ERROR%
