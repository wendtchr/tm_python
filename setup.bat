@echo off
echo Creating virtual environment...
python -m venv venv
echo.
echo Activating virtual environment...
call venv\Scripts\activate
echo.
echo Installing dependencies...

REM Install dependencies with exact versions for critical packages
pip install -r app_files/requirements.txt

REM Verify Shiny version
python -c "import shiny; print(f'Installed Shiny version: {shiny.__version__}')"

echo.
echo Setup complete! Run the application with run.bat 