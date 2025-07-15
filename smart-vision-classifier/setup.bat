@echo off
echo.
echo ========================================
echo   Smart Vision Classifier - Windows
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found!
echo.

REM Run setup
echo Running setup...
python setup.py

echo.
echo Setup complete! You can now run:
echo   - python main.py (start web app)
echo   - python demo.py (run demo)
echo.
pause
