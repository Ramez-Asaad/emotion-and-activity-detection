@echo off
REM Activate virtual environment and run custom_cnn.py

echo Activating virtual environment...
call "venv\Scripts\activate.bat"

echo.
echo Running custom_cnn.py with GPU support...
echo.

python custom_cnn.py

pause
