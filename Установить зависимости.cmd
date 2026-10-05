@echo off
setlocal
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" goto install

where py >nul 2>&1
if not errorlevel 1 (
  py -3 -c "import sys, tkinter; assert sys.version_info >= (3, 10)" >nul 2>&1
  if not errorlevel 1 (
    py -3 -m venv "%~dp0.venv"
    goto check_venv
  )
)

python -c "import sys, tkinter; assert sys.version_info >= (3, 10)" >nul 2>&1
if errorlevel 1 goto python_missing
python -m venv "%~dp0.venv"

:check_venv
if errorlevel 1 goto failed
if not exist "%VENV_PYTHON%" goto failed

:install
"%VENV_PYTHON%" -c "import sys, tkinter; assert sys.version_info >= (3, 10)"
if errorlevel 1 goto python_missing
"%VENV_PYTHON%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto failed
echo Model Anvi dependencies installed.
if /i not "%~1"=="--no-pause" pause
exit /b 0

:python_missing
echo Install Python 3.10 or newer with Tcl/Tk and add Python to PATH.
echo Download: https://www.python.org/downloads/windows/
if /i not "%~1"=="--no-pause" pause
exit /b 1

:failed
echo Dependency setup failed. See the error above.
if /i not "%~1"=="--no-pause" pause
exit /b 1
