@echo off
setlocal
set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not exist "%PYTHON%" (
  echo Python was not found: %PYTHON%
  pause
  exit /b 1
)
if not exist "%~dp0.venv\Scripts\python.exe" "%PYTHON%" -m venv "%~dp0.venv"
if errorlevel 1 pause & exit /b 1
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 pause & exit /b 1
echo Dependencies installed.
pause
