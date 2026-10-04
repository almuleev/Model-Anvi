@echo off
setlocal
set "PYTHONW=%LOCALAPPDATA%\Programs\Python\Python310\pythonw.exe"
if not exist "%PYTHONW%" (
  echo Python was not found: %PYTHONW%
  pause
  exit /b 1
)
if not exist "%~dp0clear_history.pyw" (
  echo History cleanup script was not found next to this file.
  pause
  exit /b 1
)
start "" "%PYTHONW%" "%~dp0clear_history.pyw"
exit /b 0
