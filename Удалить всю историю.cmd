@echo off
setlocal
set "PYTHONW=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%PYTHONW%" (
  echo Dependencies are not installed. Run "Установить зависимости.cmd" first.
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
