@echo off
setlocal
set "PYTHONW=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%PYTHONW%" (
  echo Dependencies are not installed. Run "Установить зависимости.cmd" first.
  pause
  exit /b 1
)
if not exist "%~dp0local_chat.pyw" (
  echo Chat app was not found next to this file.
  pause
  exit /b 1
)
start "" "%PYTHONW%" "%~dp0local_chat.pyw"
exit /b 0
