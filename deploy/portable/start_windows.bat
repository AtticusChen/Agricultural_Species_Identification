@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".runtime\Scripts\python.exe" (
  echo [ERROR] Managed Python runtime is missing. Run the one-click installer first.
  exit /b 3
)

".runtime\Scripts\python.exe" launcher.py
exit /b %ERRORLEVEL%
