@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "RUNTIME_DIR=%~dp0runtime"

echo ============================================================
echo Mikania Recognition System - Windows Troubleshooting
echo ============================================================
echo Package folder: "%~dp0"
echo.

if exist "%RUNTIME_DIR%\launcher.py" (
  echo [OK] runtime\launcher.py exists
) else (
  echo [ERROR] runtime\launcher.py is missing - extract the full ZIP first
)

if exist "%RUNTIME_DIR%\requirements.txt" (
  echo [OK] runtime\requirements.txt exists
) else (
  echo [ERROR] runtime\requirements.txt is missing
)

dir /s /b "%~dp0model.onnx" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] model.onnx is missing
) else (
  echo [OK] model.onnx exists
)

where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Windows PowerShell is not available
) else (
  echo [OK] Windows PowerShell is available
)

if exist "%RUNTIME_DIR%\.runtime\Scripts\python.exe" (
  echo [OK] Managed Python runtime is installed
) else (
  echo [INFO] Managed Python runtime is not installed yet
)

echo.
if exist "%RUNTIME_DIR%\logs\install-windows.log" (
  echo ---------------- install-windows.log ----------------
  type "%RUNTIME_DIR%\logs\install-windows.log"
  echo ---------------- end of log -------------------------
) else (
  echo [INFO] No installation log exists yet.
)

echo.
echo Keep this window open and send a screenshot to technical support.
pause
exit /b 0
