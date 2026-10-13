@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo Mikania Recognition System - One-click Installation
echo ============================================================
echo No preinstalled Python or administrator account is required.
echo The first installation needs internet access and may take 5-15 minutes.
echo.

where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Windows PowerShell is not available on this computer.
  exit /b 4
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo [ERROR] Installation failed. See logs\install-windows.log.
  exit /b %EXIT_CODE%
)

echo.
echo [OK] The managed runtime and model passed the self-test.
exit /b 0
