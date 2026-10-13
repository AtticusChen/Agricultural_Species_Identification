@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "RUNTIME_DIR=%~dp0runtime"

if not exist "%RUNTIME_DIR%\start_windows.bat" (
  echo.
  echo [ERROR] Required runtime files are missing.
  echo Right-click the ZIP file, choose Extract All, and run this file from the extracted folder.
  echo.
  pause
  exit /b 2
)

if not exist "%RUNTIME_DIR%\.runtime\Scripts\python.exe" (
  echo The runtime is not installed. Starting one-click installation now.
  call "%RUNTIME_DIR%\install_windows.bat"
)

if not exist "%RUNTIME_DIR%\.runtime\Scripts\python.exe" (
  echo.
  echo [ERROR] The runtime was not installed successfully.
  echo Log file: "%RUNTIME_DIR%\logs\install-windows.log"
  echo.
  pause
  exit /b 3
)

call "%RUNTIME_DIR%\start_windows.bat"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo.
  echo [ERROR] The recognition system failed to start. Exit code: %EXIT_CODE%
  echo Run 04-Windows Troubleshooting and send the displayed details to technical support.
  echo.
  pause
)
exit /b %EXIT_CODE%
