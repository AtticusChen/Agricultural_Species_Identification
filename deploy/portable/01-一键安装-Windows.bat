@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "RUNTIME_DIR=%~dp0runtime"

if not exist "%RUNTIME_DIR%\install_windows.bat" (
  echo.
  echo [ERROR] Required runtime files are missing.
  echo Right-click the ZIP file, choose Extract All, and run this file from the extracted folder.
  echo Package folder: "%~dp0"
  echo.
  pause
  exit /b 2
)

call "%RUNTIME_DIR%\install_windows.bat"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
  echo [ERROR] Installation did not complete.
  echo Log file: "%RUNTIME_DIR%\logs\install-windows.log"
  echo Run 04-Windows Troubleshooting if you need more details.
) else (
  echo [OK] Installation completed. Run 02-Start Recognition System for Windows.
)
echo.
pause
exit /b %EXIT_CODE%
