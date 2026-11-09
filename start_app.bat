
@echo off
title GenAI Assignment 2 - Clinical Trial Screening
cd /d "%~dp0"

echo ============================================
echo   Clinical Trial Eligibility Screening
echo ============================================
echo.
echo Project directory: %CD%
echo.

REM Keep model caches and temporary files on E:
set "TEMP=E:\GenAI-A2\temp"
set "TMP=E:\GenAI-A2\temp"
set "HF_HOME=E:\GenAI-A2\cache\huggingface"
set "TORCH_HOME=E:\GenAI-A2\cache\torch"
set "PIP_CACHE_DIR=E:\GenAI-A2\cache\pip"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

if not exist "E:\GenAI-A2\temp" mkdir "E:\GenAI-A2\temp"

REM Check Python environment
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Expected: %CD%\.venv\Scripts\python.exe
    goto end
)

REM Check application file
if not exist "app\app.py" (
    echo ERROR: app\app.py not found!
    goto end
)

echo Starting application...
echo Please wait while the model loads.
echo.

".venv\Scripts\python.exe" "app\app.py"

echo.
echo Application stopped or encountered an error.

:end
echo.
echo Press any key to close this window.
pause >nul
