@echo off
REM ============================================================
REM  Run MaturityAnalyzer directly (no EXE build needed)
REM ============================================================

cd /d "%~dp0"

echo Checking pyogrio...
python -c "import pyogrio" 2>nul
if errorlevel 1 (
    echo pyogrio not found - installing...
    pip install pyogrio
)

echo.
echo Starting MaturityAnalyzer...
python maturity_app.py
pause
