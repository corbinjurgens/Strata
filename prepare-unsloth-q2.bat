@echo off
setlocal
title Strata - prepare existing Unsloth UD-Q2_K_XL download
cd /d "%~dp0"
".venv\Scripts\python.exe" "tools\prepare_unsloth_q2.py" --prepare %*
if errorlevel 1 (
    echo Preparation did not complete. Read the message above.
) else (
    echo Prepared. Run run-unsloth-ud-q2_k_xl.bat when you are ready to test.
)
pause
