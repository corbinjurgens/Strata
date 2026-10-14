@echo off
setlocal
title Strata - LAN
cd /d "%~dp0"

set "STRATA_HOST=0.0.0.0"
set "STRATA_PORT=8080"

rem Require the API key saved by the one-time setup before allowing LAN access.
".venv\Scripts\python.exe" -c "import json,sys; c=json.load(open('strata-q2_0.json', encoding='utf-8-sig')); k=c.get('api_key'); sys.exit(0 if isinstance(k,str) and k.strip() else 1)"
if errorlevel 1 (
    echo No saved API key found, or the config could not be read.
    echo Complete the guide's one-time API-key setup before using this launcher.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "serve\server.py" --engine strata --config "strata-q2_0.json" --host "%STRATA_HOST%" --port "%STRATA_PORT%"
if errorlevel 1 pause
