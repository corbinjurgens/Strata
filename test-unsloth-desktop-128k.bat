@echo off
setlocal
cd /d "%~dp0"
title Strata desktop performance comparison
echo Close any running Strata server, Codex and other apps before running this test.
echo This compares compact, desktop, desktop GPU share, then compact again.
echo The test opens no browser and saves results under bench\results\desktop-...
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\benchmark_desktop_profiles.py" %*
set "STRATA_TEST_EXIT=%ERRORLEVEL%"
pause
exit /b %STRATA_TEST_EXIT%
