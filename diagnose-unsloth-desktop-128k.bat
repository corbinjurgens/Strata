@echo off
setlocal
cd /d "%~dp0"
title Strata desktop stall diagnostics
echo AUTOMATIC TEST: this starts Strata, sends the test prompts and stops it itself.
echo Before starting, stop your normal Strata server and close Codex and other apps.
echo Leave this window open. Do not start a Strata BAT or send your own requests.
echo Your usual port 8080 is not used: the test uses a private localhost port 8081.
echo Two runs include a four-minute idle pause each. These pauses are intentional.
echo Allow around 15-25 minutes and wait for the FINISHED or STOPPED message below.
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\diagnose_desktop_stalls.py" %*
set "STRATA_DIAGNOSTIC_EXIT=%ERRORLEVEL%"
if "%STRATA_DIAGNOSTIC_EXIT%"=="0" (
  echo FINISHED: the requested operation completed. See the output above for results.
) else (
  echo STOPPED: the diagnostic did not finish. Keep the saved partial results.
)
pause
exit /b %STRATA_DIAGNOSTIC_EXIT%
