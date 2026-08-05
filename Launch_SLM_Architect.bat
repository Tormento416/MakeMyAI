@echo off
title MakeMyAI Studio
echo ========================================================
echo   Launching MakeMyAI Studio
echo ========================================================
if exist "dist\MakeMyAI.exe" (
    start "" "dist\MakeMyAI.exe"
) else (
    python app.py
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo Application exited with an error code: %ERRORLEVEL%
        pause
    )
)




