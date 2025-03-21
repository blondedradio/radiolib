@echo off

if [%1]==[] goto error
python --version >nul 2>&1

if %errorlevel% neq 0 (
    echo Python is not installed.
    pause
    exit /b
)
set args=--folder="%~1"
if not [%2]==[] set args=%args% --name="%2"
cd /d "%~dp0"
python .\emotes.py %args%
pause
exit /b

:error
echo You have to provide a folder, try drag-and-dropping a folder onto the .bat file.
pause
exit /B 1