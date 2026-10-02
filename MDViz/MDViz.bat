@echo off
rem Launch MDViz with the project virtual environment. Extra arguments are passed through.
rem Works from any folder: relative file paths are resolved from where you run it.
rem In PowerShell type .\MDViz.bat (or the full path).
setlocal
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
set "PYTHONIOENCODING=utf-8"
"%~dp0..\.venv\Scripts\python.exe" -m mdviz %*
if errorlevel 1 pause
endlocal
