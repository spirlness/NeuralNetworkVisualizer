@echo off
REM Run script for NeuralNetworkVisualizer (Windows)
REM Builds the project if needed and launches the application

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%build"
set "BUILD_TYPE=Debug"
set "MODE=%1"
set "BUILD_GUI=ON"

REM Default mode selection
if "%MODE%"=="" (
    set "AUTO_MODE=1"
)

REM Check if build directory exists
if not exist "%BUILD_DIR%" (
    echo Build directory not found. Running initial configuration...
    cmake -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DBUILD_GUI=%BUILD_GUI%
    if errorlevel 1 (
        echo ERROR: CMake configuration failed!
        pause
        exit /b 1
    )
)

REM Check if executable exists
set "EXE_PATH=%BUILD_DIR%\%BUILD_TYPE%\NeuralNetworkVisualizer.exe"
if not exist "%EXE_PATH%" (
    echo Executable not found. Building project...
    cmake --build "%BUILD_DIR%" --config %BUILD_TYPE%
    if errorlevel 1 (
        echo ERROR: Build failed!
        pause
        exit /b 1
    )
)

echo.
echo ====================================
echo Neural Network Visualizer Launcher
echo ====================================
echo.

REM Launch the application
if "%AUTO_MODE%"=="1" (
    echo Launching application (GUI mode selection)...
    "%EXE_PATH%"
) else (
    echo Launching application in mode: %MODE%
    "%EXE_PATH%" %MODE%
)

if errorlevel 1 (
    echo.
    echo Application exited with error code %errorlevel%
    pause
)
