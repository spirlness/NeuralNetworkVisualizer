@echo off
REM Neural Network Visualizer Launcher
REM Builds and runs the application

echo ====================================
echo Neural Network Visualizer Launcher
echo ====================================
echo.

REM Check if build directory exists
if not exist "build" (
    echo Build directory not found. Running initial configuration...
    cmake -B build
    if errorlevel 1 (
        echo ERROR: CMake configuration failed!
        pause
        exit /b 1
    )
)

REM Build the project
echo Building project...
cmake --build build
if errorlevel 1 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo Build successful! Launching application...
echo.

REM Run the application with optional command-line argument
if "%1"=="" (
    .\build\Debug\NeuralNetworkVisualizer.exe
) else (
    .\build\Debug\NeuralNetworkVisualizer.exe %1
)

if errorlevel 1 (
    echo.
    echo Application exited with error code %errorlevel%
    pause
)
