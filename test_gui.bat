@echo off
echo ========================================
echo Neural Network Visualizer - Qt Launcher  
echo ========================================
echo.

set EXE_PATH=build\Debug\NeuralNetworkVisualizer.exe

if not exist "%EXE_PATH%" (
    echo ERROR: Executable not found: %EXE_PATH%
    echo Please build the project first: cmake --build build
    pause
    exit /b 1
)

echo Testing MLP Mode...
echo.
start "MLP Visualizer" "%EXE_PATH%" mlp

timeout /t 2 /nobreak >nul

echo Testing CNN Mode...
echo.
start "CNN Visualizer" "%EXE_PATH%" cnn

echo.
echo Both windows should now be open.
echo If they don't appear, check for error messages.
echo.
pause
