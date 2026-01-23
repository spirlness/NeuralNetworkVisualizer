@echo off
REM Build script for NeuralNetworkVisualizer
REM Uses MSYS2 MinGW64 environment to avoid compiler toolchain conflicts

echo ========================================
echo Neural Network Visualizer Build Script
echo ========================================
echo.

REM Clean build directory
if exist build (
    echo Cleaning build directory...
    rmdir /s /q build
)

echo Configuring with CMake...
F:\msys2\msys2_shell.cmd -mingw64 -defterm -no-start -c "cd /c/Users/Administrator/cpp_demo_project/NeuralNetworkVisualizer && cmake -S . -B build -G Ninja"

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    exit /b 1
)

echo.
echo Building project...
F:\msys2\msys2_shell.cmd -mingw64 -defterm -no-start -c "cd /c/Users/Administrator/cpp_demo_project/NeuralNetworkVisualizer && cmake --build build"

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Executables:
echo   - Main app: build\NeuralNetworkVisualizer.exe
echo   - Tests:    build\tests\FunctionalTest.exe
echo               build\tests\CNNDiagnostic.exe
echo.
