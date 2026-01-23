@echo off
REM Build script for NeuralNetworkVisualizer (Windows)
REM Cross-platform compatible: uses system CMake if available, otherwise MSYS2

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%build"
set "BUILD_TYPE=Release"
set "CLEAN_BUILD=0"
set "RUN_TESTS=0"
set "MSYS2_PATH=F:\msys2"

REM Parse arguments
:parse_args
if "%1"=="" goto :args_done
if /i "%1"=="--debug" (
    set "BUILD_TYPE=Debug"
    shift
    goto :parse_args
)
if /i "%1"=="--clean" (
    set "CLEAN_BUILD=1"
    shift
    goto :parse_args
)
if /i "%1"=="--test" (
    set "RUN_TESTS=1"
    shift
    goto :parse_args
)
if /i "%1"=="--help" (
    goto :show_help
)
if /i "%1"=="-h" (
    goto :show_help
)
if /i "%1"=="--msys2" (
    if not "%2"=="" (
        set "MSYS2_PATH=%2"
        shift
    )
    shift
    goto :parse_args
)
echo Unknown option: %1
exit /b 1

:args_done

echo ========================================
echo Neural Network Visualizer Build Script
echo ========================================
echo Build type: %BUILD_TYPE%
echo ========================================
echo.

REM Check if MSYS2 exists
if exist "%MSYS2_PATH%\msys2_shell.cmd" (
    set "USE_MSYS2=1"
    echo Using MSYS2 at: %MSYS2_PATH%
) else (
    set "USE_MSYS2=0"
    echo Using system CMake
)
echo.

REM Clean build directory if requested
if "%CLEAN_BUILD%"=="1" (
    if exist "%BUILD_DIR%" (
        echo Cleaning build directory...
        rmdir /s /q "%BUILD_DIR%"
    )
)

if "%USE_MSYS2%"=="1" (
    REM Convert Windows paths to Unix paths for MSYS2
    set "UNIX_SCRIPT_DIR=/c/Users/Administrator/cpp_demo_project/NeuralNetworkVisualizer"

    echo Configuring with CMake (MSYS2)...
    "%MSYS2_PATH%\msys2_shell.cmd" -mingw64 -defterm -no-start -c "cd !UNIX_SCRIPT_DIR! && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=%BUILD_TYPE%"

    if errorlevel 1 (
        echo CMake configuration failed!
        exit /b 1
    )

    echo.
    echo Building project...
    "%MSYS2_PATH%\msys2_shell.cmd" -mingw64 -defterm -no-start -c "cd !UNIX_SCRIPT_DIR! && cmake --build build --config %BUILD_TYPE%"

    if errorlevel 1 (
        echo Build failed!
        exit /b 1
    )
) else (
    REM Use system CMake
    echo Configuring with CMake...
    cmake -S "%SCRIPT_DIR%." -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=%BUILD_TYPE%

    if errorlevel 1 (
        echo CMake configuration failed!
        exit /b 1
    )

    echo.
    echo Building project...
    cmake --build "%BUILD_DIR%" --config %BUILD_TYPE%

    if errorlevel 1 (
        echo Build failed!
        exit /b 1
    )
)

REM Run tests if requested
if "%RUN_TESTS%"=="1" (
    echo.
    echo Running tests...
    ctest --test-dir "%BUILD_DIR%" --output-on-failure
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Executables:
echo   - Main app: %BUILD_DIR%\%BUILD_TYPE%\NeuralNetworkVisualizer.exe
echo   - Tests:    %BUILD_DIR%\tests\%BUILD_TYPE%\FunctionalTest.exe
echo               %BUILD_DIR%\tests\%BUILD_TYPE%\CNNDiagnostic.exe
echo.
exit /b 0

:show_help
echo Usage: build.bat [OPTIONS]
echo.
echo Options:
echo   --debug         Build in Debug mode (default: Release)
echo   --clean         Clean build directory before building
echo   --test          Run tests after build
echo   --msys2 [PATH] Use MSYS2 at specified path (default: F:\msys2)
echo   -h, --help     Show this help message
exit /b 0
