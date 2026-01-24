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
set "BUILD_GUI=ON"
set "MSYS2_BASH="

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
if /i "%1"=="--no-gui" (
    set "BUILD_GUI=OFF"
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
if exist "%MSYS2_PATH%\usr\bin\bash.exe" (
    set "USE_MSYS2=1"
    set "MSYS2_BASH=%MSYS2_PATH%\usr\bin\bash.exe"
    echo Using MSYS2 at: %MSYS2_PATH%
) else (
    if exist "%MSYS2_PATH%\msys2_shell.cmd" (
        REM Fallback: older MSYS2 installs
        set "USE_MSYS2=1"
        set "MSYS2_BASH=%MSYS2_PATH%\usr\bin\bash.exe"
        echo Using MSYS2 at: %MSYS2_PATH%
    ) else (
        set "USE_MSYS2=0"
        echo Using system CMake
    )
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
    REM Convert Windows paths to MSYS2 Unix paths without FOR
    set "SCRIPT_DRIVE=%SCRIPT_DIR:~0,1%"
    set "SCRIPT_DIR_NO_DRIVE=%SCRIPT_DIR:~2%"
    set "UNIX_SCRIPT_DIR=/!SCRIPT_DRIVE!!SCRIPT_DIR_NO_DRIVE:\=/%!"
    if "!UNIX_SCRIPT_DIR:~-1!"=="/" set "UNIX_SCRIPT_DIR=!UNIX_SCRIPT_DIR:~0,-1!"

    set "BUILD_DRIVE=%BUILD_DIR:~0,1%"
    set "BUILD_DIR_NO_DRIVE=%BUILD_DIR:~2%"
    set "UNIX_BUILD_DIR=/!BUILD_DRIVE!!BUILD_DIR_NO_DRIVE:\=/%!"
    if "!UNIX_BUILD_DIR:~-1!"=="/" set "UNIX_BUILD_DIR=!UNIX_BUILD_DIR:~0,-1!"

    echo Configuring with CMake using MSYS2/MINGW64 + Ninja...
    "%MSYS2_BASH%" -lc "cd '!UNIX_SCRIPT_DIR!' && cmake -S . -B '!UNIX_BUILD_DIR!' -G Ninja -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DBUILD_GUI=%BUILD_GUI%"

    if errorlevel 1 (
        echo CMake configuration failed!
        exit /b 1
    )

    echo.
    echo Building project...
    "%MSYS2_BASH%" -lc "cmake --build '!UNIX_BUILD_DIR!'"

    if errorlevel 1 (
        echo Build failed!
        exit /b 1
    )
) else (
    REM Use system CMake
    echo Configuring with CMake...
    cmake -S "%SCRIPT_DIR%." -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DBUILD_GUI=%BUILD_GUI%

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

REM Normalize output locations for single-config generators such as Ninja
if "%USE_MSYS2%"=="1" (
    set "OUT_DIR=%BUILD_DIR%\%BUILD_TYPE%"
    if not exist "!OUT_DIR!" mkdir "!OUT_DIR!"

    if exist "%BUILD_DIR%\NeuralNetworkVisualizer.exe" (
        copy /y "%BUILD_DIR%\NeuralNetworkVisualizer.exe" "!OUT_DIR!\NeuralNetworkVisualizer.exe" >nul
    )

    if exist "%BUILD_DIR%\tests\FunctionalTestFixed.exe" (
        if not exist "%BUILD_DIR%\tests\%BUILD_TYPE%" mkdir "%BUILD_DIR%\tests\%BUILD_TYPE%"
        copy /y "%BUILD_DIR%\tests\FunctionalTestFixed.exe" "%BUILD_DIR%\tests\%BUILD_TYPE%\FunctionalTestFixed.exe" >nul
    )
    if exist "%BUILD_DIR%\tests\CNNDiagnostic.exe" (
        if not exist "%BUILD_DIR%\tests\%BUILD_TYPE%" mkdir "%BUILD_DIR%\tests\%BUILD_TYPE%"
        copy /y "%BUILD_DIR%\tests\CNNDiagnostic.exe" "%BUILD_DIR%\tests\%BUILD_TYPE%\CNNDiagnostic.exe" >nul
    )
)

REM Run tests if requested
if "%RUN_TESTS%"=="1" (
    echo.
    echo Running tests...
    if "%USE_MSYS2%"=="1" (
        "%MSYS2_BASH%" -lc "ctest --test-dir '!UNIX_BUILD_DIR!' --output-on-failure"
    ) else (
        ctest --test-dir "%BUILD_DIR%" --output-on-failure
    )
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Executables:
echo   - Main app: %BUILD_DIR%\%BUILD_TYPE%\NeuralNetworkVisualizer.exe
echo   - Tests:    %BUILD_DIR%\tests\%BUILD_TYPE%\FunctionalTestFixed.exe
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
echo   --no-gui        Build without the Qt GUI (tests/CLI only)
echo   --msys2 [PATH] Use MSYS2 at specified path (default: F:\msys2)
echo   -h, --help     Show this help message
exit /b 0
