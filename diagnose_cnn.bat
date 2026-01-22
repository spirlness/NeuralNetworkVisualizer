@echo off
echo ================================================
echo CNN 模式诊断报告
echo ================================================
echo.

echo [1] 检查可执行文件...
if exist "build\Debug\NeuralNetworkVisualizer.exe" (
    echo ✓ 可执行文件存在
    dir build\Debug\NeuralNetworkVisualizer.exe | find "NeuralNetworkVisualizer.exe"
) else (
    echo ✗ 可执行文件不存在！
    echo 请先运行: cmake --build build
    pause
    exit /b 1
)

echo.
echo [2] 测试命令行参数...
build\Debug\NeuralNetworkVisualizer.exe --help
if %errorlevel% equ 0 (
    echo ✓ 程序可以正常启动
) else (
    echo ✗ 程序启动失败，错误代码: %errorlevel%
)

echo.
echo [3] 运行核心功能测试...
if exist "build\tests\Debug\FunctionalTest.exe" (
    build\tests\Debug\FunctionalTest.exe
    if %errorlevel% equ 0 (
        echo ✓ 核心功能测试通过
    ) else (
        echo ✗ 核心功能测试失败
    )
) else (
    echo ⚠ 测试程序不存在，跳过
)

echo.
echo [4] 运行 CNN 诊断测试...
if exist "build\tests\Debug\CNNDiagnostic.exe" (
    build\tests\Debug\CNNDiagnostic.exe
    if %errorlevel% equ 0 (
        echo ✓ CNN 诊断测试通过
    ) else (
        echo ✗ CNN 诊断测试失败
    )
) else (
    echo ⚠ CNN 诊断程序不存在，跳过
)

echo.
echo ================================================
echo 诊断完成
echo ================================================
echo.
echo 如果所有测试都通过，说明 CNN 功能正常。
echo.
echo 要启动 CNN GUI，请运行：
echo   run.bat cnn
echo.
echo 或直接运行：
echo   build\Debug\NeuralNetworkVisualizer.exe cnn
echo.
echo 注意：GUI 程序会在新窗口中打开，请查看屏幕上的窗口。
echo       如果没有窗口出现，可能是被防火墙或杀毒软件阻止。
echo.
pause
