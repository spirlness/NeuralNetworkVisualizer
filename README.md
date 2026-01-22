# Neural Network Visualizer (Qt / C++17)

Qt-based neural network training visualizer for MLP and a simple CNN pipeline.

## Features

- MLP (multilayer perceptron) training + visualization
- CNN (Conv/Pool/Flatten + dense head) training + feature map visualization
- Qt5/Qt6 compatible CMake build
- Windows release package included under `dist/`

## Repository Layout

- `include/`: headers
- `src/`: implementation
- `tests/`: small Qt-based test binaries
- `CMakeLists.txt`: single top-level build
- `dist/NeuralNetworkVisualizer-win64-release/`: packaged Windows Release build (exe + required DLLs/plugins)

## Build (CMake)

### Configure + build (generic)

```bash
cmake -S . -B build
cmake --build build
```

### Run

Windows:

```bat
run.bat
```

Or run the built binary directly:

```bat
build\Debug\NeuralNetworkVisualizer.exe
```

## Clang + MinGW (Windows)

This project can be built using LLVM clang targeting MinGW.

Prerequisites:

- LLVM/Clang installed
- MSYS2 MinGW64 Qt5 installed (example prefix: `F:\msys2\mingw64`)

Example:

```bash
cmake -S . -B build-clang-mingw-qt \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang++.exe" \
  -DCMAKE_CXX_FLAGS="--target=x86_64-w64-windows-gnu" \
  -DCMAKE_PREFIX_PATH="F:/msys2/mingw64" \
  -DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=OFF

cmake --build build-clang-mingw-qt --target NeuralNetworkVisualizer
```

## Packaged Release

Windows Release artifacts are available here:

- `dist/NeuralNetworkVisualizer-win64-release/NeuralNetworkVisualizer.exe`

The folder also includes Qt runtime DLLs, plugins (`platforms/qwindows.dll`, etc.), and MinGW runtime DLLs.

## Tests

Build and run the test binaries:

```bash
cmake -S . -B build
cmake --build build --target FunctionalTest CNNDiagnostic
```

Then run the produced executables from the build output folder.
