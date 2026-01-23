# Build Guide

## Quick Start

### Windows
```cmd
build.bat          # Build (uses MSYS2 if available, otherwise system CMake)
run.bat            # Build and run
run.bat mlp        # Build and run MLP mode
run.bat cnn        # Build and run CNN mode
```

### Linux / macOS
```bash
chmod +x build.sh run.sh  # First time only
./build.sh          # Build
./run.sh            # Build and run
./run.sh mlp        # Build and run MLP mode
./run.sh cnn        # Build and run CNN mode
```

## Build Options

### Windows (`build.bat`)
| Option | Description |
|--------|-------------|
| `--debug` | Build in Debug mode (default: Release) |
| `--clean` | Clean build directory before building |
| `--test` | Run tests after build |
| `--no-gui` | Build without the Qt GUI (tests/CLI only) |
| `--msys2 [PATH]` | Use MSYS2 at specified path (default: F:\msys2) |
| `-h, --help` | Show help message |

```cmd
build.bat --debug --clean --test
```

### Unix/Linux/macOS (`./build.sh`)
| Option | Description |
|--------|-------------|
| `--debug` | Build in Debug mode (default: Release) |
| `--clean` | Clean build directory before building |
| `--test` | Run tests after build |
| `--install` | Install to system |
| `--no-gui` | Build without the Qt GUI (tests/CLI only) |
| `-h, --help` | Show help message |

```bash
./build.sh --debug --clean --test
```

## Manual CMake Build

### Standard CMake (All Platforms)
```bash
cmake -S . -B build
cmake --build build --config Release  # Use --config for Windows, omit for Unix
```

### Build Specific Targets
```bash
cmake --build build --target NeuralNetworkVisualizer
cmake --build build --target FunctionalTest CNNDiagnostic
```

### Build Without Tests
```bash
cmake -S . -B build -DBUILD_TESTS=OFF
cmake --build build
```

### Headless Build (No Qt)
```bash
cmake -S . -B build -DBUILD_GUI=OFF
cmake --build build
```

### CMake with Generator (Advanced)

#### Unix Makefiles
```bash
cmake -S . -B build -G "Unix Makefiles"
make -C build
```

#### Ninja (faster builds)
```bash
cmake -S . -B build -G Ninja
ninja -C build
```

#### Visual Studio (Windows)
```bash
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

#### Xcode (macOS)
```bash
cmake -S . -B build -G Xcode
xcodebuild -project build/NeuralNetworkVisualizer.xcodeproj -scheme NeuralNetworkVisualizer
```

## Running Tests

### Using CTest
```bash
cd build
ctest --output-on-failure
ctest --verbose                    # Verbose output
ctest -R FunctionalTest             # Run specific test
```

When building without the GUI target, tests are located under `build/tests`:
```bash
ctest --test-dir build/tests --output-on-failure
```

### Direct Execution
```bash
# Windows
.\build\tests\Debug\FunctionalTest.exe
.\build\tests\Debug\CNNDiagnostic.exe

# Linux/macOS
./build/tests/FunctionalTest
./build/tests/CNNDiagnostic
```

## Platform-Specific Notes

### Windows
- **Qt Installation**: Required. MSYS2 MinGW64, Visual Studio, or system Qt.
- **Dependencies**: Qt5/6, C++17 compiler, CMake 3.16+
- **MSYS2 Path**: Edit `build.bat` if MSYS2 is not at `F:\msys2`

### Linux
```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install cmake qt5-default qtbase5-dev build-essential

# Or Qt6
sudo apt-get install cmake qt6-base-dev qt6-base-private-dev build-essential

# Build
./build.sh
```

If Qt is not installed, build tests/CLI only:
```bash
./build.sh --no-gui
```

### macOS
```bash
# Install dependencies via Homebrew
brew install cmake qt5

# Or Qt6
brew install cmake qt6

# Build
./build.sh
```

If Qt is not installed, build tests/CLI only:
```bash
./build.sh --no-gui
```

## Installation

```bash
cmake --install build
```

Default installation locations:
- **Windows**: `C:\Program Files\NeuralNetworkVisualizer\`
- **Linux**: `/usr/local/bin/` (requires sudo)
- **macOS**: `/usr/local/bin/`

## Clean Build

```bash
rm -rf build      # Unix/macOS
rmdir /s /q build  # Windows
```

Or use the `--clean` flag with build scripts.
