#!/bin/bash
# Run script for NeuralNetworkVisualizer (Unix/Linux/macOS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="Debug"
MODE="${1:-}"
BUILD_GUI="ON"

# Check if build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Build directory not found. Running initial configuration..."
    cmake -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DBUILD_GUI="${BUILD_GUI}"
    if [ $? -ne 0 ]; then
        echo "ERROR: CMake configuration failed!"
        exit 1
    fi
fi

# Check if executable exists
EXE_PATH="${BUILD_DIR}/NeuralNetworkVisualizer"
if [ ! -f "${EXE_PATH}" ]; then
    echo "Executable not found. Building project..."
    cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}"
    if [ $? -ne 0 ]; then
        echo "ERROR: Build failed!"
        exit 1
    fi
fi

echo ""
echo "===================================="
echo "Neural Network Visualizer Launcher"
echo "===================================="
echo ""

# Launch the application
if [ -z "${MODE}" ]; then
    echo "Launching application (GUI mode selection)..."
    "${EXE_PATH}"
else
    echo "Launching application in mode: ${MODE}"
    "${EXE_PATH}" "${MODE}"
fi

if [ $? -ne 0 ]; then
    echo ""
    echo "Application exited with error code $?"
fi
