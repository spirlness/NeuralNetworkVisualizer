#!/bin/bash
# Build script for NeuralNetworkVisualizer (Unix/Linux/macOS)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Default values
BUILD_TYPE="Release"
CLEAN_BUILD=false
RUN_TESTS=false
INSTALL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --install)
            INSTALL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug      Build in Debug mode (default: Release)"
            echo "  --clean      Clean build directory before building"
            echo "  --test       Run tests after build"
            echo "  --install    Install to system"
            echo "  -h, --help  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Neural Network Visualizer Build Script"
echo "========================================"
echo "Build type: ${BUILD_TYPE}"
echo "========================================"
echo ""

# Clean build directory if requested
if [ "${CLEAN_BUILD}" = true ] && [ -d "${BUILD_DIR}" ]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
fi

# Detect generator
GENERATOR="Unix Makefiles"
if command -v ninja &> /dev/null; then
    GENERATOR="Ninja"
    echo "Using Ninja generator"
else
    echo "Using Unix Makefiles"
fi

# Configure CMake
echo "Configuring with CMake..."
cmake -S "${SCRIPT_DIR}" \
      -B "${BUILD_DIR}" \
      -G "${GENERATOR}" \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

# Build
echo ""
echo "Building project..."
cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}"

# Run tests if requested
if [ "${RUN_TESTS}" = true ]; then
    echo ""
    echo "Running tests..."
    cd "${BUILD_DIR}"
    ctest --output-on-failure
fi

# Install if requested
if [ "${INSTALL}" = true ]; then
    echo ""
    echo "Installing..."
    cmake --install "${BUILD_DIR}"
fi

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo ""
echo "Executables:"
echo "  - Main app: ${BUILD_DIR}/NeuralNetworkVisualizer"
if [ "${BUILD_TESTS}" != "OFF" ]; then
    echo "  - Tests:    ${BUILD_DIR}/tests/FunctionalTest"
    echo "              ${BUILD_DIR}/tests/CNNDiagnostic"
fi
echo ""
