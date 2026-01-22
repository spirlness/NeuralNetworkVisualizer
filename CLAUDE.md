# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Configure (from project root)
cmake -B build

# Build
cmake --build build

# Run (Windows)
# Option 1: Use run.bat launcher
run.bat

# Option 2: Run directly with mode parameter (optional)
.\build\Debug\NeuralNetworkVisualizer.exe mlp    # MLP mode
.\build\Debug\NeuralNetworkVisualizer.exe cnn    # CNN mode

# Option 3: Run without arguments to see GUI selection dialog
.\build\Debug\NeuralNetworkVisualizer.exe
```

## Project Overview

A Qt-based neural network training visualizer application that demonstrates both:
1. **MLP (Multi-Layer Perceptron)** - Fully connected feedforward neural network
2. **CNN (Convolutional Neural Network)** - Convolutional layers with pooling

**Requirements:** CMake 3.16+, C++17, Qt5 or Qt6 Widgets

## Project Structure

The codebase follows a clean modular structure:
- **Headers**: All in `include/`, organized by module (`cnn/`, `visualization/`)
- **Sources**: All in `src/`, mirroring the `include/` structure
- **Entry Point**: `src/main.cpp` with CLI and GUI selection support
- **Build**: Single monolithic CMake configuration

## Architecture

### MLP Module

- **NeuralNetwork** (`include/neural_network.h`, `src/neural_network.cpp`): Core feedforward neural network implementation with forward/backward propagation, Xavier weight initialization, and support for Sigmoid/ReLU/Tanh activations. Uses MSE loss.

- **TrainingThread** (`include/training_thread.h`, `src/training_thread.cpp`): QThread subclass that runs training in background, emitting signals for epoch completion and weight updates. Supports pause/resume/stop controls.

- **NetworkView** (`include/network_view.h`, `src/network_view.cpp`): Custom QWidget that renders network topology with neurons and weighted connections.

- **LossChart** (`include/loss_chart.h`, `src/loss_chart.cpp`): Custom QWidget that plots training loss over epochs.

- **MainWindow** (`include/mainwindow.h`, `src/mainwindow.cpp`): Main application window for MLP visualizer, managing UI controls for network configuration, training parameters, and dataset selection (XOR, AND, OR, Circle classification).

### CNN Module

- **Tensor** (`include/cnn/tensor.h`, `src/cnn/tensor.cpp`): 3D tensor class for feature maps (CHW format: Channels, Height, Width). Supports Xavier/He initialization, padding, and tensor operations.

- **CNNLayerBase** (`include/cnn/cnn_layer_base.h`): Abstract base class for all CNN layers with virtual forward/backward/updateWeights methods.

- **ConvolutionalLayer** (`include/cnn/conv_layer.h`, `src/cnn/conv_layer.cpp`): 2D convolutional layer with padding, stride support, forward/backward propagation, and gradient computation.

- **PoolingLayer** (`include/cnn/pooling_layer.h`, `src/cnn/pooling_layer.cpp`): MaxPool and AvgPool layers with max index tracking for backpropagation.

- **FlattenLayer** (`include/cnn/flatten_layer.h`, `src/cnn/flatten_layer.cpp`): Flatten layer bridging CNN to Dense layers.

- **CNNNetwork** (`include/cnn/cnn_network.h`, `src/cnn/cnn_network.cpp`): Complete CNN network combining CNN layers with dense layers. Includes train method with full forward/backward propagation.

### CNN Visualization

- **CNNView** (`include/visualization/cnn_view.h`, `src/visualization/cnn_view.cpp`): 3D architecture visualization with layer boxes, connections, and feature map thumbnails using QPainter.

- **FeatureMapView** (`include/visualization/feature_map_view.h`, `src/visualization/feature_map_view.cpp`): Grid display of all channels in a feature map with heatmap/grayscale/Viridis color mapping.

- **CNNMainWindow** (`include/cnn_mainwindow.h`, `src/cnn_mainwindow.cpp`): Main window for CNN visualizer with configuration (input size, filters, kernel size, classes), training controls, synthetic shape dataset generation (circles, squares, crosses), and integration of CNNView, FeatureMapView, LossChart.

## Key Patterns

- Headers in `include/`, implementations in `src/`
- Qt MOC handled automatically via `CMAKE_AUTOMOC`
- MSVC builds use `/utf-8` for Chinese comment support
- **MLP Training**: Runs on dedicated `QThread` for non-blocking UI
- **CNN Training**: Currently uses `QTimer` on main thread (synchronous)
- **CNN Format**: CHW (Channels-Height-Width) tensor format
- **Mode Selection**: Supports both CLI arguments (`mlp`/`cnn`) and GUI dialog

## Usage

1. **Build**: Run `cmake -B build && cmake --build build` or use `run.bat`
2. **Launch**: Run `.\build\Debug\NeuralNetworkVisualizer.exe [mlp|cnn]` or use GUI selection
3. **Configure**: Set network parameters (layers, neurons, filters, kernel size, etc.)
4. **Generate Data**: Create training datasets
5. **Train**: Start training and watch real-time visualization
6. **Inspect**: Click on layers to view feature maps (CNN mode)
