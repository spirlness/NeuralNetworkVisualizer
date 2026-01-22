# NEURAL NETWORK VISUALIZER

**Generated:** 2026-01-22 09:18
**Project:** Qt-based neural network training visualizer
**Branch:** no-git

## 概览
C++17 Qt 应用，可视化演示 MLP（多层感知器）和 CNN（卷积神经网络）训练过程。

## 结构
```
cpp_demo_project/
├── include/           # 所有头文件
│   ├── cnn/          # CNN 核心层实现（Tensor, Conv, Pooling, Flatten）
│   └── visualization/ # CNN 3D 架构和特征图可视化
├── src/              # 所有实现文件（镜像 include 结构）
│   ├── main.cpp      # 入口：CLI 参数或 QMessageBox 选择 MLP/CNN 模式
│   ├── cnn/          # CNN 层实现
│   └── visualization/# CNN 可视化组件实现
├── tests/            # 测试文件
│   └── test_qt.cpp   # Qt 基础测试
├── run.bat           # Windows 启动脚本
└── CMakeLists.txt    # 单体构建，Qt5/6 兼容
```

## 哪里找什么

| 任务 | 位置 | 说明 |
|------|------|------|
| **MLP 实现** | `src/neural_network.cpp` | Xavier 初始化（带除零检查）、前向/反向传播、MSE 损失 |
| **CNN 核心** | `src/cnn/cnn_network.cpp` | 混合架构：CNN 层 + 密集层，梯度桥接 |
| **CNN 层抽象** | `include/cnn/cnn_layer_base.h` | 多态基类（forward/backward/updateWeights） |
| **Tensor 运算** | `src/cnn/tensor.cpp` | CHW 格式（通道-高-宽），He/Xavier 初始化（带除零检查） |
| **MLP 可视化** | `src/network_view.cpp` | QPainter 绘制神经元和加权连接 |
| **CNN 可视化** | `src/visualization/cnn_view.cpp` | 3D 层架构 + 特征图缩略图 |
| **特征图查看器** | `src/visualization/feature_map_view.cpp` | 热图/灰度/Viridis 色彩映射 |
| **训练循环** | `src/training_thread.cpp` (MLP) | MLP 使用 QThread 异步训练 |
| | `src/cnn_mainwindow.cpp` (CNN) | CNN 使用 QTimer 同步训练（简化可视化） |
| **入口点** | `src/main.cpp` | CLI 参数支持 + GUI 模式选择对话框 |

## 代码规范

### 命名
- **类**: `CamelCase` (`NeuralNetwork`, `CNNMainWindow`)
- **方法**: `camelCase` (`addLayer()`, `onStartTraining()`)
- **成员变量**: `camelCase_` 后缀下划线 (`inputSize_`, `cnnNetwork_`)
- **文件**: `snake_case` (`neural_network.cpp`)

### 内存管理
- 优先 `std::unique_ptr`/`std::shared_ptr`，避免裸指针
- 头文件保护：传统 `#ifndef` 而非 `#pragma once`
- Qt 组件：使用父子关系自动管理生命周期

### Qt 模式
- 信号处理器命名：`on<Action>` (`onStartTraining`)
- 使用 `CMAKE_AUTOMOC` 自动处理 MOC
- 支持 Qt5/Qt6（CMake 先找 Qt6，失败回退 Qt5）

### Include 路径
- **仅添加 `include/` 到搜索路径**
- 使用限定路径：`#include "cnn/tensor.h"` 而非 `#include "tensor.h"`
- 避免路径污染（已修复）

## 严格约束（禁止模式）

| 模式 | 上下文 | 后果 |
|------|--------|------|
| **Flatten 后加 CNN 层** | `CNNNetwork` | 抛出 `std::runtime_error("Cannot add CNN layer after Flatten")` |
| **未调用 build() 就 forward()** | 所有网络 | 抛出 `std::runtime_error("Network not built")` |
| **零输入/无层配置** | `NeuralNetwork` | 抛出 `std::runtime_error("Invalid network configuration")` |
| **激活函数溢出** | 所有层 | 必须用 `std::clamp(sum, -500.0, 500.0)` 避免 `exp()` 溢出 |
| **直接 Tensor 越界** | `Tensor`, `ConvolutionalLayer` | 抛出 `std::out_of_range` |
| **除零风险** | `Tensor::xavierInit/heInit`, `ConvolutionalLayer/PoolingLayer` 构造 | 已添加检查：`stride > 0`, `fanIn/fanOut > 0` |
| **空数据训练** | `NeuralNetwork::train` | 抛出 `std::invalid_argument("Training data cannot be empty")` |

## 架构要点

### MLP vs CNN 设计差异
- **MLP**: 单体类 `NeuralNetwork` 管理 `vector<Layer>`（简单结构体）
- **CNN**: 混合架构
  - `vector<CNNLayerPtr>` 管理空间提取层（多态）
  - **复用 MLP 的 `Layer` 结构体**作为分类头（密集层）
  - `FlattenLayer` 桥接 3D Tensor → 1D 向量

### 训练模型
- **MLP**: `QThread` 异步训练，通过信号更新 UI
- **CNN**: `QTimer` 主线程训练，简化特征图同步（可能卡顿）

### Tensor 格式
- **固定 CHW**（通道-高-宽），所有 CNN 层统一

## 命令

```bash
# 配置
cmake -B build

# 构建
cmake --build build

# 运行（Windows）- 使用启动脚本
run.bat [mlp|cnn]

# 或直接运行可执行文件
.\build\Debug\NeuralNetworkVisualizer.exe mlp  # MLP 模式（命令行）
.\build\Debug\NeuralNetworkVisualizer.exe cnn  # CNN 模式（命令行）
.\build\Debug\NeuralNetworkVisualizer.exe      # GUI 选择对话框
```

## 已知问题

- **CNN UI 响应**: CNN 训练在主线程，大数据集可能卡顿（建议迁移到 QThread，类似 MLP）
- **MSVC 特定**: 需要 `/utf-8` 编译选项支持中文注释

## 最近修复 (2026-01-22)

- ✅ **CLI 参数支持**: `main.cpp` 现在支持 `mlp`/`cnn` 命令行参数
- ✅ **文档一致性**: 修复 `CLAUDE.md` 中的可执行文件名拼写和版本号
- ✅ **根目录清理**: 移动 `main.cpp` 到 `src/`，`test_qt.cpp` 到 `tests/`，删除 `nul` 文件
- ✅ **Include 路径污染**: 移除 `CMakeLists.txt` 中的子目录路径，使用限定路径
- ✅ **数值稳定性**: 添加除零检查到 Xavier/He 初始化、stride 参数、训练数据验证
- ✅ **run.bat**: 创建启动脚本自动构建和运行
- ✅ **构建验证**: 成功编译并生成可执行文件

## 注意事项

1. **修改 CNN 架构必须先 Conv/Pool，最后 Flatten**
2. **数值稳定性：激活函数输入必须 clamp，初始化参数必须 > 0**
3. **Qt 版本：优先 Qt6，向下兼容 Qt5**
4. **编码：MSVC 必须 UTF-8 模式**
5. **模式选择：支持 CLI 参数（`mlp`/`cnn`）或 GUI 对话框**
6. **Include 规范：使用限定路径（`cnn/tensor.h` 而非 `tensor.h`）**
