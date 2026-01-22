#include <iostream>
#include <vector>
#include <cassert>
#include "neural_network.h"
#include "cnn/cnn_network.h"
#include "cnn/tensor.h"

// 自动化功能测试

void testMLPBasicFunctionality() {
    std::cout << "=== 测试 MLP 基本功能 ===" << std::endl;
    
    try {
        // 创建简单的 XOR 网络
        NeuralNetwork network;
        network.setInputSize(2);
        network.addLayer(4, ActivationType::ReLU);
        network.addLayer(1, ActivationType::Sigmoid);
        network.build();
        
        std::cout << "✓ MLP 网络创建成功" << std::endl;
        
        // 测试前向传播
        std::vector<double> input = {0.5, 0.5};
        std::vector<double> output = network.forward(input);
        
        assert(output.size() == 1);
        assert(output[0] >= 0.0 && output[0] <= 1.0);
        std::cout << "✓ MLP 前向传播成功，输出: " << output[0] << std::endl;
        
        // 测试训练数据
        std::vector<std::vector<double>> inputs = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
        std::vector<std::vector<double>> targets = {
            {0}, {1}, {1}, {0}
        };
        
        // 训练一个 epoch
        double loss = network.train(inputs, targets, 0.1);
        assert(loss >= 0.0);
        std::cout << "✓ MLP 训练成功，损失: " << loss << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ MLP 测试失败: " << e.what() << std::endl;
        throw;
    }
}

void testMLPEdgeCases() {
    std::cout << "\n=== 测试 MLP 边界情况 ===" << std::endl;
    
    // 测试除零保护
    try {
        NeuralNetwork network;
        network.setInputSize(0); // 无效配置
        network.build();
        std::cerr << "✗ 应该抛出异常但没有" << std::endl;
        assert(false);
    } catch (const std::runtime_error& e) {
        std::cout << "✓ 正确捕获零输入异常: " << e.what() << std::endl;
    }
    
    // 测试空数据训练
    try {
        NeuralNetwork network;
        network.setInputSize(2);
        network.addLayer(1, ActivationType::Sigmoid);
        network.build();
        
        std::vector<std::vector<double>> empty;
        network.train(empty, empty, 0.1); // 应该抛出异常
        std::cerr << "✗ 应该抛出空数据异常但没有" << std::endl;
        assert(false);
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ 正确捕获空数据异常: " << e.what() << std::endl;
    }
    
    // 测试未构建网络
    try {
        NeuralNetwork network;
        network.setInputSize(2);
        network.addLayer(1, ActivationType::Sigmoid);
        std::vector<double> input = {0.5, 0.5};
        network.forward(input); // 未调用 build()
        std::cerr << "✗ 应该抛出未构建异常但没有" << std::endl;
        assert(false);
    } catch (const std::runtime_error& e) {
        std::cout << "✓ 正确捕获未构建异常: " << e.what() << std::endl;
    }
}

void testCNNBasicFunctionality() {
    std::cout << "\n=== 测试 CNN 基本功能 ===" << std::endl;
    
    try {
        CNNNetwork network;
        network.setInputSize(1, 8, 8); // 1通道, 8x8 图像
        network.addConvLayer(4, 3, 1, 1, CNNActivationType::ReLU); // 4 filters, 3x3 kernel
        network.addPoolingLayer(2, 2, PoolingType::Max);
        network.addDenseLayer(2, ActivationType::Sigmoid); // 2类分类
        network.build();
        
        std::cout << "✓ CNN 网络创建成功" << std::endl;
        
        // 创建测试输入
        Tensor input(1, 8, 8);
        for (size_t i = 0; i < 8 * 8; ++i) {
            input.data()[i] = static_cast<double>(i) / 64.0; // 归一化到 [0,1]
        }
        
        // 测试前向传播
        std::vector<double> output = network.forward(input);
        assert(output.size() == 2);
        assert(output[0] >= 0.0 && output[0] <= 1.0);
        assert(output[1] >= 0.0 && output[1] <= 1.0);
        std::cout << "✓ CNN 前向传播成功，输出: [" << output[0] << ", " << output[1] << "]" << std::endl;
        
        // 测试训练
        std::vector<Tensor> inputs = {input};
        std::vector<std::vector<double>> targets = {{1.0, 0.0}};
        double loss = network.train(inputs, targets, 0.01);
        assert(loss >= 0.0);
        std::cout << "✓ CNN 训练成功，损失: " << loss << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ CNN 测试失败: " << e.what() << std::endl;
        throw;
    }
}

void testCNNEdgeCases() {
    std::cout << "\n=== 测试 CNN 边界情况 ===" << std::endl;
    
    // 测试 stride 为 0
    try {
        CNNNetwork network;
        network.setInputSize(1, 8, 8);
        network.addConvLayer(4, 3, 0, 1); // stride = 0，应该抛出异常
        std::cerr << "✗ 应该抛出 stride=0 异常但没有" << std::endl;
        assert(false);
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ 正确捕获 stride=0 异常: " << e.what() << std::endl;
    }
    
    // 测试 Flatten 后添加 CNN 层
    try {
        CNNNetwork network;
        network.setInputSize(1, 8, 8);
        network.addConvLayer(4, 3);
        network.addFlattenLayer();
        network.addConvLayer(8, 3); // Flatten 后不能添加 CNN 层
        std::cerr << "✗ 应该抛出 Flatten 后添加层异常但没有" << std::endl;
        assert(false);
    } catch (const std::runtime_error& e) {
        std::cout << "✓ 正确捕获 Flatten 后添加层异常: " << e.what() << std::endl;
    }
    
    // 测试 Xavier 初始化除零保护
    try {
        Tensor t(10, 10, 10);
        t.xavierInit(0, 10); // fanIn = 0
        std::cerr << "✗ 应该抛出 Xavier 初始化异常但没有" << std::endl;
        assert(false);
    } catch (const std::invalid_argument& e) {
        std::cout << "✓ 正确捕获 Xavier 初始化异常: " << e.what() << std::endl;
    }
}

void testTensorOperations() {
    std::cout << "\n=== 测试 Tensor 操作 ===" << std::endl;
    
    try {
        Tensor t1(2, 3, 3);
        for (size_t i = 0; i < 2 * 3 * 3; ++i) {
            t1.data()[i] = static_cast<double>(i);
        }
        
        // 测试访问
        double val = t1.at(0, 1, 1);
        assert(val == t1.data()[0 * 9 + 1 * 3 + 1]);
        std::cout << "✓ Tensor 访问成功" << std::endl;
        
        // 测试 padding
        Tensor padded = t1.pad(1, 1);
        assert(padded.channels() == 2);
        assert(padded.height() == 5);
        assert(padded.width() == 5);
        std::cout << "✓ Tensor padding 成功" << std::endl;
        
        // 测试越界保护
        try {
            t1.at(10, 10, 10); // 越界
            std::cerr << "✗ 应该抛出越界异常但没有" << std::endl;
            assert(false);
        } catch (const std::out_of_range& e) {
            std::cout << "✓ 正确捕获越界异常" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Tensor 测试失败: " << e.what() << std::endl;
        throw;
    }
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  Neural Network 自动化功能测试" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    try {
        testMLPBasicFunctionality();
        testMLPEdgeCases();
        testCNNBasicFunctionality();
        testCNNEdgeCases();
        testTensorOperations();
        
        std::cout << "\n==========================================" << std::endl;
        std::cout << "  ✓ 所有测试通过！" << std::endl;
        std::cout << "==========================================" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n==========================================" << std::endl;
        std::cerr << "  ✗ 测试失败: " << e.what() << std::endl;
        std::cerr << "==========================================" << std::endl;
        return 1;
    }
}
