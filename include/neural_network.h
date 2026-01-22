#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <memory>
#include <mutex>

// 激活函数类型
enum class ActivationType {
    Sigmoid,
    ReLU,
    Tanh
};

// 单个层的结构
struct Layer {
    int inputSize;
    int outputSize;
    std::vector<double> weights;
    std::vector<double> biases;
    std::vector<double> output;
    std::vector<double> input;
    std::vector<double> delta;
    ActivationType activation;

    Layer(int inSize, int outSize, ActivationType act = ActivationType::Sigmoid);
    void initializeWeights();
    
    inline double& weight(int out, int in) {
        const size_t idx = static_cast<size_t>(out) * static_cast<size_t>(inputSize) + static_cast<size_t>(in);
        return weights[idx];
    }
    inline const double& weight(int out, int in) const {
        const size_t idx = static_cast<size_t>(out) * static_cast<size_t>(inputSize) + static_cast<size_t>(in);
        return weights[idx];
    }
};

// 神经网络类
class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork() = default;

    // 网络构建
    void addLayer(int neurons, ActivationType activation = ActivationType::Sigmoid);
    void setInputSize(int size);
    void build();

    // 前向传播
    std::vector<double> forward(const std::vector<double>& input);

    // 反向传播
    void backward(const std::vector<double>& target);

    // 更新权重
    void updateWeights(double learningRate);

    // 训练
    double train(const std::vector<std::vector<double>>& inputs,
                 const std::vector<std::vector<double>>& targets,
                 double learningRate);

    // 计算损失 (均方误差)
    double calculateLoss(const std::vector<double>& output,
                         const std::vector<double>& target);

    // 获取网络结构信息
    std::vector<int> getLayerSizes() const;
    const std::vector<Layer>& getLayers() const { return layers_; }
    int getInputSize() const { return inputSize_; }
    std::vector<Layer> getLayersSnapshot() const;

    // 获取权重信息（用于可视化）
    std::vector<std::vector<std::vector<double>>> getAllWeights() const;

private:
    // 激活函数
    double activate(double x, ActivationType type);
    double activateDerivative(double x, ActivationType type);

    std::vector<double> forwardInternal(const std::vector<double>& input);
    void backwardInternal(const std::vector<double>& target);
    void updateWeightsInternal(double learningRate);
    double calculateLossInternal(const std::vector<double>& output,
                                 const std::vector<double>& target);

    int inputSize_;
    std::vector<int> layerSizes_;
    std::vector<ActivationType> activations_;
    std::vector<Layer> layers_;
    bool isBuilt_;

    mutable std::mutex mutex_;

    std::mt19937 rng_;
};

#endif // NEURAL_NETWORK_H
