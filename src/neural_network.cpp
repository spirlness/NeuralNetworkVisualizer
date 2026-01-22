#include "neural_network.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <mutex>

// Layer 构造函数
Layer::Layer(int inSize, int outSize, ActivationType act)
    : inputSize(inSize), outputSize(outSize), activation(act) {
    weights.resize(static_cast<size_t>(outSize) * static_cast<size_t>(inSize));
    biases.resize(static_cast<size_t>(outSize), 0.0);
    output.resize(static_cast<size_t>(outSize), 0.0);
    input.resize(static_cast<size_t>(inSize), 0.0);
    delta.resize(static_cast<size_t>(outSize), 0.0);
    initializeWeights();
}

void Layer::initializeWeights() {
    if (inputSize == 0 || outputSize == 0) {
        throw std::invalid_argument("Layer initialization: inputSize and outputSize must be greater than 0");
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    double limit = std::sqrt(6.0 / (static_cast<double>(inputSize) + static_cast<double>(outputSize)));
    std::uniform_real_distribution<> dis(-limit, limit);

    for (auto& w : weights) {
        w = dis(gen);
    }
    for (auto& b : biases) {
        b = dis(gen) * 0.1;
    }
}

// NeuralNetwork 实现
NeuralNetwork::NeuralNetwork()
    : inputSize_(0), isBuilt_(false) {
    std::random_device rd;
    rng_ = std::mt19937(rd());
}

void NeuralNetwork::setInputSize(int size) {
    inputSize_ = size;
}

void NeuralNetwork::addLayer(int neurons, ActivationType activation) {
    layerSizes_.push_back(neurons);
    activations_.push_back(activation);
}

void NeuralNetwork::build() {
    if (inputSize_ <= 0 || layerSizes_.empty()) {
        throw std::runtime_error("Invalid network configuration");
    }

    layers_.clear();
    int prevSize = inputSize_;

    for (size_t i = 0; i < layerSizes_.size(); ++i) {
        layers_.emplace_back(prevSize, layerSizes_[i], activations_[i]);
        prevSize = layerSizes_[i];
    }

    isBuilt_ = true;
}

double NeuralNetwork::activate(double x, ActivationType type) {
    switch (type) {
        case ActivationType::Sigmoid:
            return 1.0 / (1.0 + std::exp(-std::clamp(x, -500.0, 500.0)));
        case ActivationType::ReLU:
            return std::max(0.0, x);
        case ActivationType::Tanh:
            return std::tanh(x);
        default:
            return x;
    }
}

double NeuralNetwork::activateDerivative(double x, ActivationType type) {
    switch (type) {
        case ActivationType::Sigmoid: {
            double s = activate(x, ActivationType::Sigmoid);
            return s * (1.0 - s);
        }
        case ActivationType::ReLU:
            return x > 0 ? 1.0 : 0.0;
        case ActivationType::Tanh: {
            double t = std::tanh(x);
            return 1.0 - t * t;
        }
        default:
            return 1.0;
    }
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    std::lock_guard<std::mutex> lock(mutex_);
    return forwardInternal(input);
}

void NeuralNetwork::backward(const std::vector<double>& target) {
    std::lock_guard<std::mutex> lock(mutex_);
    backwardInternal(target);
}

void NeuralNetwork::updateWeights(double learningRate) {
    std::lock_guard<std::mutex> lock(mutex_);
    updateWeightsInternal(learningRate);
}

double NeuralNetwork::train(const std::vector<std::vector<double>>& inputs,
                            const std::vector<std::vector<double>>& targets,
                            double learningRate) {
    if (inputs.empty() || targets.empty()) {
        throw std::invalid_argument("Training data cannot be empty");
    }
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Training data size mismatch");
    }
    if (!isBuilt_) {
        throw std::runtime_error("Network not built");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    double totalLoss = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<double> output = forwardInternal(inputs[i]);
        totalLoss += calculateLossInternal(output, targets[i]);
        backwardInternal(targets[i]);
        updateWeightsInternal(learningRate);
    }

    return totalLoss / static_cast<double>(inputs.size());
}

double NeuralNetwork::calculateLoss(const std::vector<double>& output,
                                    const std::vector<double>& target) {
    std::lock_guard<std::mutex> lock(mutex_);
    return calculateLossInternal(output, target);
}

std::vector<int> NeuralNetwork::getLayerSizes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> sizes;
    sizes.push_back(inputSize_);
    for (const auto& layer : layers_) {
        sizes.push_back(layer.outputSize);
    }
    return sizes;
}

std::vector<Layer> NeuralNetwork::getLayersSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return layers_;
}

std::vector<std::vector<std::vector<double>>> NeuralNetwork::getAllWeights() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::vector<std::vector<double>>> allWeights;
    for (const auto& layer : layers_) {
        std::vector<std::vector<double>> layerWeights(
            static_cast<size_t>(layer.outputSize),
            std::vector<double>(static_cast<size_t>(layer.inputSize)));
        for (int j = 0; j < layer.outputSize; ++j) {
            const size_t jIdx = static_cast<size_t>(j);
            for (int i = 0; i < layer.inputSize; ++i) {
                layerWeights[jIdx][static_cast<size_t>(i)] = layer.weight(j, i);
            }
        }
        allWeights.push_back(layerWeights);
    }
    return allWeights;
}

std::vector<double> NeuralNetwork::forwardInternal(const std::vector<double>& input) {
    if (!isBuilt_) {
        throw std::runtime_error("Network not built");
    }
    if (static_cast<int>(input.size()) != inputSize_) {
        throw std::invalid_argument("Input size mismatch");
    }

    std::vector<double> currentInput = input;

    for (auto& layer : layers_) {
        layer.input = currentInput;

        for (int j = 0; j < layer.outputSize; ++j) {
            const size_t jIdx = static_cast<size_t>(j);
            double sum = layer.biases[jIdx];
            const size_t rowOffset = jIdx * static_cast<size_t>(layer.inputSize);
            const double* w = &layer.weights[rowOffset];
            for (int i = 0; i < layer.inputSize; ++i) {
                const size_t iIdx = static_cast<size_t>(i);
                sum += w[iIdx] * currentInput[iIdx];
            }
            layer.output[jIdx] = activate(sum, layer.activation);
        }

        currentInput = layer.output;
    }

    return layers_.back().output;
}

void NeuralNetwork::backwardInternal(const std::vector<double>& target) {
    if (layers_.empty()) {
        throw std::runtime_error("Network not built");
    }
    Layer& outputLayer = layers_.back();
    if (target.size() != static_cast<size_t>(outputLayer.outputSize)) {
        throw std::invalid_argument("Target size mismatch");
    }

    for (int j = 0; j < outputLayer.outputSize; ++j) {
        const size_t jIdx = static_cast<size_t>(j);
        double error = target[jIdx] - outputLayer.output[jIdx];
        double sum = outputLayer.biases[jIdx];
        const size_t rowOffset = jIdx * static_cast<size_t>(outputLayer.inputSize);
        const double* w = &outputLayer.weights[rowOffset];
        for (int i = 0; i < outputLayer.inputSize; ++i) {
            const size_t iIdx = static_cast<size_t>(i);
            sum += w[iIdx] * outputLayer.input[iIdx];
        }
        outputLayer.delta[jIdx] = error * activateDerivative(sum, outputLayer.activation);
    }

    for (int l = static_cast<int>(layers_.size()) - 2; l >= 0; --l) {
        const size_t lIdx = static_cast<size_t>(l);
        Layer& currentLayer = layers_[lIdx];
        Layer& nextLayer = layers_[lIdx + 1];

        for (int i = 0; i < currentLayer.outputSize; ++i) {
            const size_t iIdx = static_cast<size_t>(i);
            double error = 0.0;
            for (int j = 0; j < nextLayer.outputSize; ++j) {
                const size_t jIdx = static_cast<size_t>(j);
                error += nextLayer.weight(j, i) * nextLayer.delta[jIdx];
            }
            double sum = currentLayer.biases[iIdx];
            const size_t rowOffset = iIdx * static_cast<size_t>(currentLayer.inputSize);
            const double* w = &currentLayer.weights[rowOffset];
            for (int k = 0; k < currentLayer.inputSize; ++k) {
                const size_t kIdx = static_cast<size_t>(k);
                sum += w[kIdx] * currentLayer.input[kIdx];
            }
            currentLayer.delta[iIdx] = error * activateDerivative(sum, currentLayer.activation);
        }
    }
}

void NeuralNetwork::updateWeightsInternal(double learningRate) {
    for (auto& layer : layers_) {
        for (int j = 0; j < layer.outputSize; ++j) {
            const size_t jIdx = static_cast<size_t>(j);
            const size_t rowOffset = jIdx * static_cast<size_t>(layer.inputSize);
            double* w = &layer.weights[rowOffset];
            double delta_lr = learningRate * layer.delta[jIdx];
            for (int i = 0; i < layer.inputSize; ++i) {
                const size_t iIdx = static_cast<size_t>(i);
                w[iIdx] += delta_lr * layer.input[iIdx];
            }
            layer.biases[jIdx] += delta_lr;
        }
    }
}

double NeuralNetwork::calculateLossInternal(const std::vector<double>& output,
                                            const std::vector<double>& target) {
    if (output.empty()) {
        throw std::invalid_argument("Output cannot be empty for loss calculation");
    }
    if (target.size() != output.size()) {
        throw std::invalid_argument("Target size mismatch");
    }
    double loss = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        double diff = target[i] - output[i];
        loss += diff * diff;
    }
    return loss / static_cast<double>(output.size());
}
