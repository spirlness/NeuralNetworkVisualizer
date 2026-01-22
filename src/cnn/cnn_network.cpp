#include "cnn/cnn_network.h"
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <mutex>

#include <limits>

CNNNetwork::CNNNetwork()
    : inputChannels_(0), inputHeight_(0), inputWidth_(0),
      currentChannels_(0), currentHeight_(0), currentWidth_(0) {}

void CNNNetwork::setInputSize(size_t channels, size_t height, size_t width) {
    inputChannels_ = channels;
    inputHeight_ = height;
    inputWidth_ = width;
    currentChannels_ = channels;
    currentHeight_ = height;
    currentWidth_ = width;
}

void CNNNetwork::addConvLayer(size_t outputChannels, size_t kernelSize,
                               size_t stride, size_t padding,
                               CNNActivationType activation) {
    if (hasFlatten_) {
        throw std::runtime_error("Cannot add CNN layer after Flatten");
    }

    auto layer = std::make_shared<ConvolutionalLayer>(
        currentChannels_, currentHeight_, currentWidth_,
        outputChannels, kernelSize, stride, padding, activation);

    cnnLayers_.push_back(layer);

    currentChannels_ = layer->outputChannels();
    currentHeight_ = layer->outputHeight();
    currentWidth_ = layer->outputWidth();
}

void CNNNetwork::addPoolingLayer(size_t poolSize, size_t stride, PoolingType type) {
    if (hasFlatten_) {
        throw std::runtime_error("Cannot add CNN layer after Flatten");
    }

    auto layer = std::make_shared<PoolingLayer>(
        currentChannels_, currentHeight_, currentWidth_,
        poolSize, stride, type);

    cnnLayers_.push_back(layer);

    currentChannels_ = layer->outputChannels();
    currentHeight_ = layer->outputHeight();
    currentWidth_ = layer->outputWidth();
}

void CNNNetwork::addFlattenLayer() {
    if (hasFlatten_) {
        throw std::runtime_error("Flatten layer already added");
    }

    auto layer = std::make_shared<FlattenLayer>(
        currentChannels_, currentHeight_, currentWidth_);

    cnnLayers_.push_back(layer);
    hasFlatten_ = true;
    flattenedSize_ = layer->flattenedSize();
}

void CNNNetwork::addDenseLayer(size_t neurons, ActivationType activation) {
    if (!hasFlatten_) {
        addFlattenLayer();
    }

    denseLayerSizes_.push_back(static_cast<int>(neurons));
    denseActivations_.push_back(activation);
}

void CNNNetwork::build() {
    if (inputChannels_ == 0 || inputHeight_ == 0 || inputWidth_ == 0) {
        throw std::runtime_error("Input size not set");
    }

    if (!hasFlatten_ && !denseLayerSizes_.empty()) {
        addFlattenLayer();
    }

    // 构建全连接层
    denseLayers_.clear();

    if (flattenedSize_ > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Flattened size exceeds supported dense layer size");
    }
    int prevSize = static_cast<int>(flattenedSize_);

    for (size_t i = 0; i < denseLayerSizes_.size(); ++i) {
        denseLayers_.emplace_back(prevSize, denseLayerSizes_[i], denseActivations_[i]);
        prevSize = denseLayerSizes_[i];
    }

    isBuilt_ = true;
}

void CNNNetwork::buildSimpleCNN(size_t inputChannels, size_t inputHeight, size_t inputWidth,
                                 size_t numClasses) {
    setInputSize(inputChannels, inputHeight, inputWidth);

    // Conv1: 32 filters, 3x3
    addConvLayer(32, 3, 1, 1, CNNActivationType::ReLU);
    // Pool1: 2x2
    addPoolingLayer(2, 2, PoolingType::Max);
    // Conv2: 64 filters, 3x3
    addConvLayer(64, 3, 1, 1, CNNActivationType::ReLU);
    // Pool2: 2x2
    addPoolingLayer(2, 2, PoolingType::Max);
    // Flatten
    addFlattenLayer();
    // FC1: 128 neurons
    addDenseLayer(128, ActivationType::ReLU);
    // Output
    addDenseLayer(numClasses, ActivationType::Sigmoid);

    build();
}

std::vector<double> CNNNetwork::forward(const Tensor& input) {
    std::lock_guard<std::mutex> lock(mutex_);
    return forwardInternal(input);
}

void CNNNetwork::backward(const std::vector<double>& target) {
    std::lock_guard<std::mutex> lock(mutex_);
    backwardInternal(target);
}

void CNNNetwork::updateWeights(double learningRate) {
    std::lock_guard<std::mutex> lock(mutex_);
    updateWeightsInternal(learningRate);
}

double CNNNetwork::train(const std::vector<Tensor>& inputs,
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

double CNNNetwork::calculateLoss(const std::vector<double>& output,
                                  const std::vector<double>& target) {
    std::lock_guard<std::mutex> lock(mutex_);
    return calculateLossInternal(output, target);
}

size_t CNNNetwork::totalParameters() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto& layer : cnnLayers_) {
        total += layer->parameterCount();
    }
    for (const auto& layer : denseLayers_) {
        total += static_cast<size_t>(layer.inputSize) * static_cast<size_t>(layer.outputSize) +
                 static_cast<size_t>(layer.outputSize);
    }
    return total;
}

std::vector<Tensor> CNNNetwork::getAllFeatureMaps() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Tensor> featureMaps;
    featureMaps.reserve(cnnLayers_.size());
    for (const auto& layer : cnnLayers_) {
        featureMaps.push_back(layer->getOutput());
    }
    return featureMaps;
}

std::vector<std::vector<Tensor>> CNNNetwork::getAllKernels() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::vector<Tensor>> allKernels;
    allKernels.reserve(cnnLayers_.size());
    for (const auto& layer : cnnLayers_) {
        allKernels.push_back(layer->getWeights());
    }
    return allKernels;
}

std::vector<std::string> CNNNetwork::getLayerDescriptions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> descriptions;

    std::stringstream ss;
    ss << "Input: " << inputChannels_ << "x" << inputHeight_ << "x" << inputWidth_;
    descriptions.push_back(ss.str());

    for (const auto& layer : cnnLayers_) {
        ss.str("");
        ss << layer->name() << ": "
           << layer->outputChannels() << "x"
           << layer->outputHeight() << "x"
           << layer->outputWidth();
        if (layer->hasTrainableParams()) {
            ss << " (" << layer->parameterCount() << " params)";
        }
        descriptions.push_back(ss.str());
    }

    for (const auto& layer : denseLayers_) {
        ss.str("");
        ss << "Dense: " << layer.outputSize
           << " (" << (static_cast<size_t>(layer.inputSize) * static_cast<size_t>(layer.outputSize) +
                        static_cast<size_t>(layer.outputSize))
           << " params)";
        descriptions.push_back(ss.str());
    }

    return descriptions;
}

std::vector<double> CNNNetwork::forwardInternal(const Tensor& input) {
    if (!isBuilt_) {
        throw std::runtime_error("Network not built");
    }
    validateInputShape(input);

    Tensor current = input;
    for (auto& layer : cnnLayers_) {
        current = layer->forward(current);
    }

    flattenedOutput_ = current.flatten();

    std::vector<double> denseInput = flattenedOutput_;

    for (auto& layer : denseLayers_) {
        layer.input = denseInput;

        for (int j = 0; j < layer.outputSize; ++j) {
            const size_t jIdx = static_cast<size_t>(j);
            double sum = layer.biases[jIdx];
            for (int i = 0; i < layer.inputSize; ++i) {
                const size_t iIdx = static_cast<size_t>(i);
                sum += layer.weight(j, i) * denseInput[iIdx];
            }

            switch (layer.activation) {
                case ActivationType::Sigmoid:
                    layer.output[jIdx] = 1.0 / (1.0 + std::exp(-std::clamp(sum, -500.0, 500.0)));
                    break;
                case ActivationType::ReLU:
                    layer.output[jIdx] = std::max(0.0, sum);
                    break;
                case ActivationType::Tanh:
                    layer.output[jIdx] = std::tanh(sum);
                    break;
            }
        }

        denseInput = layer.output;
    }

    lastOutput_ = denseLayers_.empty() ? flattenedOutput_ : denseLayers_.back().output;
    return lastOutput_;
}

void CNNNetwork::backwardInternal(const std::vector<double>& target) {
    if (denseLayers_.empty()) return;

    Layer& outputLayer = denseLayers_.back();
    if (target.size() != static_cast<size_t>(outputLayer.outputSize)) {
        throw std::invalid_argument("Target size mismatch");
    }

    for (int j = 0; j < outputLayer.outputSize; ++j) {
        const size_t jIdx = static_cast<size_t>(j);
        double error = target[jIdx] - outputLayer.output[jIdx];
        double sum = outputLayer.biases[jIdx];
        for (int i = 0; i < outputLayer.inputSize; ++i) {
            const size_t iIdx = static_cast<size_t>(i);
            sum += outputLayer.weight(j, i) * outputLayer.input[iIdx];
        }

        double derivative = 1.0;
        switch (outputLayer.activation) {
            case ActivationType::Sigmoid: {
                double s = 1.0 / (1.0 + std::exp(-std::clamp(sum, -500.0, 500.0)));
                derivative = s * (1.0 - s);
                break;
            }
            case ActivationType::ReLU:
                derivative = sum > 0 ? 1.0 : 0.0;
                break;
            case ActivationType::Tanh: {
                double t = std::tanh(sum);
                derivative = 1.0 - t * t;
                break;
            }
        }
        outputLayer.delta[jIdx] = error * derivative;
    }

    for (int l = static_cast<int>(denseLayers_.size()) - 2; l >= 0; --l) {
        const size_t lIdx = static_cast<size_t>(l);
        Layer& currentLayer = denseLayers_[lIdx];
        Layer& nextLayer = denseLayers_[lIdx + 1];

        for (int i = 0; i < currentLayer.outputSize; ++i) {
            const size_t iIdx = static_cast<size_t>(i);
            double error = 0.0;
            for (int j = 0; j < nextLayer.outputSize; ++j) {
                const size_t jIdx = static_cast<size_t>(j);
                error += nextLayer.weight(j, i) * nextLayer.delta[jIdx];
            }
            double sum = currentLayer.biases[iIdx];
            for (int k = 0; k < currentLayer.inputSize; ++k) {
                const size_t kIdx = static_cast<size_t>(k);
                sum += currentLayer.weight(i, k) * currentLayer.input[kIdx];
            }

            double derivative = 1.0;
            switch (currentLayer.activation) {
                case ActivationType::Sigmoid: {
                    double s = 1.0 / (1.0 + std::exp(-std::clamp(sum, -500.0, 500.0)));
                    derivative = s * (1.0 - s);
                    break;
                }
                case ActivationType::ReLU:
                    derivative = sum > 0 ? 1.0 : 0.0;
                    break;
                case ActivationType::Tanh: {
                    double t = std::tanh(sum);
                    derivative = 1.0 - t * t;
                    break;
                }
            }
            currentLayer.delta[iIdx] = error * derivative;
        }
    }

    Layer& firstDense = denseLayers_[0];
    Tensor gradFromDense(1, 1, flattenedSize_);
    for (size_t i = 0; i < flattenedSize_; ++i) {
        const int iInt = static_cast<int>(i);
        double grad = 0.0;
        for (int j = 0; j < firstDense.outputSize; ++j) {
            grad += firstDense.weight(j, iInt) * firstDense.delta[static_cast<size_t>(j)];
        }
        gradFromDense(0, 0, i) = grad;
    }

    Tensor gradCurrent = gradFromDense;
    for (int i = static_cast<int>(cnnLayers_.size()) - 1; i >= 0; --i) {
        gradCurrent = cnnLayers_[static_cast<size_t>(i)]->backward(gradCurrent);
    }
}

void CNNNetwork::updateWeightsInternal(double learningRate) {
    for (auto& layer : denseLayers_) {
        for (int j = 0; j < layer.outputSize; ++j) {
            const size_t jIdx = static_cast<size_t>(j);
            double delta_lr = learningRate * layer.delta[jIdx];
            for (int i = 0; i < layer.inputSize; ++i) {
                const size_t iIdx = static_cast<size_t>(i);
                layer.weight(j, i) += delta_lr * layer.input[iIdx];
            }
            layer.biases[jIdx] += delta_lr;
        }
    }

    for (auto& layer : cnnLayers_) {
        layer->updateWeights(learningRate);
    }
}

double CNNNetwork::calculateLossInternal(const std::vector<double>& output,
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

void CNNNetwork::validateInputShape(const Tensor& input) const {
    if (input.channels() != inputChannels_ ||
        input.height() != inputHeight_ ||
        input.width() != inputWidth_) {
        throw std::invalid_argument("Input shape mismatch");
    }
}
