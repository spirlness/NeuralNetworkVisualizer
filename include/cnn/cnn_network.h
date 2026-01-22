#ifndef CNN_NETWORK_H
#define CNN_NETWORK_H

#include "cnn/cnn_layer_base.h"
#include "cnn/conv_layer.h"
#include "cnn/pooling_layer.h"
#include "cnn/flatten_layer.h"
#include "../neural_network.h"
#include <vector>
#include <memory>
#include <string>
#include <mutex>

/**
 * @brief CNN网络类
 *
 * 支持卷积层、池化层、展平层，并与全连接层集成
 */
class CNNNetwork {
public:
    CNNNetwork();
    ~CNNNetwork() = default;

    // 网络构建
    void setInputSize(size_t channels, size_t height, size_t width);

    void addConvLayer(size_t outputChannels, size_t kernelSize,
                      size_t stride = 1, size_t padding = 0,
                      CNNActivationType activation = CNNActivationType::ReLU);

    void addPoolingLayer(size_t poolSize, size_t stride,
                         PoolingType type = PoolingType::Max);

    void addFlattenLayer();

    void addDenseLayer(size_t neurons, ActivationType activation = ActivationType::ReLU);

    void build();

    // 预设架构
    void buildSimpleCNN(size_t inputChannels, size_t inputHeight, size_t inputWidth,
                        size_t numClasses);

    // 训练接口
    std::vector<double> forward(const Tensor& input);
    void backward(const std::vector<double>& target);
    void updateWeights(double learningRate);

    double train(const std::vector<Tensor>& inputs,
                 const std::vector<std::vector<double>>& targets,
                 double learningRate);

    double calculateLoss(const std::vector<double>& output,
                         const std::vector<double>& target);

    // 网络信息
    size_t layerCount() const { return cnnLayers_.size() + denseLayers_.size(); }
    size_t cnnLayerCount() const { return cnnLayers_.size(); }
    size_t denseLayerCount() const { return denseLayers_.size(); }

    const std::vector<CNNLayerPtr>& getCNNLayers() const { return cnnLayers_; }
    const std::vector<Layer>& getDenseLayers() const { return denseLayers_; }

    size_t inputChannels() const { return inputChannels_; }
    size_t inputHeight() const { return inputHeight_; }
    size_t inputWidth() const { return inputWidth_; }

    size_t totalParameters() const;

    // 可视化支持
    std::vector<Tensor> getAllFeatureMaps() const;
    std::vector<std::vector<Tensor>> getAllKernels() const;
    std::vector<std::string> getLayerDescriptions() const;

    std::mutex& getMutex() { return mutex_; }

private:
    std::vector<double> forwardInternal(const Tensor& input);
    void backwardInternal(const std::vector<double>& target);
    void updateWeightsInternal(double learningRate);
    double calculateLossInternal(const std::vector<double>& output,
                                 const std::vector<double>& target);

    void validateInputShape(const Tensor& input) const;

    // 输入尺寸
    size_t inputChannels_;
    size_t inputHeight_;
    size_t inputWidth_;

    // 当前输出尺寸（用于层链接）
    size_t currentChannels_;
    size_t currentHeight_;
    size_t currentWidth_;

    // CNN层
    std::vector<CNNLayerPtr> cnnLayers_;

    // 全连接层
    std::vector<Layer> denseLayers_;
    std::vector<int> denseLayerSizes_;
    std::vector<ActivationType> denseActivations_;

    bool isBuilt_ = false;
    bool hasFlatten_ = false;
    size_t flattenedSize_ = 0;

    std::vector<double> flattenedOutput_;
    std::vector<double> lastOutput_;

    mutable std::mutex mutex_;
};

#endif // CNN_NETWORK_H
