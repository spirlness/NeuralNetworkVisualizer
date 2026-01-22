#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "cnn/cnn_layer_base.h"
#include <vector>
#include <utility>

/**
 * @brief 池化类型
 */
enum class PoolingType {
    Max,
    Average
};

/**
 * @brief 池化层实现（MaxPool / AvgPool）
 */
class PoolingLayer : public CNNLayerBase {
public:
    PoolingLayer(size_t inputChannels, size_t inputHeight, size_t inputWidth,
                 size_t poolSize, size_t stride,
                 PoolingType poolType = PoolingType::Max);

    // CNNLayerBase 接口实现
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
    void updateWeights(double /*learningRate*/) override { /* no-op */ }

    CNNLayerType type() const override {
        return poolType_ == PoolingType::Max ?
               CNNLayerType::MaxPooling : CNNLayerType::AvgPooling;
    }
    std::string name() const override {
        return poolType_ == PoolingType::Max ? "MaxPool2D" : "AvgPool2D";
    }

    size_t inputChannels() const override { return inputChannels_; }
    size_t inputHeight() const override { return inputHeight_; }
    size_t inputWidth() const override { return inputWidth_; }

    size_t outputChannels() const override { return inputChannels_; }
    size_t outputHeight() const override { return outputHeight_; }
    size_t outputWidth() const override { return outputWidth_; }

    size_t parameterCount() const override { return 0; }
    bool hasTrainableParams() const override { return false; }

    const Tensor& getOutput() const override { return lastOutput_; }
    const Tensor& getInput() const override { return lastInput_; }

    // 池化层特有方法
    PoolingType poolingType() const { return poolType_; }
    size_t poolSize() const { return poolSize_; }

private:
    void computeOutputSize();

    size_t inputChannels_;
    size_t inputHeight_;
    size_t inputWidth_;
    size_t poolSize_;
    size_t stride_;
    PoolingType poolType_;

    size_t outputHeight_;
    size_t outputWidth_;

    // MaxPool需要记录最大值位置（用于反向传播）
    std::vector<std::vector<std::vector<std::pair<size_t, size_t>>>> maxIndices_;
};

#endif // POOLING_LAYER_H
