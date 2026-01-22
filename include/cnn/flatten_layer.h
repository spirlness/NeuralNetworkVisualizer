#ifndef FLATTEN_LAYER_H
#define FLATTEN_LAYER_H

#include "cnn/cnn_layer_base.h"

/**
 * @brief 展平层 - 将3D张量展平为1D向量
 *
 * 用于连接卷积层与全连接层
 */
class FlattenLayer : public CNNLayerBase {
public:
    FlattenLayer(size_t inputChannels, size_t inputHeight, size_t inputWidth);

    // CNNLayerBase 接口实现
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
    void updateWeights(double /*learningRate*/) override { /* no-op */ }

    CNNLayerType type() const override { return CNNLayerType::Flatten; }
    std::string name() const override { return "Flatten"; }

    size_t inputChannels() const override { return inputChannels_; }
    size_t inputHeight() const override { return inputHeight_; }
    size_t inputWidth() const override { return inputWidth_; }

    size_t outputChannels() const override { return 1; }
    size_t outputHeight() const override { return 1; }
    size_t outputWidth() const override { return flattenedSize_; }

    size_t parameterCount() const override { return 0; }
    bool hasTrainableParams() const override { return false; }

    const Tensor& getOutput() const override { return lastOutput_; }

    size_t flattenedSize() const { return flattenedSize_; }

    std::vector<double> getFlattenedOutput() const;

private:
    size_t inputChannels_;
    size_t inputHeight_;
    size_t inputWidth_;
    size_t flattenedSize_;
};

#endif // FLATTEN_LAYER_H
