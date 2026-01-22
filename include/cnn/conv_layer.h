#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "cnn/cnn_layer_base.h"
#include <vector>

/**
 * @brief 2D卷积层实现
 */
class ConvolutionalLayer : public CNNLayerBase {
public:
    ConvolutionalLayer(size_t inputChannels, size_t inputHeight, size_t inputWidth,
                       size_t outputChannels, size_t kernelSize,
                       size_t stride = 1, size_t padding = 0,
                       CNNActivationType activation = CNNActivationType::ReLU);

    // CNNLayerBase 接口实现
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradOutput) override;
    void updateWeights(double learningRate) override;

    CNNLayerType type() const override { return CNNLayerType::Convolutional; }
    std::string name() const override { return "Conv2D"; }

    size_t inputChannels() const override { return inputChannels_; }
    size_t inputHeight() const override { return inputHeight_; }
    size_t inputWidth() const override { return inputWidth_; }

    size_t outputChannels() const override { return outputChannels_; }
    size_t outputHeight() const override { return outputHeight_; }
    size_t outputWidth() const override { return outputWidth_; }

    size_t parameterCount() const override;
    bool hasTrainableParams() const override { return true; }

    const Tensor& getOutput() const override { return lastOutput_; }
    std::vector<Tensor> getWeights() const override { return kernels_; }
    std::vector<double> getBiases() const override { return biases_; }

    // 卷积层特有方法
    size_t kernelSize() const { return kernelSize_; }
    size_t stride() const { return stride_; }
    size_t padding() const { return padding_; }
    CNNActivationType activationType() const { return activation_; }

    Tensor getKernel(size_t outputChannel) const;

private:
    void initializeWeights();
    void computeOutputSize();

    double activate(double x) const;
    double activateDerivative(double x) const;

    size_t inputChannels_;
    size_t inputHeight_;
    size_t inputWidth_;
    size_t outputChannels_;
    size_t kernelSize_;
    size_t stride_;
    size_t padding_;
    CNNActivationType activation_;

    size_t outputHeight_;
    size_t outputWidth_;

    std::vector<Tensor> kernels_;
    std::vector<double> biases_;

    std::vector<Tensor> kernelGradients_;
    std::vector<double> biasGradients_;

    Tensor preActivation_;
    Tensor lastOutput_;
    Tensor lastInput_;
    Tensor paddedInputBuffer_;
    Tensor outputBuffer_;
};

#endif // CONV_LAYER_H
