#include "cnn/conv_layer.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

ConvolutionalLayer::ConvolutionalLayer(size_t inputChannels, size_t inputHeight, size_t inputWidth,
                                       size_t outputChannels, size_t kernelSize,
                                       size_t stride, size_t padding,
                                       CNNActivationType activation)
    : inputChannels_(inputChannels), inputHeight_(inputHeight), inputWidth_(inputWidth),
      outputChannels_(outputChannels), kernelSize_(kernelSize),
      stride_(stride), padding_(padding), activation_(activation) {
    if (stride == 0) {
        throw std::invalid_argument("ConvolutionalLayer: stride must be greater than 0");
    }
    computeOutputSize();
    initializeWeights();
}

void ConvolutionalLayer::computeOutputSize() {
    const long long paddedH = static_cast<long long>(inputHeight_) + 2LL * static_cast<long long>(padding_);
    const long long paddedW = static_cast<long long>(inputWidth_) + 2LL * static_cast<long long>(padding_);
    const long long k = static_cast<long long>(kernelSize_);
    const long long s = static_cast<long long>(stride_);

    if (k <= 0 || s <= 0) {
        throw std::invalid_argument("ConvolutionalLayer: kernelSize and stride must be greater than 0");
    }
    if (paddedH < k || paddedW < k) {
        throw std::invalid_argument("ConvolutionalLayer: kernelSize too large for input size");
    }

    const long long outH = (paddedH - k) / s + 1;
    const long long outW = (paddedW - k) / s + 1;
    if (outH <= 0 || outW <= 0) {
        throw std::invalid_argument("ConvolutionalLayer: invalid output size");
    }

    outputHeight_ = static_cast<size_t>(outH);
    outputWidth_ = static_cast<size_t>(outW);
}

void ConvolutionalLayer::initializeWeights() {
    kernels_.resize(outputChannels_);
    kernelGradients_.resize(outputChannels_);
    biases_.resize(outputChannels_, 0.0);
    biasGradients_.resize(outputChannels_, 0.0);

    size_t fanIn = inputChannels_ * kernelSize_ * kernelSize_;
    size_t fanOut = outputChannels_ * kernelSize_ * kernelSize_;

    for (size_t oc = 0; oc < outputChannels_; ++oc) {
        kernels_[oc] = Tensor(inputChannels_, kernelSize_, kernelSize_);
        kernelGradients_[oc] = Tensor(inputChannels_, kernelSize_, kernelSize_);

        if (activation_ == CNNActivationType::ReLU || activation_ == CNNActivationType::LeakyReLU) {
            kernels_[oc].heInit(fanIn);
        } else {
            kernels_[oc].xavierInit(fanIn, fanOut);
        }
    }

    preActivation_ = Tensor(outputChannels_, outputHeight_, outputWidth_);
    lastOutput_ = Tensor(outputChannels_, outputHeight_, outputWidth_);
    outputBuffer_ = Tensor(outputChannels_, outputHeight_, outputWidth_);
    size_t paddedH = inputHeight_ + 2 * padding_;
    size_t paddedW = inputWidth_ + 2 * padding_;
    paddedInputBuffer_ = Tensor(inputChannels_, paddedH, paddedW);
}

double ConvolutionalLayer::activate(double x) const {
    switch (activation_) {
        case CNNActivationType::ReLU:
            return std::max(0.0, x);
        case CNNActivationType::LeakyReLU:
            return x > 0 ? x : 0.01 * x;
        case CNNActivationType::Sigmoid:
            return 1.0 / (1.0 + std::exp(-std::clamp(x, -500.0, 500.0)));
        case CNNActivationType::Tanh:
            return std::tanh(x);
        case CNNActivationType::None:
        default:
            return x;
    }
}

double ConvolutionalLayer::activateDerivative(double x) const {
    switch (activation_) {
        case CNNActivationType::ReLU:
            return x > 0 ? 1.0 : 0.0;
        case CNNActivationType::LeakyReLU:
            return x > 0 ? 1.0 : 0.01;
        case CNNActivationType::Sigmoid: {
            double s = 1.0 / (1.0 + std::exp(-std::clamp(x, -500.0, 500.0)));
            return s * (1.0 - s);
        }
        case CNNActivationType::Tanh: {
            double t = std::tanh(x);
            return 1.0 - t * t;
        }
        case CNNActivationType::None:
        default:
            return 1.0;
    }
}

Tensor ConvolutionalLayer::forward(const Tensor& input) {
    if (input.channels() != inputChannels_ ||
        input.height() != inputHeight_ ||
        input.width() != inputWidth_) {
        throw std::invalid_argument("ConvolutionalLayer: input shape mismatch");
    }

    lastInput_ = input;

    const Tensor* paddedInput = &input;
    if (padding_ > 0) {
        paddedInputBuffer_ = input.pad(padding_, padding_, 0.0);
        paddedInput = &paddedInputBuffer_;
    }

    for (size_t oc = 0; oc < outputChannels_; ++oc) {
        const Tensor& kernel = kernels_[oc];
        double bias = biases_[oc];

        for (size_t oh = 0; oh < outputHeight_; ++oh) {
            for (size_t ow = 0; ow < outputWidth_; ++ow) {
                double sum = bias;
                size_t base_ih = oh * stride_;
                size_t base_iw = ow * stride_;

                for (size_t ic = 0; ic < inputChannels_; ++ic) {
                    for (size_t kh = 0; kh < kernelSize_; ++kh) {
                        size_t ih = base_ih + kh;
                        #if defined(_MSC_VER)
                        #pragma loop(hint_parallel(4))
                        #endif
                        for (size_t kw = 0; kw < kernelSize_; ++kw) {
                            size_t iw = base_iw + kw;
                            sum += (*paddedInput)(ic, ih, iw) * kernel(ic, kh, kw);
                        }
                    }
                }

                preActivation_(oc, oh, ow) = sum;
                outputBuffer_(oc, oh, ow) = activate(sum);
            }
        }
    }

    lastOutput_ = outputBuffer_;
    return lastOutput_;
}

Tensor ConvolutionalLayer::backward(const Tensor& gradOutput) {
    if (gradOutput.channels() != outputChannels_ ||
        gradOutput.height() != outputHeight_ ||
        gradOutput.width() != outputWidth_) {
        throw std::invalid_argument("ConvolutionalLayer: gradOutput shape mismatch");
    }

    for (auto& kg : kernelGradients_) kg.zero();
    std::fill(biasGradients_.begin(), biasGradients_.end(), 0.0);

    Tensor gradInput(inputChannels_, inputHeight_, inputWidth_);
    gradInput.zero();

    Tensor delta(outputChannels_, outputHeight_, outputWidth_);
    for (size_t oc = 0; oc < outputChannels_; ++oc) {
        for (size_t oh = 0; oh < outputHeight_; ++oh) {
            for (size_t ow = 0; ow < outputWidth_; ++ow) {
                delta(oc, oh, ow) = gradOutput(oc, oh, ow) *
                                   activateDerivative(preActivation_(oc, oh, ow));
            }
        }
    }

    Tensor paddedInput = lastInput_;
    if (padding_ > 0) {
        paddedInput = lastInput_.pad(padding_, padding_, 0.0);
    }

    for (size_t oc = 0; oc < outputChannels_; ++oc) {
        for (size_t oh = 0; oh < outputHeight_; ++oh) {
            for (size_t ow = 0; ow < outputWidth_; ++ow) {
                biasGradients_[oc] += delta(oc, oh, ow);
            }
        }

        for (size_t ic = 0; ic < inputChannels_; ++ic) {
            for (size_t kh = 0; kh < kernelSize_; ++kh) {
                for (size_t kw = 0; kw < kernelSize_; ++kw) {
                    double grad = 0.0;
                    for (size_t oh = 0; oh < outputHeight_; ++oh) {
                        for (size_t ow = 0; ow < outputWidth_; ++ow) {
                            size_t ih = oh * stride_ + kh;
                            size_t iw = ow * stride_ + kw;
                            grad += delta(oc, oh, ow) * paddedInput(ic, ih, iw);
                        }
                    }
                    kernelGradients_[oc](ic, kh, kw) += grad;
                }
            }
        }

        for (size_t oh = 0; oh < outputHeight_; ++oh) {
            for (size_t ow = 0; ow < outputWidth_; ++ow) {
                for (size_t ic = 0; ic < inputChannels_; ++ic) {
                    for (size_t kh = 0; kh < kernelSize_; ++kh) {
                        for (size_t kw = 0; kw < kernelSize_; ++kw) {
                            int ih = static_cast<int>(oh * stride_ + kh) - static_cast<int>(padding_);
                            int iw = static_cast<int>(ow * stride_ + kw) - static_cast<int>(padding_);

                            if (ih >= 0 && ih < static_cast<int>(inputHeight_) &&
                                iw >= 0 && iw < static_cast<int>(inputWidth_)) {
                                gradInput(ic, static_cast<size_t>(ih), static_cast<size_t>(iw)) +=
                                    delta(oc, oh, ow) * kernels_[oc](ic, kh, kw);
                            }
                        }
                    }
                }
            }
        }
    }

    return gradInput;
}

void ConvolutionalLayer::updateWeights(double learningRate) {
    for (size_t oc = 0; oc < outputChannels_; ++oc) {
        for (size_t ic = 0; ic < inputChannels_; ++ic) {
            for (size_t kh = 0; kh < kernelSize_; ++kh) {
                for (size_t kw = 0; kw < kernelSize_; ++kw) {
                    kernels_[oc](ic, kh, kw) -= learningRate * kernelGradients_[oc](ic, kh, kw);
                }
            }
        }
        biases_[oc] -= learningRate * biasGradients_[oc];
    }
}

size_t ConvolutionalLayer::parameterCount() const {
    return outputChannels_ * (inputChannels_ * kernelSize_ * kernelSize_ + 1);
}

Tensor ConvolutionalLayer::getKernel(size_t outputChannel) const {
    if (outputChannel >= outputChannels_) {
        throw std::out_of_range("Kernel index out of range");
    }
    return kernels_[outputChannel];
}
