#include "cnn/pooling_layer.h"
#include <limits>
#include <algorithm>
#include <stdexcept>

PoolingLayer::PoolingLayer(size_t inputChannels, size_t inputHeight, size_t inputWidth,
                           size_t poolSize, size_t stride, PoolingType poolType)
    : inputChannels_(inputChannels), inputHeight_(inputHeight), inputWidth_(inputWidth),
      poolSize_(poolSize), stride_(stride), poolType_(poolType) {
    if (poolSize == 0) {
        throw std::invalid_argument("PoolingLayer: poolSize must be greater than 0");
    }
    if (stride == 0) {
        throw std::invalid_argument("PoolingLayer: stride must be greater than 0");
    }
    computeOutputSize();
}

void PoolingLayer::computeOutputSize() {
    const long long inH = static_cast<long long>(inputHeight_);
    const long long inW = static_cast<long long>(inputWidth_);
    const long long p = static_cast<long long>(poolSize_);
    const long long s = static_cast<long long>(stride_);

    if (p <= 0 || s <= 0) {
        throw std::invalid_argument("PoolingLayer: poolSize and stride must be greater than 0");
    }
    if (inH < p || inW < p) {
        throw std::invalid_argument("PoolingLayer: poolSize too large for input size");
    }

    const long long outH = (inH - p) / s + 1;
    const long long outW = (inW - p) / s + 1;
    if (outH <= 0 || outW <= 0) {
        throw std::invalid_argument("PoolingLayer: invalid output size");
    }

    outputHeight_ = static_cast<size_t>(outH);
    outputWidth_ = static_cast<size_t>(outW);
}

Tensor PoolingLayer::forward(const Tensor& input) {
    if (input.channels() != inputChannels_ ||
        input.height() != inputHeight_ ||
        input.width() != inputWidth_) {
        throw std::invalid_argument("PoolingLayer: input shape mismatch");
    }

    lastInput_ = input;

    Tensor output(inputChannels_, outputHeight_, outputWidth_);

    if (poolType_ == PoolingType::Max) {
        maxIndices_.resize(inputChannels_);
        for (size_t c = 0; c < inputChannels_; ++c) {
            maxIndices_[c].resize(outputHeight_);
            for (size_t oh = 0; oh < outputHeight_; ++oh) {
                maxIndices_[c][oh].resize(outputWidth_);
            }
        }
    }

    for (size_t c = 0; c < inputChannels_; ++c) {
        for (size_t oh = 0; oh < outputHeight_; ++oh) {
            for (size_t ow = 0; ow < outputWidth_; ++ow) {
                size_t startH = oh * stride_;
                size_t startW = ow * stride_;

                if (poolType_ == PoolingType::Max) {
                    double maxVal = std::numeric_limits<double>::lowest();
                    size_t maxH = startH, maxW = startW;

                    for (size_t ph = 0; ph < poolSize_; ++ph) {
                        for (size_t pw = 0; pw < poolSize_; ++pw) {
                            size_t ih = startH + ph;
                            size_t iw = startW + pw;
                            if (ih < inputHeight_ && iw < inputWidth_) {
                                double val = input(c, ih, iw);
                                if (val > maxVal) {
                                    maxVal = val;
                                    maxH = ih;
                                    maxW = iw;
                                }
                            }
                        }
                    }

                    output(c, oh, ow) = maxVal;
                    maxIndices_[c][oh][ow] = {maxH, maxW};

                } else {
                    double sum = 0.0;
                    int count = 0;

                    for (size_t ph = 0; ph < poolSize_; ++ph) {
                        for (size_t pw = 0; pw < poolSize_; ++pw) {
                            size_t ih = startH + ph;
                            size_t iw = startW + pw;
                            if (ih < inputHeight_ && iw < inputWidth_) {
                                sum += input(c, ih, iw);
                                count++;
                            }
                        }
                    }

                    output(c, oh, ow) = count > 0 ? sum / count : 0.0;
                }
            }
        }
    }

    lastOutput_ = output;
    return output;
}

Tensor PoolingLayer::backward(const Tensor& gradOutput) {
    if (gradOutput.channels() != inputChannels_ ||
        gradOutput.height() != outputHeight_ ||
        gradOutput.width() != outputWidth_) {
        throw std::invalid_argument("PoolingLayer: gradOutput shape mismatch");
    }

    Tensor gradInput(inputChannels_, inputHeight_, inputWidth_);
    gradInput.zero();

    for (size_t c = 0; c < inputChannels_; ++c) {
        for (size_t oh = 0; oh < outputHeight_; ++oh) {
            for (size_t ow = 0; ow < outputWidth_; ++ow) {
                if (poolType_ == PoolingType::Max) {
                    auto [maxH, maxW] = maxIndices_[c][oh][ow];
                    gradInput(c, maxH, maxW) += gradOutput(c, oh, ow);

                } else {
                    size_t startH = oh * stride_;
                    size_t startW = ow * stride_;

                    int count = 0;
                    for (size_t ph = 0; ph < poolSize_; ++ph) {
                        for (size_t pw = 0; pw < poolSize_; ++pw) {
                            size_t ih = startH + ph;
                            size_t iw = startW + pw;
                            if (ih < inputHeight_ && iw < inputWidth_) {
                                count++;
                            }
                        }
                    }

                    double avgGrad = gradOutput(c, oh, ow) / count;
                    for (size_t ph = 0; ph < poolSize_; ++ph) {
                        for (size_t pw = 0; pw < poolSize_; ++pw) {
                            size_t ih = startH + ph;
                            size_t iw = startW + pw;
                            if (ih < inputHeight_ && iw < inputWidth_) {
                                gradInput(c, ih, iw) += avgGrad;
                            }
                        }
                    }
                }
            }
        }
    }

    return gradInput;
}
