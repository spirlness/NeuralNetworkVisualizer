#include "cnn/flatten_layer.h"
#include <stdexcept>

FlattenLayer::FlattenLayer(size_t inputChannels, size_t inputHeight, size_t inputWidth)
    : inputChannels_(inputChannels), inputHeight_(inputHeight), inputWidth_(inputWidth) {
    flattenedSize_ = inputChannels_ * inputHeight_ * inputWidth_;
}

Tensor FlattenLayer::forward(const Tensor& input) {
    if (input.channels() != inputChannels_ ||
        input.height() != inputHeight_ ||
        input.width() != inputWidth_) {
        throw std::invalid_argument("FlattenLayer: input shape mismatch");
    }

    lastInput_ = input;

    Tensor output(1, 1, flattenedSize_);

    size_t idx = 0;
    for (size_t c = 0; c < inputChannels_; ++c) {
        for (size_t h = 0; h < inputHeight_; ++h) {
            for (size_t w = 0; w < inputWidth_; ++w) {
                output(0, 0, idx++) = input(c, h, w);
            }
        }
    }

    lastOutput_ = output;
    return output;
}

Tensor FlattenLayer::backward(const Tensor& gradOutput) {
    if (gradOutput.channels() != 1 ||
        gradOutput.height() != 1 ||
        gradOutput.width() != flattenedSize_) {
        throw std::invalid_argument("FlattenLayer: gradOutput shape mismatch");
    }

    Tensor gradInput(inputChannels_, inputHeight_, inputWidth_);

    size_t idx = 0;
    for (size_t c = 0; c < inputChannels_; ++c) {
        for (size_t h = 0; h < inputHeight_; ++h) {
            for (size_t w = 0; w < inputWidth_; ++w) {
                gradInput(c, h, w) = gradOutput(0, 0, idx++);
            }
        }
    }

    return gradInput;
}

std::vector<double> FlattenLayer::getFlattenedOutput() const {
    return lastOutput_.flatten();
}
