#include "cnn/tensor.h"
#include "cnn/random.h"
#include <limits>
#include <string>
#include <cstring>

Tensor::Tensor() : channels_(0), height_(0), width_(0) {}

Tensor::Tensor(size_t channels, size_t height, size_t width)
    : channels_(channels), height_(height), width_(width),
      data_(channels * height * width, 0.0) {}

Tensor::Tensor(size_t channels, size_t height, size_t width, double initValue)
    : channels_(channels), height_(height), width_(width),
      data_(channels * height * width, initValue) {}

double& Tensor::at(size_t c, size_t h, size_t w) {
    if (c >= channels_ || h >= height_ || w >= width_) {
        throw std::out_of_range("Tensor index out of range");
    }
    return data_[index(c, h, w)];
}

const double& Tensor::at(size_t c, size_t h, size_t w) const {
    if (c >= channels_ || h >= height_ || w >= width_) {
        throw std::out_of_range("Tensor index out of range");
    }
    return data_[index(c, h, w)];
}

double& Tensor::operator()(size_t c, size_t h, size_t w) {
    return data_[index(c, h, w)];
}

const double& Tensor::operator()(size_t c, size_t h, size_t w) const {
    return data_[index(c, h, w)];
}

std::vector<double> Tensor::getChannel(size_t c) const {
    if (c >= channels_) {
        throw std::out_of_range("Channel index out of range");
    }
    std::vector<double> channel(height_ * width_);
    const size_t offsetSize = c * height_ * width_;
    const size_t countSize = height_ * width_;
    if (offsetSize > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        countSize > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        offsetSize + countSize > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::out_of_range("Tensor offset out of range");
    }
    const auto offset = static_cast<std::ptrdiff_t>(offsetSize);
    const auto count = static_cast<std::ptrdiff_t>(countSize);
    std::copy(data_.begin() + offset, data_.begin() + offset + count, channel.begin());
    return channel;
}

void Tensor::setChannel(size_t c, const std::vector<double>& data) {
    if (c >= channels_ || data.size() != height_ * width_) {
        throw std::invalid_argument("Invalid channel data");
    }
    const size_t offsetSize = c * height_ * width_;
    if (offsetSize > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::out_of_range("Tensor offset out of range");
    }
    const auto offset = static_cast<std::ptrdiff_t>(offsetSize);
    std::copy(data.begin(), data.end(), data_.begin() + offset);
}

void Tensor::resize(size_t channels, size_t height, size_t width) {
    channels_ = channels;
    height_ = height;
    width_ = width;
    data_.resize(channels * height * width);
    std::fill(data_.begin(), data_.end(), 0.0);
}

void Tensor::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::randomInit(double min, double max) {
    auto& gen = getRng();
    std::uniform_real_distribution<> dis(min, max);
    for (auto& v : data_) {
        v = dis(gen);
    }
}

void Tensor::xavierInit(size_t fanIn, size_t fanOut) {
    if (fanIn == 0 || fanOut == 0) {
        throw std::invalid_argument("Xavier initialization: fanIn and fanOut must be greater than 0");
    }
    auto& gen = getRng();
    double limit = std::sqrt(6.0 / (static_cast<double>(fanIn) + static_cast<double>(fanOut)));
    std::uniform_real_distribution<> dis(-limit, limit);
    for (auto& v : data_) {
        v = dis(gen);
    }
}

void Tensor::heInit(size_t fanIn) {
    if (fanIn == 0) {
        throw std::invalid_argument("He initialization: fanIn must be greater than 0");
    }
    auto& gen = getRng();
    double stddev = std::sqrt(2.0 / static_cast<double>(fanIn));
    std::normal_distribution<> dis(0.0, stddev);
    for (auto& v : data_) {
        v = dis(gen);
    }
}

Tensor Tensor::operator+(const Tensor& other) const {
    Tensor result(*this);
    result += other;
    return result;
}

void Tensor::pad(Tensor& destination, size_t padHeight, size_t padWidth, double padValue) const {
    size_t newHeight = height_ + 2 * padHeight;
    size_t newWidth = width_ + 2 * padWidth;

    if (destination.channels() != channels_ ||
        destination.height() != newHeight ||
        destination.width() != newWidth) {
        destination.resize(channels_, newHeight, newWidth);
    }

    double* destData = destination.rawData();
    const double* srcData = rawData();
    size_t srcChannelStride = height_ * width_;
    size_t destChannelStride = newHeight * newWidth;

    for (size_t c = 0; c < channels_; ++c) {
        double* destChannel = destData + c * destChannelStride;
        const double* srcChannel = srcData + c * srcChannelStride;

        // Fill top border
        if (padHeight > 0) {
            std::fill(destChannel, destChannel + padHeight * newWidth, padValue);
        }

        // Process middle rows
        for (size_t h = 0; h < height_; ++h) {
            double* destRow = destChannel + (h + padHeight) * newWidth;
            const double* srcRow = srcChannel + h * width_;

            // Fill left border
            if (padWidth > 0) {
                std::fill(destRow, destRow + padWidth, padValue);
            }

            // Copy input row
            if (width_ > 0) {
                std::memcpy(destRow + padWidth, srcRow, width_ * sizeof(double));
            }

            // Fill right border
            if (padWidth > 0) {
                std::fill(destRow + padWidth + width_, destRow + newWidth, padValue);
            }
        }

        // Fill bottom border
        if (padHeight > 0) {
            std::fill(destChannel + (height_ + padHeight) * newWidth, destChannel + destChannelStride, padValue);
        }
    }
}

Tensor Tensor::operator-(const Tensor& other) const {
    Tensor result(*this);
    result -= other;
    return result;
}

Tensor Tensor::operator*(double scalar) const {
    Tensor result(*this);
    result *= scalar;
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator*=(double scalar) {
    for (auto& v : data_) {
        v *= scalar;
    }
    return *this;
}

void Tensor::apply(const std::function<double(double)>& func) {
    for (auto& v : data_) {
        v = func(v);
    }
}

Tensor Tensor::map(const std::function<double(double)>& func) const {
    Tensor result(*this);
    result.apply(func);
    return result;
}

double Tensor::sum() const {
    return std::accumulate(data_.begin(), data_.end(), 0.0);
}

double Tensor::mean() const {
    if (data_.empty()) return 0.0;
    return sum() / static_cast<double>(data_.size());
}

double Tensor::max() const {
    if (data_.empty()) return 0.0;
    return *std::max_element(data_.begin(), data_.end());
}

double Tensor::min() const {
    if (data_.empty()) return 0.0;
    return *std::min_element(data_.begin(), data_.end());
}

std::vector<double> Tensor::flatten() const {
    return data_;
}

Tensor Tensor::fromVector(const std::vector<double>& vec,
                          size_t channels, size_t height, size_t width) {
    Tensor t(channels, height, width);
    if (vec.size() != channels * height * width) {
        throw std::invalid_argument("Vector size mismatch");
    }
    t.data_ = vec;
    return t;
}

Tensor Tensor::pad(size_t padHeight, size_t padWidth, double padValue) const {
    // Create result with correct dimensions.
    // Note: The constructor initializes with padValue, but our optimized pad()
    // will overwrite the necessary parts efficiently.
    // Using a default 0.0 initialization might be slightly faster if padValue is complex,
    // but here we just want to ensure size is correct.
    Tensor result(channels_, height_ + 2 * padHeight, width_ + 2 * padWidth, 0.0);
    pad(result, padHeight, padWidth, padValue);
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (channels_ != other.channels_) {
        throw std::invalid_argument("Channel mismatch in matmul: " + std::to_string(channels_) + " vs " + std::to_string(other.channels_));
    }
    if (width_ != other.height_) {
        throw std::invalid_argument("Dimension mismatch in matmul: " + std::to_string(width_) + " vs " + std::to_string(other.height_));
    }

    Tensor result(channels_, height_, other.width_);
    for (size_t c = 0; c < channels_; ++c) {
        for (size_t i = 0; i < height_; ++i) {
            for (size_t k = 0; k < width_; ++k) {
                double val = at(c, i, k);
                for (size_t j = 0; j < other.width_; ++j) {
                    result(c, i, j) += val * other(c, k, j);
                }
            }
        }
    }
    return result;
}

Tensor Tensor::transpose() const {
    Tensor result(channels_, width_, height_);
    for (size_t c = 0; c < channels_; ++c) {
        for (size_t h = 0; h < height_; ++h) {
            for (size_t w = 0; w < width_; ++w) {
                result(c, w, h) = at(c, h, w);
            }
        }
    }
    return result;
}

void Tensor::softmax() {
    for (size_t c = 0; c < channels_; ++c) {
        for (size_t h = 0; h < height_; ++h) {
            double maxVal = -std::numeric_limits<double>::infinity();
            for (size_t w = 0; w < width_; ++w) {
                maxVal = std::max(maxVal, at(c, h, w));
            }

            double sumExp = 0.0;
            for (size_t w = 0; w < width_; ++w) {
                double val = std::exp(at(c, h, w) - maxVal);
                at(c, h, w) = val;
                sumExp += val;
            }

            for (size_t w = 0; w < width_; ++w) {
                at(c, h, w) /= sumExp;
            }
        }
    }
}

Tensor Tensor::randn(size_t c, size_t h, size_t w) {
    Tensor t(c, h, w);
    auto& gen = getRng();
    std::normal_distribution<> dis(0.0, 1.0);
    for (auto& v : t.data_) {
        v = dis(gen);
    }
    return t;
}
