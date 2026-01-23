#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <functional>
#include <cmath>
#include <numeric>

/**
 * @brief 3D张量类，用于表示CNN中的特征图
 *
 * 存储格式: [channels][height][width] (CHW格式)
 */
class Tensor {
public:
    Tensor();
    Tensor(size_t channels, size_t height, size_t width);
    Tensor(size_t channels, size_t height, size_t width, double initValue);

    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    // 维度访问
    size_t channels() const { return channels_; }
    size_t height() const { return height_; }
    size_t width() const { return width_; }
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

    // 元素访问 (CHW索引)
    double& at(size_t c, size_t h, size_t w);
    const double& at(size_t c, size_t h, size_t w) const;

    // 快速访问（无边界检查）
    double& operator()(size_t c, size_t h, size_t w);
    const double& operator()(size_t c, size_t h, size_t w) const;

    // 获取单个通道
    std::vector<double> getChannel(size_t c) const;
    void setChannel(size_t c, const std::vector<double>& data);

    // 数据访问
    std::vector<double>& data() { return data_; }
    const std::vector<double>& data() const { return data_; }
    double* rawData() { return data_.data(); }
    const double* rawData() const { return data_.data(); }

    // 形状操作
    void resize(size_t channels, size_t height, size_t width);
    void fill(double value);
    void zero() { fill(0.0); }

    // 初始化方法
    void randomInit(double min = -1.0, double max = 1.0);
    void xavierInit(size_t fanIn, size_t fanOut);
    void heInit(size_t fanIn);

    // 数学运算
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(double scalar) const;
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(double scalar);

    // 逐元素运算
    void apply(const std::function<double(double)>& func);
    Tensor map(const std::function<double(double)>& func) const;

    // 统计函数
    double sum() const;
    double mean() const;
    double max() const;
    double min() const;

    // 展平为1D向量
    std::vector<double> flatten() const;
    static Tensor fromVector(const std::vector<double>& vec,
                             size_t channels, size_t height, size_t width);

    // 填充操作
    Tensor pad(size_t padHeight, size_t padWidth, double padValue = 0.0) const;
    void pad(Tensor& destination, size_t padHeight, size_t padWidth, double padValue = 0.0) const;

    // 矩阵运算 (Added for Attention)
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    void softmax();
    static Tensor randn(size_t c, size_t h, size_t w);

private:
    size_t channels_;
    size_t height_;
    size_t width_;
    std::vector<double> data_;

    size_t index(size_t c, size_t h, size_t w) const {
        return c * height_ * width_ + h * width_ + w;
    }
};

#endif // TENSOR_H
