#ifndef CNN_LAYER_BASE_H
#define CNN_LAYER_BASE_H

#include "cnn/tensor.h"
#include <string>
#include <memory>
#include <vector>

/**
 * @brief CNN层类型枚举
 */
enum class CNNLayerType {
    Convolutional,
    MaxPooling,
    AvgPooling,
    Flatten,
    FullyConnected,
    BatchNorm,
    Dropout
};

/**
 * @brief 激活函数类型，未来可扩展
 */
enum class CNNActivationType {
    None,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    Softmax
};

/**
 * @brief CNN层基类 - 定义CNN的公共接口
 */
class CNNLayerBase {
public:
    virtual ~CNNLayerBase() = default;

    // ========== 核心接口 ==========

    /**
     * @brief 前向传播
     * @param input 输入张量
     * @return 输出张量
     */
    virtual Tensor forward(const Tensor& input) = 0;

    /**
     * @brief 反向传播
     * @param gradOutput 输出梯度
     * @return 输入梯度
     */
    virtual Tensor backward(const Tensor& gradOutput) = 0;

    /**
     * @brief 更新权重，如果有可训练参数
     * @param learningRate 学习率
     */
    virtual void updateWeights(double learningRate) = 0;

    // ========== 属性访问 ==========

    virtual CNNLayerType type() const = 0;
    virtual std::string name() const = 0;

    // 输入输出维度
    virtual size_t inputChannels() const = 0;
    virtual size_t inputHeight() const = 0;
    virtual size_t inputWidth() const = 0;

    virtual size_t outputChannels() const = 0;
    virtual size_t outputHeight() const = 0;
    virtual size_t outputWidth() const = 0;

    // 参数统计
    virtual size_t parameterCount() const = 0;
    virtual bool hasTrainableParams() const = 0;

    // ========== 可视化支持 ==========

    /**
     * @brief 获取最后一次前向传播输出，用于可视化
     */
    virtual const Tensor& getOutput() const = 0;

    /**
     * @brief 获取最后一次前向传播输入
     */
    virtual const Tensor& getInput() const { return lastInput_; }

    /**
     * @brief 获取可训练权重，用于可视化卷积核
     */
    virtual std::vector<Tensor> getWeights() const { return {}; }

    /**
     * @brief 获取偏置
     */
    virtual std::vector<double> getBiases() const { return {}; }

protected:
    Tensor lastOutput_;  // 保存输出
    Tensor lastInput_;   // 保存输入（反向传播需要）
};

// 使用智能指针管理
using CNNLayerPtr = std::shared_ptr<CNNLayerBase>;

#endif // CNN_LAYER_BASE_H
