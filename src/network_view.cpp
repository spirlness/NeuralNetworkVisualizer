#include "network_view.h"
#include <QPainter>
#include <QPaintEvent>
#include <cmath>
#include <algorithm>

NetworkView::NetworkView(QWidget* parent)
    : QWidget(parent)
    , network_(nullptr) {
    setMinimumSize(400, 300);
    setStyleSheet("background-color: #1e1e2e;");
}

void NetworkView::setNetwork(NeuralNetwork* network) {
    network_ = network;
    if (network_) {
        layerSizes_ = network_->getLayerSizes();
    }
    update();
}

void NetworkView::updateView() {
    if (network_) {
        layerSizes_ = network_->getLayerSizes();
    }
    update();
}

void NetworkView::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    if (!network_ || layerSizes_.empty()) {
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter, "No network configured");
        return;
    }

    int numLayers = static_cast<int>(layerSizes_.size());
    int maxNeurons = *std::max_element(layerSizes_.begin(), layerSizes_.end());

    // 计算布局
    int totalWidth = width();
    int totalHeight = height();

    layerSpacing_ = totalWidth / (numLayers + 1);
    neuronSpacing_ = std::min(50, totalHeight / (maxNeurons + 2));
    neuronRadius_ = std::min(20, neuronSpacing_ / 3);

    // 计算每层神经元位置
    std::vector<std::vector<QPoint>> positions(numLayers);

    for (int l = 0; l < numLayers; ++l) {
        int x = layerSpacing_ * (l + 1);
        int numNeurons = layerSizes_[l];
        int startY = (totalHeight - (numNeurons - 1) * neuronSpacing_) / 2;

        for (int n = 0; n < numNeurons; ++n) {
            int y = startY + n * neuronSpacing_;
            positions[l].push_back(QPoint(x, y));
        }
    }

    // 获取权重信息
    auto allWeights = network_->getAllWeights();

    // 绘制连接
    for (int l = 0; l < numLayers - 1; ++l) {
        const auto& weights = allWeights[l];
        for (int i = 0; i < static_cast<int>(positions[l].size()); ++i) {
            for (int j = 0; j < static_cast<int>(positions[l + 1].size()); ++j) {
                double weight = 0.0;
                if (j < static_cast<int>(weights.size()) && i < static_cast<int>(weights[j].size())) {
                    weight = weights[j][i];
                }
                drawConnection(painter,
                              positions[l][i].x(), positions[l][i].y(),
                              positions[l + 1][j].x(), positions[l + 1][j].y(),
                              weight);
            }
        }
    }

    // 获取激活值
    auto layers = network_->getLayersSnapshot();

    // 绘制神经元
    for (int l = 0; l < numLayers; ++l) {
        for (int n = 0; n < static_cast<int>(positions[l].size()); ++n) {
            double activation = 0.0;
            if (l > 0 && l - 1 < static_cast<int>(layers.size())) {
                const auto& layer = layers[l - 1];
                if (n < static_cast<int>(layer.output.size())) {
                    activation = layer.output[n];
                }
            }
            drawNeuron(painter, positions[l][n].x(), positions[l][n].y(),
                      neuronRadius_, activation);
        }
    }

    // 绘制图例
    painter.setPen(Qt::white);
    painter.setFont(QFont("Arial", 10));

    int legendY = totalHeight - 60;
    painter.drawText(10, legendY, "Layer sizes:");

    QString sizesText;
    for (int i = 0; i < numLayers; ++i) {
        if (i > 0) sizesText += " → ";
        sizesText += QString::number(layerSizes_[i]);
    }
    painter.drawText(10, legendY + 20, sizesText);

    // 权重颜色图例
    painter.drawText(10, legendY + 40, "Weight: ");
    int gradientX = 70;
    for (int i = 0; i < 100; ++i) {
        double w = (i - 50) / 25.0;
        painter.setPen(getWeightColor(w));
        painter.drawLine(gradientX + i, legendY + 35, gradientX + i, legendY + 45);
    }
    painter.setPen(Qt::white);
    painter.drawText(gradientX - 15, legendY + 55, "-2");
    painter.drawText(gradientX + 85, legendY + 55, "+2");
}

void NetworkView::drawNeuron(QPainter& painter, int x, int y, int radius, double activation) {
    QColor fillColor = getActivationColor(activation);
    painter.setBrush(fillColor);
    painter.setPen(QPen(Qt::white, 2));
    painter.drawEllipse(QPoint(x, y), radius, radius);
}

void NetworkView::drawConnection(QPainter& painter, int x1, int y1, int x2, int y2, double weight) {
    QColor color = getWeightColor(weight);
    int penWidth = std::max(1, std::min(4, static_cast<int>(std::abs(weight) * 2)));
    painter.setPen(QPen(color, penWidth));
    painter.drawLine(x1 + neuronRadius_, y1, x2 - neuronRadius_, y2);
}

QColor NetworkView::getWeightColor(double weight) {
    // 红色表示负权重，绿色表示正权重
    weight = std::clamp(weight, -2.0, 2.0);
    if (weight < 0) {
        int intensity = static_cast<int>(std::abs(weight) / 2.0 * 200);
        return QColor(100 + intensity, 50, 50, 150);
    } else {
        int intensity = static_cast<int>(weight / 2.0 * 200);
        return QColor(50, 100 + intensity, 50, 150);
    }
}

QColor NetworkView::getActivationColor(double activation) {
    // 激活值从蓝色（0）到黄色（1）
    activation = std::clamp(activation, 0.0, 1.0);
    int r = static_cast<int>(activation * 255);
    int g = static_cast<int>(activation * 200);
    int b = static_cast<int>((1.0 - activation) * 150);
    return QColor(r, g, b);
}
