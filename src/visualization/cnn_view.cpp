#include "visualization/cnn_view.h"
#include <cmath>
#include <algorithm>
#include <mutex>

CNNView::CNNView(QWidget* parent)
    : QWidget(parent) {
    setMinimumSize(600, 300);
    setStyleSheet("background-color: #1e1e2e;");
}

void CNNView::setNetwork(CNNNetwork* network) {
    network_ = network;
    update();
}

void CNNView::updateView() {
    update();
}

void CNNView::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    if (!network_) {
        painter.setPen(Qt::gray);
        painter.drawText(rect(), Qt::AlignCenter, "No CNN Network");
        return;
    }

    layerRects_.clear();

    std::lock_guard<std::mutex> lock(network_->getMutex());
    const auto& cnnLayers = network_->getCNNLayers();
    const auto& denseLayers = network_->getDenseLayers();

    int totalLayers = 1 + static_cast<int>(cnnLayers.size()) + static_cast<int>(denseLayers.size());
    layerSpacing_ = std::max(80, (width() - 100) / (totalLayers + 1));

    int centerY = height() / 2;
    int currentX = 50;

    // 绘制输入层
    drawInputLayer(painter, currentX, centerY);
    layerRects_.push_back(QRect(currentX - 30, centerY - 60, 60, 120));
    currentX += layerSpacing_;

    // 绘制CNN层
    for (size_t i = 0; i < cnnLayers.size(); ++i) {
        // 绘制连接
        drawConnection(painter, currentX - layerSpacing_ + 30, centerY,
                      currentX - 30, centerY);

        drawCNNLayer(painter, static_cast<int>(i), currentX, centerY);
        layerRects_.push_back(QRect(currentX - 30, centerY - 60, 60, 120));
        currentX += layerSpacing_;
    }

    // 绘制全连接层
    for (size_t i = 0; i < denseLayers.size(); ++i) {
        drawConnection(painter, currentX - layerSpacing_ + 30, centerY,
                      currentX - 20, centerY);

        drawDenseLayer(painter, static_cast<int>(i), currentX, centerY);
        layerRects_.push_back(QRect(currentX - 20, centerY - 50, 40, 100));
        currentX += layerSpacing_;
    }
}

void CNNView::drawInputLayer(QPainter& painter, int x, int y) {
    int w = 50;
    int h = 60;
    int d = 10;

    draw3DBox(painter, x - w/2, y - h/2, w, h, d, QColor(100, 150, 200));

    if (showLayerInfo_) {
        painter.setPen(Qt::white);
        QFont font = painter.font();
        font.setPointSize(8);
        painter.setFont(font);

        QString info = QString("%1x%2x%3")
            .arg(network_->inputChannels())
            .arg(network_->inputHeight())
            .arg(network_->inputWidth());
        painter.drawText(x - 30, y + h/2 + 15, 60, 20, Qt::AlignCenter, info);
        painter.drawText(x - 30, y - h/2 - 20, 60, 20, Qt::AlignCenter, "Input");
    }
}

void CNNView::drawCNNLayer(QPainter& painter, int layerIndex, int x, int y) {
    const auto& layer = network_->getCNNLayers()[layerIndex];

    // 根据输出尺寸计算3D盒子大小
    int channels = static_cast<int>(layer->outputChannels());
    int h = static_cast<int>(layer->outputHeight());
    int w = static_cast<int>(layer->outputWidth());

    // 缩放到可视化尺寸
    int boxW = std::max(20, std::min(60, w * 2));
    int boxH = std::max(30, std::min(80, h * 2));
    int boxD = std::max(10, std::min(30, channels / 2));

    QColor color = getLayerColor(layer->type());
    draw3DBox(painter, x - boxW/2, y - boxH/2, boxW, boxH, boxD, color);

    if (showLayerInfo_) {
        painter.setPen(Qt::white);
        QFont font = painter.font();
        font.setPointSize(8);
        painter.setFont(font);

        QString info = QString("%1x%2x%3")
            .arg(channels).arg(h).arg(w);
        painter.drawText(x - 40, y + boxH/2 + 15, 80, 20, Qt::AlignCenter, info);

        QString name = QString::fromStdString(layer->name());
        painter.drawText(x - 40, y - boxH/2 - 20, 80, 20, Qt::AlignCenter, name);
    }

    // 绘制特征图缩略图
    if (showFeatureMaps_ && !layer->getOutput().empty()) {
        const Tensor& output = layer->getOutput();
        int thumbSize = 12;
        int maxThumbs = std::min(4, static_cast<int>(output.channels()));

        for (int c = 0; c < maxThumbs; ++c) {
            int tx = x + boxW/2 + 5 + (c % 2) * (thumbSize + 2);
            int ty = y - boxH/4 + (c / 2) * (thumbSize + 2);

            // 绘制特征图缩略图
            double maxVal = output.max();
            double minVal = output.min();
            double range = (maxVal - minVal > 0.001) ? (maxVal - minVal) : 1.0;

            for (int i = 0; i < thumbSize; ++i) {
                for (int j = 0; j < thumbSize; ++j) {
                    int oh = i * output.height() / thumbSize;
                    int ow = j * output.width() / thumbSize;
                    double val = (output(c, oh, ow) - minVal) / range;
                    QColor pixelColor = getActivationColor(val);
                    painter.setPen(pixelColor);
                    painter.drawPoint(tx + j, ty + i);
                }
            }

            painter.setPen(QColor(100, 100, 120));
            painter.drawRect(tx, ty, thumbSize, thumbSize);
        }
    }
}

void CNNView::drawDenseLayer(QPainter& painter, int layerIndex, int x, int y) {
    const auto& layer = network_->getDenseLayers()[layerIndex];

    int neurons = layer.outputSize;
    int displayNeurons = std::min(10, neurons);
    int radius = 8;
    int spacing = std::min(15, (maxHeight_ - 40) / displayNeurons);

    int startY = y - (displayNeurons - 1) * spacing / 2;

    for (int i = 0; i < displayNeurons; ++i) {
        int ny = startY + i * spacing;

        double activation = 0.0;
        if (!layer.output.empty() && i < static_cast<int>(layer.output.size())) {
            activation = layer.output[i];
        }

        QColor color = getActivationColor(activation);
        painter.setBrush(color);
        painter.setPen(QPen(Qt::white, 1));
        painter.drawEllipse(QPoint(x, ny), radius, radius);
    }

    if (neurons > displayNeurons) {
        painter.setPen(Qt::gray);
        painter.drawText(x - 5, y + displayNeurons * spacing / 2 + 10, "...");
    }

    if (showLayerInfo_) {
        painter.setPen(Qt::white);
        QFont font = painter.font();
        font.setPointSize(8);
        painter.setFont(font);

        painter.drawText(x - 20, y + displayNeurons * spacing / 2 + 25, 40, 20,
                        Qt::AlignCenter, QString::number(neurons));
        painter.drawText(x - 20, y - displayNeurons * spacing / 2 - 25, 40, 20,
                        Qt::AlignCenter, "Dense");
    }
}

void CNNView::draw3DBox(QPainter& painter, int x, int y, int w, int h, int d,
                         const QColor& color) {
    // 3D偏移
    int dx = d;
    int dy = -d / 2;

    // 后面
    painter.setBrush(color.darker(150));
    painter.setPen(QPen(color.darker(180), 1));
    QPolygon backFace;
    backFace << QPoint(x + dx, y + dy)
             << QPoint(x + w + dx, y + dy)
             << QPoint(x + w + dx, y + h + dy)
             << QPoint(x + dx, y + h + dy);
    painter.drawPolygon(backFace);

    // 顶面
    painter.setBrush(color.lighter(120));
    QPolygon topFace;
    topFace << QPoint(x, y)
            << QPoint(x + w, y)
            << QPoint(x + w + dx, y + dy)
            << QPoint(x + dx, y + dy);
    painter.drawPolygon(topFace);

    // 右侧面
    painter.setBrush(color.darker(110));
    QPolygon rightFace;
    rightFace << QPoint(x + w, y)
              << QPoint(x + w + dx, y + dy)
              << QPoint(x + w + dx, y + h + dy)
              << QPoint(x + w, y + h);
    painter.drawPolygon(rightFace);

    // 前面
    painter.setBrush(color);
    painter.setPen(QPen(color.darker(130), 1.5));
    painter.drawRect(x, y, w, h);

    // 绘制通道分层线
    painter.setPen(QPen(color.darker(140), 0.5));
    int numLines = std::min(d / 3, 5);
    for (int i = 1; i <= numLines; ++i) {
        int lineX = x + i * dx / (numLines + 1);
        int lineY = y + i * dy / (numLines + 1);
        painter.drawLine(lineX, lineY, lineX, lineY + h);
    }
}

void CNNView::drawConnection(QPainter& painter, int x1, int y1, int x2, int y2) {
    painter.setPen(QPen(QColor(100, 100, 120, 150), 1.5));

    QPainterPath path;
    path.moveTo(x1, y1);

    int ctrlX = (x1 + x2) / 2;
    path.cubicTo(ctrlX, y1, ctrlX, y2, x2, y2);

    painter.drawPath(path);
}

QColor CNNView::getLayerColor(CNNLayerType type) {
    switch (type) {
        case CNNLayerType::Convolutional:
            return QColor(70, 130, 180);  // Steel blue
        case CNNLayerType::MaxPooling:
            return QColor(60, 179, 113);  // Medium sea green
        case CNNLayerType::AvgPooling:
            return QColor(46, 139, 87);   // Sea green
        case CNNLayerType::Flatten:
            return QColor(255, 165, 0);   // Orange
        default:
            return QColor(100, 100, 100);
    }
}

QColor CNNView::getActivationColor(double value) {
    value = std::clamp(value, 0.0, 1.0);

    // 使用热力图颜色：蓝->青->绿->黄->红
    if (value < 0.25) {
        int t = static_cast<int>(value / 0.25 * 255);
        return QColor(0, t, 255);
    } else if (value < 0.5) {
        int t = static_cast<int>((value - 0.25) / 0.25 * 255);
        return QColor(0, 255, 255 - t);
    } else if (value < 0.75) {
        int t = static_cast<int>((value - 0.5) / 0.25 * 255);
        return QColor(t, 255, 0);
    } else {
        int t = static_cast<int>((value - 0.75) / 0.25 * 255);
        return QColor(255, 255 - t, 0);
    }
}

void CNNView::mousePressEvent(QMouseEvent* event) {
    for (size_t i = 0; i < layerRects_.size(); ++i) {
        if (layerRects_[i].contains(event->pos())) {
            emit layerClicked(static_cast<int>(i));
            break;
        }
    }
}
