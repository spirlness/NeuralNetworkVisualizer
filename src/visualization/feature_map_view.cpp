#include "visualization/feature_map_view.h"
#include <cmath>
#include <algorithm>

FeatureMapView::FeatureMapView(QWidget* parent)
    : QWidget(parent) {
    setMinimumSize(400, 200);
    setStyleSheet("background-color: #252536;");
}

void FeatureMapView::setFeatureMap(const Tensor& featureMap, const QString& layerName) {
    featureMap_ = featureMap;
    layerName_ = layerName;
    selectedChannel_ = -1;

    // 调整widget大小以适应所有特征图
    int rows = (static_cast<int>(featureMap_.channels()) + gridColumns_ - 1) / gridColumns_;
    int neededHeight = rows * (thumbnailSize_ + padding_) + 50;
    setMinimumHeight(std::max(200, neededHeight));

    update();
}

void FeatureMapView::clear() {
    featureMap_ = Tensor();
    layerName_.clear();
    selectedChannel_ = -1;
    update();
}

void FeatureMapView::setColorMap(ColorMap colorMap) {
    colorMap_ = colorMap;
    update();
}

void FeatureMapView::setGridColumns(int columns) {
    gridColumns_ = std::max(1, columns);
    update();
}

void FeatureMapView::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    if (featureMap_.empty()) {
        painter.setPen(Qt::gray);
        painter.drawText(rect(), Qt::AlignCenter, "No Feature Map Selected");
        return;
    }

    // 绘制标题
    painter.setPen(Qt::white);
    QFont font = painter.font();
    font.setPointSize(10);
    font.setBold(true);
    painter.setFont(font);
    painter.drawText(10, 25, QString("%1 - %2 channels")
                    .arg(layerName_)
                    .arg(featureMap_.channels()));

    // 计算归一化范围
    double minVal = featureMap_.min();
    double maxVal = featureMap_.max();
    double range = (maxVal - minVal > 0.001) ? (maxVal - minVal) : 1.0;

    // 绘制每个通道的特征图
    int startY = 40;

    for (size_t c = 0; c < featureMap_.channels(); ++c) {
        QRect cellRect = getChannelRect(static_cast<int>(c));
        cellRect.translate(0, startY);

        // 绘制特征图像素
        int h = static_cast<int>(featureMap_.height());
        int w = static_cast<int>(featureMap_.width());

        QImage image(thumbnailSize_, thumbnailSize_, QImage::Format_RGB32);

        for (int py = 0; py < thumbnailSize_; ++py) {
            for (int px = 0; px < thumbnailSize_; ++px) {
                int fy = py * h / thumbnailSize_;
                int fx = px * w / thumbnailSize_;
                double val = (featureMap_(c, fy, fx) - minVal) / range;
                QColor color = valueToColor(val);
                image.setPixelColor(px, py, color);
            }
        }

        painter.drawImage(cellRect.topLeft(), image);

        // 绘制边框
        if (static_cast<int>(c) == selectedChannel_) {
            painter.setPen(QPen(Qt::yellow, 2));
        } else {
            painter.setPen(QPen(QColor(80, 80, 100), 1));
        }
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(cellRect);

        // 绘制通道编号
        painter.setPen(Qt::white);
        font.setPointSize(7);
        font.setBold(false);
        painter.setFont(font);
        painter.drawText(cellRect.left() + 2, cellRect.bottom() + 12,
                        QString::number(c));
    }
}

QColor FeatureMapView::valueToColor(double value) {
    value = std::clamp(value, 0.0, 1.0);

    switch (colorMap_) {
        case Grayscale: {
            int v = static_cast<int>(value * 255);
            return QColor(v, v, v);
        }
        case Heatmap: {
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
        case Viridis: {
            // 简化的Viridis近似
            int r = static_cast<int>(255 * (0.267 + 0.329*value + 1.260*value*value - 1.856*value*value*value));
            int g = static_cast<int>(255 * (0.004 + 1.016*value - 0.316*value*value));
            int b = static_cast<int>(255 * (0.329 + 0.424*value - 0.753*value*value + 0.401*value*value*value));
            return QColor(std::clamp(r, 0, 255), std::clamp(g, 0, 255), std::clamp(b, 0, 255));
        }
    }

    return Qt::black;
}

QRect FeatureMapView::getChannelRect(int channelIndex) {
    int col = channelIndex % gridColumns_;
    int row = channelIndex / gridColumns_;

    int x = padding_ + col * (thumbnailSize_ + padding_);
    int y = row * (thumbnailSize_ + padding_ + 15);

    return QRect(x, y, thumbnailSize_, thumbnailSize_);
}

void FeatureMapView::mousePressEvent(QMouseEvent* event) {
    int startY = 40;

    for (size_t c = 0; c < featureMap_.channels(); ++c) {
        QRect cellRect = getChannelRect(static_cast<int>(c));
        cellRect.translate(0, startY);

        if (cellRect.contains(event->pos())) {
            selectedChannel_ = static_cast<int>(c);
            emit channelSelected(selectedChannel_);
            update();
            return;
        }
    }

    selectedChannel_ = -1;
    update();
}
