#include "loss_chart.h"
#include <QPainter>
#include <QPainterPath>
#include <QPaintEvent>
#include <cmath>
#include <algorithm>

LossChart::LossChart(QWidget* parent)
    : QWidget(parent) {
    setMinimumSize(400, 200);
    setStyleSheet("background-color: #1e1e2e;");
}

void LossChart::addDataPoint(int epoch, double loss) {
    dataPoints_.push_back({epoch, loss});

    if (static_cast<int>(dataPoints_.size()) > maxPoints_) {
        dataPoints_.pop_front();
    }

    // 更新最大最小损失值
    if (!dataPoints_.empty()) {
        maxLoss_ = 0.0;
        minLoss_ = std::numeric_limits<double>::max();
        for (const auto& point : dataPoints_) {
            maxLoss_ = std::max(maxLoss_, point.second);
            minLoss_ = std::min(minLoss_, point.second);
        }
        // 添加一些边距
        double range = maxLoss_ - minLoss_;
        if (range < 0.001) range = 0.1;
        maxLoss_ += range * 0.1;
        minLoss_ = std::max(0.0, minLoss_ - range * 0.1);
    }

    update();
}

void LossChart::clear() {
    dataPoints_.clear();
    maxLoss_ = 1.0;
    minLoss_ = 0.0;
    update();
}

void LossChart::setMaxPoints(int maxPoints) {
    maxPoints_ = maxPoints;
}

void LossChart::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // 绘制标题
    painter.setPen(Qt::white);
    painter.setFont(QFont("Arial", 12, QFont::Bold));
    painter.drawText(marginLeft_, 20, "Training Loss");

    if (dataPoints_.empty()) {
        painter.setFont(QFont("Arial", 10));
        painter.drawText(rect(), Qt::AlignCenter, "No training data yet");
        return;
    }

    drawGrid(painter);
    drawAxes(painter);
    drawCurve(painter);
    drawLabels(painter);
}

void LossChart::drawGrid(QPainter& painter) {
    int chartWidth = width() - marginLeft_ - marginRight_;
    int chartHeight = height() - marginTop_ - marginBottom_;

    painter.setPen(QPen(QColor(80, 80, 100), 1, Qt::DotLine));

    // 水平网格线
    int numHLines = 5;
    for (int i = 0; i <= numHLines; ++i) {
        int y = marginTop_ + chartHeight * i / numHLines;
        painter.drawLine(marginLeft_, y, marginLeft_ + chartWidth, y);
    }

    // 垂直网格线
    int numVLines = 10;
    for (int i = 0; i <= numVLines; ++i) {
        int x = marginLeft_ + chartWidth * i / numVLines;
        painter.drawLine(x, marginTop_, x, marginTop_ + chartHeight);
    }
}

void LossChart::drawAxes(QPainter& painter) {
    int chartWidth = width() - marginLeft_ - marginRight_;
    int chartHeight = height() - marginTop_ - marginBottom_;

    painter.setPen(QPen(Qt::white, 2));
    // Y轴
    painter.drawLine(marginLeft_, marginTop_, marginLeft_, marginTop_ + chartHeight);
    // X轴
    painter.drawLine(marginLeft_, marginTop_ + chartHeight,
                    marginLeft_ + chartWidth, marginTop_ + chartHeight);
}

void LossChart::drawCurve(QPainter& painter) {
    if (dataPoints_.size() < 2) return;

    int chartWidth = width() - marginLeft_ - marginRight_;
    int chartHeight = height() - marginTop_ - marginBottom_;

    // 计算epoch范围
    int minEpoch = dataPoints_.front().first;
    int maxEpoch = dataPoints_.back().first;
    int epochRange = std::max(1, maxEpoch - minEpoch);

    // 损失值范围
    double lossRange = maxLoss_ - minLoss_;
    if (lossRange < 0.0001) lossRange = 1.0;

    // 绘制曲线
    QPainterPath path;
    bool firstPoint = true;

    for (const auto& point : dataPoints_) {
        double xRatio = static_cast<double>(point.first - minEpoch) / epochRange;
        double yRatio = (point.second - minLoss_) / lossRange;

        int x = marginLeft_ + static_cast<int>(xRatio * chartWidth);
        int y = marginTop_ + chartHeight - static_cast<int>(yRatio * chartHeight);

        if (firstPoint) {
            path.moveTo(x, y);
            firstPoint = false;
        } else {
            path.lineTo(x, y);
        }
    }

    // 绘制曲线阴影
    QPainterPath fillPath = path;
    fillPath.lineTo(marginLeft_ + chartWidth, marginTop_ + chartHeight);
    fillPath.lineTo(marginLeft_, marginTop_ + chartHeight);
    fillPath.closeSubpath();

    QLinearGradient gradient(0, marginTop_, 0, marginTop_ + chartHeight);
    gradient.setColorAt(0, QColor(100, 150, 255, 100));
    gradient.setColorAt(1, QColor(100, 150, 255, 20));
    painter.fillPath(fillPath, gradient);

    // 绘制主曲线
    painter.setPen(QPen(QColor(100, 150, 255), 2));
    painter.drawPath(path);

    // 绘制当前点
    if (!dataPoints_.empty()) {
        const auto& lastPoint = dataPoints_.back();
        double xRatio = static_cast<double>(lastPoint.first - minEpoch) / epochRange;
        double yRatio = (lastPoint.second - minLoss_) / lossRange;

        int x = marginLeft_ + static_cast<int>(xRatio * chartWidth);
        int y = marginTop_ + chartHeight - static_cast<int>(yRatio * chartHeight);

        painter.setBrush(QColor(255, 200, 100));
        painter.setPen(QPen(Qt::white, 2));
        painter.drawEllipse(QPoint(x, y), 5, 5);
    }
}

void LossChart::drawLabels(QPainter& painter) {
    int chartWidth = width() - marginLeft_ - marginRight_;
    int chartHeight = height() - marginTop_ - marginBottom_;

    painter.setPen(Qt::white);
    painter.setFont(QFont("Arial", 9));

    // Y轴标签
    int numYLabels = 5;
    for (int i = 0; i <= numYLabels; ++i) {
        int y = marginTop_ + chartHeight * i / numYLabels;
        double value = maxLoss_ - (maxLoss_ - minLoss_) * i / numYLabels;
        QString label = QString::number(value, 'f', 4);
        painter.drawText(5, y + 4, label);
    }

    // X轴标签
    if (!dataPoints_.empty()) {
        int minEpoch = dataPoints_.front().first;
        int maxEpoch = dataPoints_.back().first;

        painter.drawText(marginLeft_, height() - 5, QString::number(minEpoch));
        painter.drawText(marginLeft_ + chartWidth - 30, height() - 5, QString::number(maxEpoch));

        // 轴标题
        painter.drawText(marginLeft_ + chartWidth / 2 - 20, height() - 5, "Epoch");
    }

    // 当前损失值
    if (!dataPoints_.empty()) {
        const auto& lastPoint = dataPoints_.back();
        QString info = QString("Epoch: %1  Loss: %2")
                           .arg(lastPoint.first)
                           .arg(lastPoint.second, 0, 'f', 6);
        painter.setFont(QFont("Arial", 10, QFont::Bold));
        painter.drawText(marginLeft_ + 100, 20, info);
    }
}
