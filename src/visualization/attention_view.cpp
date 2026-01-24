#include "visualization/attention_view.h"
#include <QPainter>
#include <QPaintEvent>
#include <algorithm>
#include <cmath>

AttentionView::AttentionView(QWidget* parent) : QWidget(parent), network_(nullptr) {
    setMinimumSize(800, 600);
}

void AttentionView::setNetwork(AttentionNetwork* network) {
    network_ = network;
    update();
}

void AttentionView::updateView() {
    update();
}

void AttentionView::paintEvent(QPaintEvent* event) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect(), QColor("#2d2d3d"));

    if (!network_) {
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter, "No Network Available");
        return;
    }

    std::lock_guard<std::mutex> lock(network_->getMutex());

    if (network_->getBlocks().empty()) {
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter, "No Blocks Available");
        return;
    }

    const auto& block = network_->getBlocks()[0];
    const auto& attn = block.getAttention();

    Tensor Q = attn.getQ();
    Tensor K = attn.getK();
    Tensor V = attn.getV();
    Tensor W = attn.getWeights();

    // Also Input and Output
    Tensor input = network_->getInput();
    Tensor output = network_->getOutput();

    if (Q.empty()) return;

    int margin = 30;
    int topH = 60; // For input/output
    int w = width() - 2 * margin;
    int h = height() - 2 * margin - 2 * topH;

    // Draw Input (Top)
    QRect rectInput(margin, 10, w / 2 - 10, 40);
    drawSequence(painter, input, rectInput, "Input Sequence (Sorted Target)");

    // Draw Output (Top Right)
    QRect rectOutput(margin + w / 2 + 10, 10, w / 2 - 10, 40);
    drawSequence(painter, output, rectOutput, "Output Sequence");

    // Grid Layout for Matrix
    int boxW = (w - margin) / 2;
    int boxH = (h - margin) / 2;
    int startY = margin + topH;

    QRect rectQ(margin, startY, boxW, boxH);
    QRect rectK(margin + boxW + margin, startY, boxW, boxH);
    QRect rectV(margin, startY + boxH + margin, boxW, boxH);
    QRect rectW(margin + boxW + margin, startY + boxH + margin, boxW, boxH);

    drawMatrix(painter, Q, rectQ, "Query (Q) - [Seq x D_k]");
    drawMatrix(painter, K, rectK, "Key (K) - [Seq x D_k]");
    drawMatrix(painter, V, rectV, "Value (V) - [Seq x D_k]");
    drawMatrix(painter, W, rectW, "Attention Weights - [Seq x Seq]");
}

void AttentionView::drawMatrix(QPainter& painter, const Tensor& tensor, const QRect& rect, const QString& title) {
    painter.setPen(Qt::white);
    painter.drawText(rect.left(), rect.top() - 5, title);

    int rows = tensor.height();
    int cols = tensor.width();
    if (rows == 0 || cols == 0) return;

    double cellW = (double)rect.width() / cols;
    double cellH = (double)rect.height() / rows;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double val = tensor(0, r, c);

            // Normalize for color
            QColor color;
            if (title.contains("Weights")) {
                // 0..1
                int v = std::clamp(static_cast<int>(val * 255), 0, 255);
                // Heatmap: 0->Black, 1->Red/Yellow
                color = QColor(v, v/3, 0);
            } else {
                // -1..1
                val = std::clamp(val, -1.0, 1.0);
                int v = static_cast<int>(std::abs(val) * 255);
                // Blue (-) -> Black (0) -> Red (+)
                if (val < 0) color = QColor(0, 0, v);
                else color = QColor(v, 0, 0);
            }

            painter.fillRect(QRectF(rect.left() + c * cellW, rect.top() + r * cellH, cellW, cellH), color);
            painter.setPen(QColor(100, 100, 100, 50));
            painter.drawRect(QRectF(rect.left() + c * cellW, rect.top() + r * cellH, cellW, cellH));

            // Draw value text if cell is big enough
            if (cellW > 30 && cellH > 15) {
                painter.setPen(Qt::white);
                painter.drawText(QRectF(rect.left() + c * cellW, rect.top() + r * cellH, cellW, cellH),
                                 Qt::AlignCenter, QString::number(val, 'f', 2));
            }
        }
    }
}

void AttentionView::drawSequence(QPainter& painter, const Tensor& seq, const QRect& rect, const QString& title) {
    painter.setPen(Qt::white);
    painter.drawText(rect.left(), rect.top() - 5, title);

    // Seq is (1, L, 1) usually
    int len = seq.height();
    if (len == 0) return;

    double cellW = (double)rect.width() / len;
    double cellH = rect.height();

    for (int i = 0; i < len; ++i) {
        double val = seq(0, i, 0);
        // Normalize 0..1
        int v = std::clamp(static_cast<int>(val * 255), 0, 255);
        QColor color(v, v, v);

        painter.fillRect(QRectF(rect.left() + i * cellW, rect.top(), cellW, cellH), color);
        painter.setPen(Qt::yellow);
        painter.drawRect(QRectF(rect.left() + i * cellW, rect.top(), cellW, cellH));

        // Draw Value
        painter.setPen(val > 0.5 ? Qt::black : Qt::white);
        painter.drawText(QRectF(rect.left() + i * cellW, rect.top(), cellW, cellH),
                         Qt::AlignCenter, QString::number(val, 'f', 2));
    }
}
