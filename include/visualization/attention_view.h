#ifndef ATTENTION_VIEW_H
#define ATTENTION_VIEW_H

#include <QWidget>
#include "attention/attention_network.h"

class AttentionView : public QWidget {
    Q_OBJECT

public:
    explicit AttentionView(QWidget* parent = nullptr);
    void setNetwork(AttentionNetwork* network);
    void updateView();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    AttentionNetwork* network_;

    // Helper to draw a matrix/tensor
    void drawMatrix(QPainter& painter, const Tensor& tensor, const QRect& rect, const QString& title);
    // Helper to draw sequence
    void drawSequence(QPainter& painter, const Tensor& seq, const QRect& rect, const QString& title);
};

#endif // ATTENTION_VIEW_H
