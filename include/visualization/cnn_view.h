#ifndef CNN_VIEW_H
#define CNN_VIEW_H

#include <QWidget>
#include <QPainter>
#include <QPainterPath>
#include <QMouseEvent>
#include <vector>
#include "../cnn/cnn_network.h"

/**
 * @brief CNN网络架构可视化组件
 *
 * 显示卷积层、池化层的3D效果图和特征图缩略图
 */
class CNNView : public QWidget {
    Q_OBJECT

public:
    explicit CNNView(QWidget* parent = nullptr);

    void setNetwork(CNNNetwork* network);
    void updateView();

    void setShowFeatureMaps(bool show) { showFeatureMaps_ = show; update(); }
    void setShowLayerInfo(bool show) { showLayerInfo_ = show; update(); }

signals:
    void layerClicked(int layerIndex);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    void drawInputLayer(QPainter& painter, int x, int y);
    void drawCNNLayer(QPainter& painter, int layerIndex, int x, int y);
    void drawDenseLayer(QPainter& painter, int layerIndex, int x, int y);
    void draw3DBox(QPainter& painter, int x, int y, int w, int h, int d, const QColor& color);
    void drawConnection(QPainter& painter, int x1, int y1, int x2, int y2);

    QColor getLayerColor(CNNLayerType type);
    QColor getActivationColor(double value);

    CNNNetwork* network_ = nullptr;

    int layerSpacing_ = 100;
    int maxHeight_ = 200;

    bool showFeatureMaps_ = true;
    bool showLayerInfo_ = true;

    std::vector<QRect> layerRects_;
};

#endif // CNN_VIEW_H
