#ifndef FEATURE_MAP_VIEW_H
#define FEATURE_MAP_VIEW_H

#include <QWidget>
#include <QPainter>
#include <QScrollArea>
#include <QMouseEvent>
#include "../cnn/tensor.h"

/**
 * @brief 特征图详细可视化组件
 *
 * 以网格形式显示某一层所有通道的特征图
 */
class FeatureMapView : public QWidget {
    Q_OBJECT

public:
    explicit FeatureMapView(QWidget* parent = nullptr);

    void setFeatureMap(const Tensor& featureMap, const QString& layerName);
    void clear();

    enum ColorMap { Grayscale, Heatmap, Viridis };
    void setColorMap(ColorMap colorMap);
    void setGridColumns(int columns);

signals:
    void channelSelected(int channelIndex);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    QColor valueToColor(double value);
    QRect getChannelRect(int channelIndex);

    Tensor featureMap_;
    QString layerName_;

    ColorMap colorMap_ = Heatmap;
    int gridColumns_ = 8;
    int thumbnailSize_ = 50;
    int padding_ = 5;

    int selectedChannel_ = -1;
};

#endif // FEATURE_MAP_VIEW_H
