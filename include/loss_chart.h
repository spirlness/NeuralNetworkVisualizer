#ifndef LOSS_CHART_H
#define LOSS_CHART_H

#include <QWidget>
#include <QPainter>
#include <vector>
#include <deque>

class LossChart : public QWidget {
    Q_OBJECT

public:
    explicit LossChart(QWidget* parent = nullptr);

    void addDataPoint(int epoch, double loss);
    void clear();
    void setMaxPoints(int maxPoints);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    void drawGrid(QPainter& painter);
    void drawAxes(QPainter& painter);
    void drawCurve(QPainter& painter);
    void drawLabels(QPainter& painter);

    std::deque<std::pair<int, double>> dataPoints_;
    int maxPoints_ = 500;
    double maxLoss_ = 1.0;
    double minLoss_ = 0.0;

    int marginLeft_ = 60;
    int marginRight_ = 20;
    int marginTop_ = 30;
    int marginBottom_ = 40;
};

#endif // LOSS_CHART_H
