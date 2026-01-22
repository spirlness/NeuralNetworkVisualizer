#ifndef NETWORK_VIEW_H
#define NETWORK_VIEW_H

#include <QWidget>
#include <QPainter>
#include <vector>
#include "neural_network.h"

class NetworkView : public QWidget {
    Q_OBJECT

public:
    explicit NetworkView(QWidget* parent = nullptr);

    void setNetwork(NeuralNetwork* network);
    void updateView();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    void drawNeuron(QPainter& painter, int x, int y, int radius, double activation = 0.0);
    void drawConnection(QPainter& painter, int x1, int y1, int x2, int y2, double weight);
    QColor getWeightColor(double weight);
    QColor getActivationColor(double activation);

    NeuralNetwork* network_;
    std::vector<int> layerSizes_;
    int neuronRadius_ = 20;
    int layerSpacing_ = 150;
    int neuronSpacing_ = 50;
};

#endif // NETWORK_VIEW_H
