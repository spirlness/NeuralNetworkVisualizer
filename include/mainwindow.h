#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWidget>
#include <QPushButton>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QProgressBar>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QTextEdit>
#include <memory>

#include "neural_network.h"
#include "network_view.h"
#include "loss_chart.h"
#include "training_thread.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void onStartTraining();
    void onStopTraining();
    void onPauseResumeTraining();
    void onResetNetwork();
    void onEpochCompleted(int epoch, double loss);
    void onTrainingCompleted();
    void onWeightsUpdated();
    void onDatasetChanged(int index);

private:
    void setupUI();
    void createNetwork();
    void loadDataset(int datasetIndex);
    void updateStatus(const QString& message);
    void log(const QString& message);

    // 神经网络
    std::unique_ptr<NeuralNetwork> network_;
    std::unique_ptr<TrainingThread> trainingThread_;

    // 训练数据
    std::vector<std::vector<double>> trainInputs_;
    std::vector<std::vector<double>> trainTargets_;

    // UI组件
    NetworkView* networkView_;
    LossChart* lossChart_;

    // 控制面板
    QSpinBox* epochsSpinBox_;
    QDoubleSpinBox* learningRateSpinBox_;
    QComboBox* datasetComboBox_;
    QSpinBox* hiddenLayerSpinBox_;
    QSpinBox* hiddenNeuronsSpinBox_;
    QComboBox* activationComboBox_;

    QPushButton* startButton_;
    QPushButton* stopButton_;
    QPushButton* pauseButton_;
    QPushButton* resetButton_;

    QProgressBar* progressBar_;
    QLabel* statusLabel_;
    QTextEdit* logTextEdit_;

    int currentEpoch_ = 0;
    int totalEpochs_ = 100;
};

#endif // MAINWINDOW_H
