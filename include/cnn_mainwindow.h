#ifndef CNN_MAINWINDOW_H
#define CNN_MAINWINDOW_H

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
#include <QTabWidget>
#include <QScrollArea>
#include <memory>

#include "cnn/cnn_network.h"
#include "cnn/cnn_training_thread.h"
#include "visualization/cnn_view.h"
#include "visualization/feature_map_view.h"
#include "loss_chart.h"

/**
 * @brief CNN可视化主窗口
 */
class CNNMainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit CNNMainWindow(QWidget* parent = nullptr);
    ~CNNMainWindow() override;

private slots:
    void onStartTraining();
    void onStopTraining();
    void onPauseResumeTraining();
    void onResetNetwork();
    void onBuildNetwork();
    void onLayerClicked(int layerIndex);
    void onEpochCompleted(int epoch, double loss);
    void onTrainingCompleted();
    void onWeightsUpdated();

private:
    void setupUI();
    void createNetwork();
    void generateTrainingData();
    void updateStatus(const QString& message);
    void log(const QString& message);

    // CNN网络
    std::unique_ptr<CNNNetwork> cnnNetwork_;

    // 训练数据
    std::vector<Tensor> trainImages_;
    std::vector<std::vector<double>> trainLabels_;

    // UI组件
    CNNView* cnnView_;
    FeatureMapView* featureMapView_;
    LossChart* lossChart_;

    // CNN配置
    QSpinBox* inputSizeSpinBox_;
    QSpinBox* numClassesSpinBox_;
    QSpinBox* conv1FiltersSpinBox_;
    QSpinBox* conv2FiltersSpinBox_;
    QSpinBox* kernelSizeSpinBox_;
    QSpinBox* hiddenNeuronsSpinBox_;

    // 训练参数
    QSpinBox* epochsSpinBox_;
    QDoubleSpinBox* learningRateSpinBox_;
    QSpinBox* samplesSpinBox_;

    // 控制按钮
    QPushButton* buildButton_;
    QPushButton* startButton_;
    QPushButton* stopButton_;
    QPushButton* pauseButton_;
    QPushButton* resetButton_;

    QProgressBar* progressBar_;
    QLabel* statusLabel_;
    QTextEdit* logTextEdit_;

    std::unique_ptr<CNNTrainingThread> trainingThread_;
    int totalEpochs_ = 100;
};

#endif // CNN_MAINWINDOW_H
