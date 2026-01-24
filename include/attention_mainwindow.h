#ifndef ATTENTION_MAINWINDOW_H
#define ATTENTION_MAINWINDOW_H

#include <QMainWindow>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QTextEdit>
#include <QLabel>
#include <QProgressBar>
#include <memory>

#include "attention/attention_network.h"
#include "attention/attention_training_thread.h"
#include "visualization/attention_view.h"
#include "loss_chart.h"

class AttentionMainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit AttentionMainWindow(QWidget* parent = nullptr);
    ~AttentionMainWindow();

private slots:
    void onStartTraining();
    void onStopTraining();
    void onPauseResumeTraining();
    void onResetNetwork();

    void onEpochCompleted(int epoch, double loss);
    void onTrainingCompleted();
    void onWeightsUpdated();

private:
    void setupUI();
    void createNetwork();
    void updateStatus(const QString& message);
    void log(const QString& message);

    // Network & Training
    std::unique_ptr<AttentionNetwork> network_;
    std::unique_ptr<AttentionTrainingThread> trainingThread_;

    // UI Controls
    QSpinBox* epochsSpinBox_;
    QDoubleSpinBox* learningRateSpinBox_;
    QSpinBox* seqLenSpinBox_;
    QSpinBox* dModelSpinBox_;
    QSpinBox* layersSpinBox_;

    QPushButton* startButton_;
    QPushButton* stopButton_;
    QPushButton* pauseButton_;
    QPushButton* resetButton_;

    QProgressBar* progressBar_;
    QLabel* statusLabel_;
    QTextEdit* logTextEdit_;

    // Visualization
    AttentionView* attentionView_;
    LossChart* lossChart_;

    int totalEpochs_;
};

#endif // ATTENTION_MAINWINDOW_H
