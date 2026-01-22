#ifndef TRAINING_THREAD_H
#define TRAINING_THREAD_H

#include <QThread>
#include <QMutex>
#include <atomic>
#include <vector>
#include "neural_network.h"

class TrainingThread : public QThread {
    Q_OBJECT

public:
    explicit TrainingThread(QObject* parent = nullptr);
    ~TrainingThread() override;

    // 设置训练参数
    void setNetwork(NeuralNetwork* network);
    void setTrainingData(const std::vector<std::vector<double>>& inputs,
                         const std::vector<std::vector<double>>& targets);
    void setParameters(int epochs, double learningRate);

    // 控制
    void stopTraining();
    void pauseTraining();
    void resumeTraining();

    bool isPaused() const { return paused_; }
    bool isRunning() const { return running_; }

signals:
    void epochCompleted(int epoch, double loss);
    void trainingCompleted();
    void weightsUpdated();

protected:
    void run() override;

private:
    NeuralNetwork* network_;
    std::vector<std::vector<double>> inputs_;
    std::vector<std::vector<double>> targets_;

    int epochs_;
    double learningRate_;

    std::atomic<bool> running_;
    std::atomic<bool> paused_;
    std::atomic<bool> stopRequested_;

    QMutex mutex_;
};

#endif // TRAINING_THREAD_H
