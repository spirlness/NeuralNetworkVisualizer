#ifndef CNN_TRAINING_THREAD_H
#define CNN_TRAINING_THREAD_H

#include <QThread>
#include <QMutex>
#include <atomic>
#include <vector>
#include "cnn/cnn_network.h"
#include "cnn/tensor.h"

/**
 * @brief CNN训练线程类
 * 
 * 在独立线程中训练CNN网络，通过信号通知主线程更新UI
 */
class CNNTrainingThread : public QThread {
    Q_OBJECT

public:
    explicit CNNTrainingThread(QObject* parent = nullptr);
    ~CNNTrainingThread() override;

    // 设置训练参数
    void setNetwork(CNNNetwork* network);
    void setTrainingData(const std::vector<Tensor>& inputs,
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
    CNNNetwork* network_;
    std::vector<Tensor> inputs_;
    std::vector<std::vector<double>> targets_;

    int epochs_;
    double learningRate_;

    std::atomic<bool> running_;
    std::atomic<bool> paused_;
    std::atomic<bool> stopRequested_;

    QMutex mutex_;
};

#endif // CNN_TRAINING_THREAD_H
