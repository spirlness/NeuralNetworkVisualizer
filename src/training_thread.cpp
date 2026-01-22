#include "training_thread.h"
#include <QThread>

TrainingThread::TrainingThread(QObject* parent)
    : QThread(parent)
    , network_(nullptr)
    , epochs_(100)
    , learningRate_(0.1)
    , running_(false)
    , paused_(false)
    , stopRequested_(false) {
}

TrainingThread::~TrainingThread() {
    stopTraining();
    wait();
}

void TrainingThread::setNetwork(NeuralNetwork* network) {
    QMutexLocker locker(&mutex_);
    network_ = network;
}

void TrainingThread::setTrainingData(const std::vector<std::vector<double>>& inputs,
                                     const std::vector<std::vector<double>>& targets) {
    QMutexLocker locker(&mutex_);
    inputs_ = inputs;
    targets_ = targets;
}

void TrainingThread::setParameters(int epochs, double learningRate) {
    QMutexLocker locker(&mutex_);
    epochs_ = epochs;
    learningRate_ = learningRate;
}

void TrainingThread::stopTraining() {
    stopRequested_ = true;
    paused_ = false;
}

void TrainingThread::pauseTraining() {
    paused_ = true;
}

void TrainingThread::resumeTraining() {
    paused_ = false;
}

void TrainingThread::run() {
    if (!network_) return;

    running_ = true;
    stopRequested_ = false;

    for (int epoch = 0; epoch < epochs_ && !stopRequested_; ++epoch) {
        // 检查暂停
        while (paused_ && !stopRequested_) {
            msleep(100);
        }

        if (stopRequested_) break;

        double loss;
        {
            QMutexLocker locker(&mutex_);
            loss = network_->train(inputs_, targets_, learningRate_);
        }

        emit epochCompleted(epoch + 1, loss);
        emit weightsUpdated();

        // 添加小延迟以便可视化
        msleep(10);
    }

    running_ = false;
    emit trainingCompleted();
}
