#include "cnn/cnn_training_thread.h"
#include <QThread>

CNNTrainingThread::CNNTrainingThread(QObject* parent)
    : QThread(parent)
    , network_(nullptr)
    , epochs_(100)
    , learningRate_(0.01)
    , running_(false)
    , paused_(false)
    , stopRequested_(false) {
}

CNNTrainingThread::~CNNTrainingThread() {
    stopTraining();
    wait();
}

void CNNTrainingThread::setNetwork(CNNNetwork* network) {
    QMutexLocker locker(&mutex_);
    network_ = network;
}

void CNNTrainingThread::setTrainingData(const std::vector<Tensor>& inputs,
                                        const std::vector<std::vector<double>>& targets) {
    QMutexLocker locker(&mutex_);
    inputs_ = inputs;
    targets_ = targets;
}

void CNNTrainingThread::setParameters(int epochs, double learningRate) {
    QMutexLocker locker(&mutex_);
    epochs_ = epochs;
    learningRate_ = learningRate;
}

void CNNTrainingThread::stopTraining() {
    stopRequested_ = true;
    paused_ = false;
}

void CNNTrainingThread::pauseTraining() {
    paused_ = true;
}

void CNNTrainingThread::resumeTraining() {
    paused_ = false;
}

void CNNTrainingThread::run() {
    if (!network_) return;

    running_ = true;
    stopRequested_ = false;

    for (int epoch = 0; epoch < epochs_ && !stopRequested_; ++epoch) {
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

        msleep(10);
    }

    running_ = false;
    emit trainingCompleted();
}
