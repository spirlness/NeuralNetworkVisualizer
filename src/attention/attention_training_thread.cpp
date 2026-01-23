#include "attention/attention_training_thread.h"
#include <algorithm>
#include <vector>
#include <random>

AttentionTrainingThread::AttentionTrainingThread()
    : network_(nullptr), epochs_(1000), learningRate_(0.01), seqLen_(5),
      running_(false), paused_(false) {}

AttentionTrainingThread::~AttentionTrainingThread() {
    stopTraining();
    wait();
}

void AttentionTrainingThread::setNetwork(AttentionNetwork* network) {
    network_ = network;
}

void AttentionTrainingThread::setParameters(int epochs, double learningRate, int seqLen) {
    epochs_ = epochs;
    learningRate_ = learningRate;
    seqLen_ = seqLen;
}

void AttentionTrainingThread::startTraining() {
    running_ = true;
    paused_ = false;
    start();
}

void AttentionTrainingThread::stopTraining() {
    running_ = false;
    resumeTraining();
}

void AttentionTrainingThread::pauseTraining() {
    paused_ = true;
}

void AttentionTrainingThread::resumeTraining() {
    QMutexLocker locker(&mutex_);
    paused_ = false;
    pauseCondition_.wakeAll();
}

void AttentionTrainingThread::run() {
    if (!network_) return;

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int epoch = 1; epoch <= epochs_; ++epoch) {
        if (!running_) break;

        // Handle pause
        {
            QMutexLocker locker(&mutex_);
            if (paused_) {
                pauseCondition_.wait(&mutex_);
            }
        }

        double epochLoss = 0.0;
        int batchSize = 50;

        for (int b = 0; b < batchSize; ++b) {
            // Generate Data
            std::vector<double> data(seqLen_);
            for (auto& v : data) v = dis(gen);

            // Create Input Tensor
            Tensor input = Tensor::fromVector(data, 1, seqLen_, 1);

            // Create Target (Sorted)
            std::vector<double> sortedData = data;
            std::sort(sortedData.begin(), sortedData.end());
            Tensor target = Tensor::fromVector(sortedData, 1, seqLen_, 1);

            // Forward & Backward
            network_->forward(input);
            double loss = network_->backward(target, learningRate_);
            epochLoss += loss;
        }

        epochLoss /= batchSize;

        emit epochCompleted(epoch, epochLoss);

        if (epoch % 10 == 0) {
            emit weightsUpdated();
        }
    }

    emit trainingCompleted();
}
