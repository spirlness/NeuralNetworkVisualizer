#ifndef ATTENTION_TRAINING_THREAD_H
#define ATTENTION_TRAINING_THREAD_H

#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include "attention/attention_network.h"

class AttentionTrainingThread : public QThread {
    Q_OBJECT

public:
    AttentionTrainingThread();
    ~AttentionTrainingThread();

    void setNetwork(AttentionNetwork* network);
    void setParameters(int epochs, double learningRate, int seqLen);

    void startTraining();
    void stopTraining();
    void pauseTraining();
    void resumeTraining();

    bool isPaused() const { return paused_; }

signals:
    void epochCompleted(int epoch, double loss);
    void trainingCompleted();
    void weightsUpdated();

protected:
    void run() override;

private:
    AttentionNetwork* network_;
    int epochs_;
    double learningRate_;
    int seqLen_;

    bool running_;
    bool paused_;
    QMutex mutex_;
    QWaitCondition pauseCondition_;
};

#endif // ATTENTION_TRAINING_THREAD_H
