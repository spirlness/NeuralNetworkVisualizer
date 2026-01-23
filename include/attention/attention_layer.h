#ifndef ATTENTION_LAYER_H
#define ATTENTION_LAYER_H

#include "cnn/tensor.h"
#include <cmath>

class AttentionLayer {
public:
    AttentionLayer(size_t d_model, size_t d_k);

    Tensor forward(const Tensor& input); // Input: (1, SeqLen, D_model)
    Tensor backward(const Tensor& gradOutput, double learningRate);

    // Visualization getters
    const Tensor& getQ() const { return Q_; }
    const Tensor& getK() const { return K_; }
    const Tensor& getV() const { return V_; }
    const Tensor& getWeights() const { return attentionWeights_; }

private:
    size_t d_model_;
    size_t d_k_;

    // Weights
    Tensor W_Q_, W_K_, W_V_, W_O_;

    // Cache for backward/viz
    Tensor input_;
    Tensor Q_, K_, V_;
    Tensor scores_;     // Before softmax
    Tensor attentionWeights_; // After softmax
};

#endif // ATTENTION_LAYER_H
