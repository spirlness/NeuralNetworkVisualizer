#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "attention/attention_layer.h"

class TransformerBlock {
public:
    TransformerBlock(size_t d_model, size_t d_k, size_t d_ff);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& gradOutput, double learningRate);

    const AttentionLayer& getAttention() const { return attention_; }

private:
    size_t d_model_;

    AttentionLayer attention_;

    // Feed Forward Weights
    Tensor W1_, b1_; // D -> FF
    Tensor W2_, b2_; // FF -> D

    // Layer Norm Weights
    Tensor gamma1_, beta1_;
    Tensor gamma2_, beta2_;

    // Cache for backward
    Tensor input_;
    Tensor attnOutput_; // Post attention, pre-add
    Tensor norm1Input_; // input + attnOutput
    Tensor norm1Output_; // After first norm

    Tensor ffHidden_;   // After W1, before ReLU
    Tensor ffRelu_;     // After ReLU
    Tensor ffOutput_;   // After W2
    Tensor norm2Input_; // norm1Output + ffOutput

    // Helper to apply LayerNorm forward
    Tensor forwardLayerNorm(const Tensor& x, const Tensor& gamma, const Tensor& beta);
    // Helper for LayerNorm backward
    Tensor backwardLayerNorm(const Tensor& dY, const Tensor& x, const Tensor& gamma, Tensor& dGamma, Tensor& dBeta);
};

#endif // TRANSFORMER_BLOCK_H
