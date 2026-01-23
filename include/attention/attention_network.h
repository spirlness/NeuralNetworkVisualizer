#ifndef ATTENTION_NETWORK_H
#define ATTENTION_NETWORK_H

#include "attention/transformer_block.h"
#include <vector>

class AttentionNetwork {
public:
    AttentionNetwork(size_t seqLen, size_t d_model, size_t d_k, size_t d_ff, size_t num_layers);

    Tensor forward(const Tensor& input); // Input (1, L, 1)
    double backward(const Tensor& target, double learningRate); // Returns loss

    const std::vector<TransformerBlock>& getBlocks() const { return blocks_; }
    const Tensor& getInput() const { return input_; }
    const Tensor& getOutput() const { return output_; }

private:
    size_t seqLen_;
    size_t d_model_;

    // Embedding: Linear (1 -> d_model)
    Tensor W_embed_; // (1, 1, d_model)
    Tensor b_embed_; // (1, 1, d_model)

    Tensor posEncoding_; // (1, L, d_model)

    std::vector<TransformerBlock> blocks_;

    // Output Head: Linear (d_model -> 1)
    Tensor W_out_; // (1, d_model, 1)
    Tensor b_out_; // (1, 1, 1)

    // Cache
    Tensor input_;
    Tensor embedded_;
    Tensor blocksInput_; // embedded + pos
    Tensor finalBlockOutput_;
    Tensor output_;

    void initPosEncoding();
};

#endif // ATTENTION_NETWORK_H
