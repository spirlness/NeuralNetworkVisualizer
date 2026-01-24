#include "attention/attention_network.h"
#include "cnn/random.h"
#include <cmath>

AttentionNetwork::AttentionNetwork(size_t seqLen, size_t d_model, size_t d_k, size_t d_ff, size_t num_layers)
    : seqLen_(seqLen), d_model_(d_model) {

    // Embed
    W_embed_ = Tensor(1, 1, d_model);
    W_embed_.xavierInit(1, d_model);
    b_embed_ = Tensor(1, 1, d_model);

    // Pos Encoding
    posEncoding_ = Tensor(1, seqLen, d_model);
    initPosEncoding();

    // Blocks
    for (size_t i = 0; i < num_layers; ++i) {
        blocks_.emplace_back(d_model, d_k, d_ff);
    }

    // Output
    W_out_ = Tensor(1, d_model, 1);
    W_out_.xavierInit(d_model, 1);
    b_out_ = Tensor(1, 1, 1);
}

void AttentionNetwork::initPosEncoding() {
    for (size_t pos = 0; pos < seqLen_; ++pos) {
        for (size_t i = 0; i < d_model_; ++i) {
            if (i % 2 == 0) {
                posEncoding_(0, pos, i) = std::sin(pos / std::pow(10000.0, (double)i / d_model_));
            } else {
                posEncoding_(0, pos, i) = std::cos(pos / std::pow(10000.0, (double)(i - 1) / d_model_));
            }
        }
    }
}

Tensor AttentionNetwork::forward(const Tensor& input) {
    std::lock_guard<std::mutex> lock(mutex_);
    input_ = input; // (1, L, 1)

    // Dynamic Sequence Length Support
    if (input.height() != seqLen_) {
        seqLen_ = input.height();
        if (posEncoding_.height() != seqLen_) {
            posEncoding_.resize(1, seqLen_, d_model_);
            initPosEncoding();
        }
    }

    // Embedding
    // (1, L, 1) * (1, 1, D) -> (1, L, D)
    embedded_ = input.matmul(W_embed_);

    // Add bias to embedding
    for(size_t c=0; c<embedded_.channels(); ++c)
        for(size_t h=0; h<embedded_.height(); ++h)
            for(size_t w=0; w<embedded_.width(); ++w)
                embedded_(c, h, w) += b_embed_(0, 0, w);

    // Add Pos Encoding
    blocksInput_ = embedded_ + posEncoding_;

    // Blocks
    Tensor x = blocksInput_;
    for (auto& block : blocks_) {
        x = block.forward(x);
    }
    finalBlockOutput_ = x;

    // Output Head
    // (1, L, D) * (1, D, 1) -> (1, L, 1)
    output_ = finalBlockOutput_.matmul(W_out_);

    // Add bias
    for(size_t c=0; c<output_.channels(); ++c)
        for(size_t h=0; h<output_.height(); ++h)
            for(size_t w=0; w<output_.width(); ++w)
                output_(c, h, w) += b_out_(0, 0, w);

    return output_;
}

double AttentionNetwork::backward(const Tensor& target, double learningRate) {
    std::lock_guard<std::mutex> lock(mutex_);
    // MSE Loss
    // L = 1/N * sum((y - t)^2)
    // dL/dy = 2/N * (y - t)

    Tensor gradOutput = output_ - target;
    double N = output_.size();
    double loss = 0.0;

    for(size_t i=0; i<gradOutput.size(); ++i) {
        double diff = gradOutput.data()[i];
        loss += diff * diff;
        gradOutput.data()[i] = 2.0 * diff / N;
    }
    loss /= N;

    // Output Head Backward
    Tensor dFinalBlockOutput = gradOutput.matmul(W_out_.transpose());
    Tensor dW_out = finalBlockOutput_.transpose().matmul(gradOutput);

    // db_out
    Tensor db_out(1, 1, 1);
    for(size_t i=0; i<gradOutput.size(); ++i) db_out.data()[0] += gradOutput.data()[i];

    W_out_ -= dW_out * learningRate;
    b_out_ -= db_out * learningRate;

    // Blocks Backward
    Tensor dX = dFinalBlockOutput;
    for (int i = (int)blocks_.size() - 1; i >= 0; --i) {
        dX = blocks_[i].backward(dX, learningRate);
    }

    // Pos Encoding (Fixed, no grad)
    Tensor dEmbedded = dX;

    // Embedding Backward
    Tensor dW_embed = input_.transpose().matmul(dEmbedded);

    // db_embed
    Tensor db_embed(1, 1, d_model_);
    for(size_t c=0; c<dEmbedded.channels(); ++c)
        for(size_t h=0; h<dEmbedded.height(); ++h)
            for(size_t w=0; w<dEmbedded.width(); ++w)
                db_embed(0, 0, w) += dEmbedded(c, h, w);

    W_embed_ -= dW_embed * learningRate;
    b_embed_ -= db_embed * learningRate;

    return loss;
}
