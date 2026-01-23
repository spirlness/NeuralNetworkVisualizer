#include "attention/attention_layer.h"
#include "cnn/random.h"
#include <cmath>

AttentionLayer::AttentionLayer(size_t d_model, size_t d_k)
    : d_model_(d_model), d_k_(d_k) {

    // Initialize weights
    // Xavier initialization
    size_t fanIn = d_model;
    size_t fanOut = d_k;

    W_Q_ = Tensor(1, d_model, d_k); W_Q_.xavierInit(fanIn, fanOut);
    W_K_ = Tensor(1, d_model, d_k); W_K_.xavierInit(fanIn, fanOut);
    W_V_ = Tensor(1, d_model, d_k); W_V_.xavierInit(fanIn, fanOut);
    W_O_ = Tensor(1, d_k, d_model); W_O_.xavierInit(fanOut, fanIn);
}

Tensor AttentionLayer::forward(const Tensor& input) {
    input_ = input; // Cache (1, Seq, D_model)

    // Linear projections
    // Input is (1, L, D). W is (1, D, K).
    // Matmul: (1, L, D) * (1, D, K) -> (1, L, K)
    Q_ = input.matmul(W_Q_);
    K_ = input.matmul(W_K_);
    V_ = input.matmul(W_V_);

    // Scaled Dot-Product Attention
    // Q: (1, L, K). K^T: (1, K, L).
    // Scores: (1, L, L)
    Tensor K_T = K_.transpose();
    scores_ = Q_.matmul(K_T);
    scores_ *= (1.0 / std::sqrt(static_cast<double>(d_k_)));

    // Softmax
    attentionWeights_ = scores_; // copy
    attentionWeights_.softmax();

    // Output
    // Weights: (1, L, L). V: (1, L, K) -> Output: (1, L, K)
    Tensor context = attentionWeights_.matmul(V_);

    // Final projection
    // Context (1, L, K) * W_O (1, K, D) -> (1, L, D)
    Tensor output = context.matmul(W_O_);

    return output;
}

Tensor AttentionLayer::backward(const Tensor& gradOutput, double learningRate) {
    // Simple gradient descent implementation (Backpropagation)

    // 1. Gradient of W_O
    // Output = Context * W_O
    // dW_O = Context^T * gradOutput
    // dContext = gradOutput * W_O^T
    Tensor context = attentionWeights_.matmul(V_); // Recompute context
    Tensor dW_O = context.transpose().matmul(gradOutput);
    Tensor dContext = gradOutput.matmul(W_O_.transpose());

    // 2. Gradient of Attention Weights and V
    // Context = Weights * V
    // dV = Weights^T * dContext
    // dWeights = dContext * V^T
    Tensor dV = attentionWeights_.transpose().matmul(dContext);
    Tensor dWeights = dContext.matmul(V_.transpose());

    // 3. Gradient of Softmax (Scores)
    // dScores = Weights * (dWeights - sum(dWeights * Weights))
    // Note: simplified row-wise softmax gradient
    Tensor dScores = dWeights; // placeholder for shape
    for(size_t c=0; c<dScores.channels(); ++c) {
        for(size_t h=0; h<dScores.height(); ++h) {
            double sum_grad_p = 0.0;
            for(size_t w=0; w<dScores.width(); ++w) {
                sum_grad_p += dWeights(c, h, w) * attentionWeights_(c, h, w);
            }
            for(size_t w=0; w<dScores.width(); ++w) {
                double s = attentionWeights_(c, h, w);
                dScores(c, h, w) = s * (dWeights(c, h, w) - sum_grad_p);
            }
        }
    }

    // 4. Scaling
    dScores *= (1.0 / std::sqrt(static_cast<double>(d_k_)));

    // 5. Gradient of Q and K
    // Scores = Q * K^T
    // dQ = dScores * K
    // dK^T = Q^T * dScores -> dK = (Q^T * dScores)^T = dScores^T * Q
    Tensor dQ = dScores.matmul(K_);
    Tensor dK = dScores.transpose().matmul(Q_);

    // 6. Gradient of Weights Q, K, V
    // Q = Input * W_Q -> dW_Q = Input^T * dQ, dInput_Q = dQ * W_Q^T
    Tensor input_T = input_.transpose();
    Tensor dW_Q = input_T.matmul(dQ);
    Tensor dW_K = input_T.matmul(dK);
    Tensor dW_V = input_T.matmul(dV);

    Tensor dInput = dQ.matmul(W_Q_.transpose()) +
                    dK.matmul(W_K_.transpose()) +
                    dV.matmul(W_V_.transpose());

    // Update Weights
    W_Q_ -= dW_Q * learningRate;
    W_K_ -= dW_K * learningRate;
    W_V_ -= dW_V * learningRate;
    W_O_ -= dW_O * learningRate;

    return dInput;
}
