#include "attention/transformer_block.h"
#include "cnn/random.h"
#include <cmath>

TransformerBlock::TransformerBlock(size_t d_model, size_t d_k, size_t d_ff)
    : d_model_(d_model), attention_(d_model, d_k) {

    // Initialize FF Weights
    size_t fanIn = d_model;
    size_t fanOut = d_ff;
    W1_ = Tensor(1, d_model, d_ff); W1_.xavierInit(fanIn, fanOut);
    W2_ = Tensor(1, d_ff, d_model); W2_.xavierInit(fanOut, fanIn);

    // Biases (initialized to 0)
    b1_ = Tensor(1, 1, d_ff);
    b2_ = Tensor(1, 1, d_model);

    // Layer Norm
    gamma1_ = Tensor(1, 1, d_model, 1.0);
    beta1_ = Tensor(1, 1, d_model, 0.0);
    gamma2_ = Tensor(1, 1, d_model, 1.0);
    beta2_ = Tensor(1, 1, d_model, 0.0);
}

// Helper: Broadcast add bias (1, 1, W) to (1, H, W)
static void addBias(Tensor& x, const Tensor& b) {
    for (size_t c = 0; c < x.channels(); ++c) {
        for (size_t h = 0; h < x.height(); ++h) {
            for (size_t w = 0; w < x.width(); ++w) {
                x(c, h, w) += b(0, 0, w);
            }
        }
    }
}

// Helper: Sum gradients for bias (1, H, W) -> (1, 1, W)
static Tensor sumGradBias(const Tensor& grad) {
    Tensor db(1, 1, grad.width());
    for (size_t c = 0; c < grad.channels(); ++c) {
        for (size_t h = 0; h < grad.height(); ++h) {
            for (size_t w = 0; w < grad.width(); ++w) {
                db(0, 0, w) += grad(c, h, w);
            }
        }
    }
    return db;
}

Tensor TransformerBlock::forwardLayerNorm(const Tensor& x, const Tensor& gamma, const Tensor& beta) {
    Tensor out(x.channels(), x.height(), x.width());

    for (size_t c = 0; c < x.channels(); ++c) {
        for (size_t h = 0; h < x.height(); ++h) {
            // Compute mean
            double sum = 0.0;
            for (size_t w = 0; w < x.width(); ++w) {
                sum += x(c, h, w);
            }
            double mean = sum / x.width();

            // Compute variance
            double sqSum = 0.0;
            for (size_t w = 0; w < x.width(); ++w) {
                double d = x(c, h, w) - mean;
                sqSum += d * d;
            }
            double var = sqSum / x.width();
            double stdDev = std::sqrt(var + 1e-9);

            // Normalize
            for (size_t w = 0; w < x.width(); ++w) {
                double normalized = (x(c, h, w) - mean) / stdDev;
                out(c, h, w) = normalized * gamma(0, 0, w) + beta(0, 0, w);
            }
        }
    }
    return out;
}

Tensor TransformerBlock::backwardLayerNorm(const Tensor& dY, const Tensor& x, const Tensor& gamma, Tensor& dGamma, Tensor& dBeta) {
    Tensor dX(x.channels(), x.height(), x.width());

    for (size_t c = 0; c < x.channels(); ++c) {
        for (size_t h = 0; h < x.height(); ++h) {
            double sum = 0.0;
            for (size_t w = 0; w < x.width(); ++w) sum += x(c, h, w);
            double mean = sum / x.width();

            double sqSum = 0.0;
            for (size_t w = 0; w < x.width(); ++w) {
                double d = x(c, h, w) - mean;
                sqSum += d * d;
            }
            double var = sqSum / x.width();
            double stdDev = std::sqrt(var + 1e-9);
            double invStd = 1.0 / stdDev;

            double dSigma = 0.0;
            double dMu = 0.0;

            for (size_t w = 0; w < x.width(); ++w) {
                double x_hat = (x(c, h, w) - mean) * invStd;
                double dy = dY(c, h, w);

                // Grads for gamma/beta
                dGamma(0, 0, w) += dy * x_hat;
                dBeta(0, 0, w) += dy;

                double dx_hat = dy * gamma(0, 0, w);

                dSigma += dx_hat * (x(c, h, w) - mean);
                dMu -= dx_hat * invStd;
            }

            dSigma *= -0.5 * std::pow(var + 1e-9, -1.5);

            for (size_t w = 0; w < x.width(); ++w) {
                double d = x(c, h, w) - mean;
                // dL/dx_i term 1: dL/dMu * 1/N
                // term 2: dL/dSigma * 2(x-mu)/N
                // term 3: dL/dx_hat * 1/sigma

                // Standard LayerNorm backward formula
                double dx_hat = dY(c, h, w) * gamma(0, 0, w);
                // Recomputing easier way:
                // dx = (1/N) * invStd * (N*dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
            }

            // Correct simplified implementation
            double sum_dx_hat = 0.0;
            double sum_dx_hat_x_hat = 0.0;

            for (size_t w = 0; w < x.width(); ++w) {
                double x_hat = (x(c, h, w) - mean) * invStd;
                double dx_hat = dY(c, h, w) * gamma(0, 0, w);
                sum_dx_hat += dx_hat;
                sum_dx_hat_x_hat += dx_hat * x_hat;
            }

            for (size_t w = 0; w < x.width(); ++w) {
                double x_hat = (x(c, h, w) - mean) * invStd;
                double dx_hat = dY(c, h, w) * gamma(0, 0, w);
                double val = (1.0 / x.width()) * invStd *
                             (x.width() * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x_hat);
                dX(c, h, w) = val;
            }
        }
    }
    return dX;
}

Tensor TransformerBlock::forward(const Tensor& input) {
    input_ = input;

    // 1. Attention
    attnOutput_ = attention_.forward(input);

    // 2. Add & Norm
    norm1Input_ = input + attnOutput_;
    norm1Output_ = forwardLayerNorm(norm1Input_, gamma1_, beta1_);

    // 3. Feed Forward
    // Dense 1
    ffHidden_ = norm1Output_.matmul(W1_);
    addBias(ffHidden_, b1_);

    // ReLU
    ffRelu_ = ffHidden_.map([](double x) { return x > 0 ? x : 0; });

    // Dense 2
    ffOutput_ = ffRelu_.matmul(W2_);
    addBias(ffOutput_, b2_);

    // 4. Add & Norm
    norm2Input_ = norm1Output_ + ffOutput_;
    Tensor output = forwardLayerNorm(norm2Input_, gamma2_, beta2_);

    return output;
}

Tensor TransformerBlock::backward(const Tensor& gradOutput, double lr) {
    // 1. Layer Norm 2 Backward
    Tensor dGamma2(1, 1, d_model_);
    Tensor dBeta2(1, 1, d_model_);

    Tensor dNorm2Input = backwardLayerNorm(gradOutput, norm2Input_, gamma2_, dGamma2, dBeta2);

    // Update Norm params
    gamma2_ -= dGamma2 * lr;
    beta2_ -= dBeta2 * lr;

    // 2. Add Branch (Residual)
    // dNorm2Input goes to both ffOutput and norm1Output
    Tensor dFFOutput = dNorm2Input;
    Tensor dNorm1Output_branch2 = dNorm2Input;

    // 3. Feed Forward Backward
    // Dense 2
    Tensor dFFRelu = dFFOutput.matmul(W2_.transpose());
    Tensor dW2 = ffRelu_.transpose().matmul(dFFOutput);
    Tensor db2 = sumGradBias(dFFOutput);

    W2_ -= dW2 * lr;
    b2_ -= db2 * lr;

    // ReLU
    Tensor dFFHidden = dFFRelu;
    for(size_t i=0; i<dFFHidden.size(); ++i) {
        if (ffHidden_.data()[i] <= 0) dFFHidden.data()[i] = 0;
    }

    // Dense 1
    Tensor dNorm1Output_branch1 = dFFHidden.matmul(W1_.transpose());
    Tensor dW1 = norm1Output_.transpose().matmul(dFFHidden);
    Tensor db1 = sumGradBias(dFFHidden);

    W1_ -= dW1 * lr;
    b1_ -= db1 * lr;

    // Sum gradients at Norm1 Output
    Tensor dNorm1Output = dNorm1Output_branch1 + dNorm1Output_branch2;

    // 4. Layer Norm 1 Backward
    Tensor dGamma1(1, 1, d_model_);
    Tensor dBeta1(1, 1, d_model_);
    Tensor dNorm1Input = backwardLayerNorm(dNorm1Output, norm1Input_, gamma1_, dGamma1, dBeta1);

    gamma1_ -= dGamma1 * lr;
    beta1_ -= dBeta1 * lr;

    // 5. Add Branch (Residual)
    // dNorm1Input goes to Input and AttnOutput
    Tensor dAttnOutput = dNorm1Input;
    Tensor dInput_branch2 = dNorm1Input;

    // 6. Attention Backward
    Tensor dInput_branch1 = attention_.backward(dAttnOutput, lr);

    // Sum gradients at Input
    Tensor dInput = dInput_branch1 + dInput_branch2;

    return dInput;
}
