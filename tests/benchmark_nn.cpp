#include "neural_network.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

int main() {
    std::cout << "Starting Neural Network Benchmark..." << std::endl;

    NeuralNetwork nn;
    // Larger network to make the impact of optimization more visible
    // 500 inputs -> 1000 hidden -> 1000 hidden -> 500 outputs
    // Weights: 500*1000 + 1000*1000 + 1000*500 = 500k + 1M + 500k = 2M weights
    int inSize = 500;
    int hiddenSize = 1000;
    int outSize = 500;

    nn.setInputSize(inSize);
    nn.addLayer(hiddenSize, ActivationType::Sigmoid);
    nn.addLayer(hiddenSize, ActivationType::Sigmoid);
    nn.addLayer(outSize, ActivationType::Sigmoid);
    nn.build();

    int samples = 200;
    int epochs = 5;

    // Generate data
    std::vector<std::vector<double>> inputs(samples, std::vector<double>(inSize));
    std::vector<std::vector<double>> targets(samples, std::vector<double>(outSize));

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::cout << "Generating " << samples << " samples..." << std::endl;
    for (int i = 0; i < samples; ++i) {
        for (int j = 0; j < inSize; ++j) {
            inputs[i][j] = dis(gen);
        }
        for (int j = 0; j < outSize; ++j) {
             targets[i][j] = (inputs[i][j % inSize] + 1.0) / 2.0;
        }
    }

    std::cout << "Training for " << epochs << " epochs..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int e = 0; e < epochs; ++e) {
        nn.train(inputs, targets, 0.01);
        // std::cout << "Epoch " << e + 1 << " complete." << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Training took " << diff.count() << " s" << std::endl;

    return 0;
}
