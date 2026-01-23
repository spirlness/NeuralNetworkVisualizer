#include <iostream>
#include <vector>
#include <chrono>
#include "cnn/cnn_network.h"
#include "cnn/tensor.h"

// Simple benchmark for CNN forward pass
int main() {
    std::cout << "Starting CNN Benchmark..." << std::endl;

    try {
        CNNNetwork network;
        // Use a reasonably large input to make data copying significant
        // 3 channels, 64x64 image -> 12288 elements
        size_t channels = 3;
        size_t height = 64;
        size_t width = 64;

        network.setInputSize(channels, height, width);

        // Add layers to ensure we have processing
        network.addConvLayer(16, 3, 1, 1, CNNActivationType::ReLU);
        network.addPoolingLayer(2, 2, PoolingType::Max); // Output: 16 x 32 x 32

        // Flatten layer is added automatically or manually before Dense
        // network.addFlattenLayer(); // Optional, addDenseLayer does it if needed

        // Add dense layers
        // Input to dense will be 16 * 32 * 32 = 16384 elements
        network.addDenseLayer(128, ActivationType::ReLU);
        network.addDenseLayer(10, ActivationType::Sigmoid);

        network.build();

        // Prepare input
        Tensor input(channels, height, width);
        input.randomInit();

        // Warmup
        for (int i = 0; i < 10; ++i) {
            network.forward(input);
        }

        // Benchmark loop
        int iterations = 1000;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            network.forward(input);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << "Total time for " << iterations << " iterations: " << elapsed.count() << " ms" << std::endl;
        std::cout << "Average time per iteration: " << elapsed.count() / iterations << " ms" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
