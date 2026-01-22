#include <iostream>
#include "cnn/cnn_network.h"
#include "cnn/tensor.h"

int main() {
    try {
        std::cout << "=== CNN Diagnostic Test ===" << std::endl;
        
        // Test 1: Create network
        std::cout << "Test 1: Creating CNN network..." << std::endl;
        CNNNetwork cnn;
        cnn.setInputSize(1, 16, 16);
        cnn.addConvLayer(8, 3, 1, 1, CNNActivationType::ReLU);
        cnn.addPoolingLayer(2, 2, PoolingType::Max);
        cnn.addDenseLayer(10, ActivationType::ReLU);
        cnn.addDenseLayer(3, ActivationType::Sigmoid);
        
        std::cout << "Test 2: Building network..." << std::endl;
        cnn.build();
        std::cout << "  ✓ Network built successfully" << std::endl;
        
        // Test 3: Forward pass
        std::cout << "Test 3: Testing forward pass..." << std::endl;
        Tensor input(1, 16, 16, 0.5);
        auto output = cnn.forward(input);
        std::cout << "  ✓ Forward pass successful, output size: " << output.size() << std::endl;
        
        // Test 4: Backward pass
        std::cout << "Test 4: Testing backward pass..." << std::endl;
        std::vector<double> target = {1.0, 0.0, 0.0};
        cnn.backward(target);
        std::cout << "  ✓ Backward pass successful" << std::endl;
        
        // Test 5: Update weights
        std::cout << "Test 5: Testing weight update..." << std::endl;
        cnn.updateWeights(0.01);
        std::cout << "  ✓ Weight update successful" << std::endl;
        
        // Test 6: Full training loop
        std::cout << "Test 6: Testing full training loop..." << std::endl;
        std::vector<Tensor> inputs;
        std::vector<std::vector<double>> targets;
        
        for (int i = 0; i < 5; ++i) {
            inputs.push_back(Tensor(1, 16, 16, 0.5));
            targets.push_back({1.0, 0.0, 0.0});
        }
        
        double loss = cnn.train(inputs, targets, 0.01);
        std::cout << "  ✓ Training successful, loss: " << loss << std::endl;
        
        std::cout << "\n=== All tests passed! ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERROR: " << e.what() << std::endl;
        return 1;
    }
}
