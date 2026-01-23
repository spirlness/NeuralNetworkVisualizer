---
## [BUG] Segmentation Fault in AttentionNetwork with Batch Size > 1

**Severity (严重程度):** Critical
**File/Location:** `src/attention/attention_network.cpp : 66`

### Description (问题描述)
The `AttentionNetwork` fails when processing a batch size greater than 1 (i.e., input tensor has `channels > 1`). In the `forward` method, the code attempts to add `posEncoding_` (which is initialized with 1 channel) to `embedded_` (which has `C` channels corresponding to the batch size).

The `Tensor::operator+` calls `Tensor::operator+=`, which iterates up to `this->data_.size()` (the size of `embedded_`, which is `C * L * D`). However, `posEncoding_` only has `1 * L * D` elements. This results in an **Out-of-Bounds Read** on `posEncoding_.data_`, causing a segmentation fault or undefined behavior (data corruption) when `C > 1`.

### Problematic Code (问题代码)
```cpp
    // embedded_ has shape (C, L, D) where C is batch size
    // posEncoding_ has shape (1, L, D)

    // Add Pos Encoding
    blocksInput_ = embedded_ + posEncoding_; // <--- CRASH: Accessing posEncoding_ out of bounds
```

---
## [BUG] Incorrect Gradient Aggregation in Attention Layers for Batch Processing

**Severity (严重程度):** High
**File/Location:** `src/attention/attention_layer.cpp : 115`

### Description (问题描述)
The gradient update logic for shared weights in `AttentionLayer` (and `TransformerBlock`) is functionally incorrect for batch sizes greater than 1.

During backpropagation, `dW_Q` (gradient of query weights) is calculated and has the shape `(C, D, K)`, preserving the gradients for each item in the batch. However, the weight parameter `W_Q_` has the shape `(1, D, K)`.

When performing `W_Q_ -= dW_Q * learningRate;`, the `Tensor::operator-=` method iterates based on `W_Q_.size()` (which is `1 * D * K`). Consequently, it updates the weights using **only the gradients from the first batch item** (the first channel of `dW_Q`). The gradients from the rest of the batch are completely ignored, leading to incorrect training behavior and poor convergence when using batch processing.

### Problematic Code (问题代码)
```cpp
    // Update Weights
    // W_Q_ shape: (1, D, K)
    // dW_Q shape: (C, D, K) if batch size > 1

    W_Q_ -= dW_Q * learningRate; // <--- LOGIC ERROR: Only updates using first batch item's gradient
```

---
## [BUG] Thread Safety Violation and Race Condition in PoolingLayer

**Severity (严重程度):** Medium
**File/Location:** `src/cnn/pooling_layer.cpp : 52`

### Description (问题描述)
The `PoolingLayer` class is designed with stateful member variables (`maxIndices_`) that persist between `forward` and `backward` passes. This design makes the layer **thread-unsafe** and prevents concurrent usage (e.g., pipelined inference or training) even if the enclosing network locks its own state, unless strictly serialized.

Specifically, `maxIndices_` is resized and populated during `forward`. If `forward` is called again (e.g., by a separate thread or process for inference) before the corresponding `backward` pass is completed for the previous input, the `maxIndices_` will be overwritten. This results in the `backward` pass using incorrect indices for gradient calculation, leading to corrupted gradients.

### Problematic Code (问题代码)
```cpp
    if (poolType_ == PoolingType::Max) {
        maxIndices_.resize(inputChannels_); // <--- STATEFUL: Modifies member variable
        for (size_t c = 0; c < inputChannels_; ++c) {
            // ...
            maxIndices_[c][oh][ow] = {maxH, maxW};
        }
    }
```

---
## [BUG] Buffer Overflow Vulnerability in TransformerBlock Input Processing

**Severity (严重程度):** Medium
**File/Location:** `src/attention/transformer_block.cpp : 44`

### Description (问题描述)
The `TransformerBlock::forwardLayerNorm` method lacks input validation. It iterates over the width of the input tensor `x` and accesses the `gamma` and `beta` tensors using the loop index `w`.

The `gamma` and `beta` tensors are initialized with a width equal to `d_model`. If the input tensor `x` has a width larger than `d_model`, the access `gamma(0, 0, w)` will go out of bounds of the `gamma` tensor's data buffer. This can lead to a crash (segfault) or reading garbage data.

### Problematic Code (问题代码)
```cpp
            for (size_t w = 0; w < x.width(); ++w) { // <--- Iterates based on input width
                double normalized = (x(c, h, w) - mean) / stdDev;
                // If x.width() > d_model, this accesses gamma out of bounds
                out(c, h, w) = normalized * gamma(0, 0, w) + beta(0, 0, w);
            }
```

---
## [BUG] Integer Overflow in Tensor Memory Allocation

**Severity (严重程度):** Low
**File/Location:** `src/cnn/tensor.cpp : 10`

### Description (问题描述)
The `Tensor` constructor calculates the total size of the data buffer by multiplying `channels * height * width`. Since these parameters are `size_t`, the multiplication is performed using `size_t` arithmetic.

However, there is no check for overflow. If the product exceeds the maximum value representable by `size_t` (or wraps around modulo $2^{64}$ or $2^{32}$), the allocated `data_` vector will be much smaller than required. Subsequent access to this tensor using valid `(c, h, w)` coordinates (which are mapped to a linear index) will likely result in a **Heap Buffer Overflow**, crashing the application or allowing arbitrary code execution.

### Problematic Code (问题代码)
```cpp
Tensor::Tensor(size_t channels, size_t height, size_t width)
    : channels_(channels), height_(height), width_(width),
      data_(channels * height * width, 0.0) {} // <--- OVERFLOW: No check for multiplication overflow
```
