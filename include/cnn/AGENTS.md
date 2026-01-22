# CNN CORE MODULE

**OVERVIEW**: Polymorphic C++17 spatial feature extraction engine using CHW tensor orchestration.

## LAYERS
| Layer | Class | Input | Output | Details |
|-------|-------|-------|--------|---------|
| **Convolution** | `ConvolutionalLayer` | 3D Tensor | 3D Tensor | Cross-correlation, supports padding/stride, He init |
| **Pooling** | `PoolingLayer` | 3D Tensor | 3D Tensor | Max or Average downsampling |
| **Flatten** | `FlattenLayer` | 3D Tensor | 1D Vector | Dimensionality reduction bridge to classification head |

## TENSOR FORMAT
- **Strict CHW**: Data ordered as Channels (C), then Height (H), then Width (W).
- **Indexing**: `data[c * H * W + h * W + w]` (internally managed by `Tensor` class).
- **Bounds**: `Tensor::at()` provides `std::out_of_range` protection.
- **Normalization**: Expected input range is [0.0, 1.0].

## CONVENTIONS
- **Topology**: MUST call `addFlattenLayer()` before any `addDenseLayer()`.
- **Constraint**: `CNNNetwork` throws `runtime_error` if spatial layers (Conv/Pool) are added after Flatten.
- **Initialization**: 
  - `ConvolutionalLayer`: He initialization for ReLU stability.
  - classification head: Xavier initialization for dense weights.
- **Safety**: Inputs to `exp()` in activations MUST be clamped via `std::clamp(val, -500.0, 500.0)`.

## GRADIENT FLOW
- **Classification to Spatial**: `FlattenLayer::backward` reshapes the 1D gradient vector back to 3D CHW `Tensor`.
- **Param Update**: `updateWeights(lr)` propagates through polymorphic `CNNLayerPtr` vector.
- **Bridging**: Dense layer gradients (from `neural_network.h` logic) are mapped to `FlattenLayer`'s output gradient.

## FILE MAP
- `tensor.h/cpp`: 3D math and CHW storage.
- `cnn_layer_base.h`: `CNNLayerBase` polymorphic interface.
- `conv_layer.h/cpp`: Convolutional operations and kernel weight management.
- `pooling_layer.h/cpp`: Spatial pooling implementations (Max/Avg).
- `flatten_layer.h/cpp`: 3D -> 1D bridge for gradient and forward paths.
- `cnn_network.h/cpp`: Hybrid graph manager for CNN + Dense layer stacks.
