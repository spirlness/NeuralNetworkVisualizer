# CNN VISUALIZATION MODULE

**Path:** `include/visualization/`, `src/visualization/`

## OVERVIEW
Custom Qt widgets for real-time 3D CNN architecture and feature map visualization using QPainter.

## COMPONENTS
- **CNNView**:
    - Renders 3D layer boxes with depth proportional to channel count.
    - Uses cubic Bezier curves for inter-layer connections.
    - Generates dynamic thumbnails (12x12) for first 4 feature maps of each layer.
    - Emits `layerClicked(int)` for interaction with training dashboard.
- **FeatureMapView**:
    - Displays all layer channels in a configurable grid layout.
    - Renders QImage thumbnails per channel with selected color mapping.
    - Auto-scales scroll area height based on channel count and `gridColumns_`.
    - Highlights selection via `channelSelected(int)` signal and yellow borders.

## RENDERING DETAILS
- **3D Projection**: Manual orthographic projection in `draw3DBox` using `dx/dy` offsets.
- **Layer Colors**:
    - Conv: Steel Blue (`#4682B4`)
    - Pool: Sea Green (`#2E8B57`) / Medium Sea Green (`#3CB371`)
    - Flatten: Orange (`#FFA500`)
- **Performance**: Downsamples large tensors to thumbnail size via nearest-neighbor sampling.
- **Themes**: Hardcoded dark backgrounds (`#1e1e2e`, `#252536`) to ensure visual consistency.

## COLOR MAPPING
- **Heatmap**: Multi-stage interpolation (Blue -> Cyan -> Green -> Yellow -> Red).
- **Grayscale**: Linear mapping of normalized [0, 1] tensor values to RGB triplets.
- **Viridis**: Polynomial approximation for high-contrast feature inspection.
    - *R:* `0.267 + 0.329v + 1.260v² - 1.856v³`
    - *G:* `0.004 + 1.016v - 0.316v²`
    - *B:* `0.329 + 0.424v - 0.753v² + 0.401v³`
- **Normalization**: Values are clamped and normalized based on global min/max of the current Tensor.
