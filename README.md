# Resolution Aware Transformer

A PyTorch implementation of a resolution-aware transformer designed for multi-scale image analysis, particularly suited for microscopy and medical imaging applications.

![GitHub - License](https://img.shields.io/github/license/rhoadesScholar/resolution-aware-transformer)
[![CI/CD Pipeline](https://github.com/rhoadesScholar/resolution-aware-transformer/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/rhoadesScholar/resolution-aware-transformer/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/github/rhoadesScholar/resolution-aware-transformer/graph/badge.svg?token=)](https://codecov.io/github/rhoadesScholar/resolution-aware-transformer)
![PyPI - Version](https://img.shields.io/pypi/v/resolution-aware-transformer)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/resolution-aware-transformer)

## Overview

The Resolution Aware Transformer is a PyTorch neural network module that processes multi-scale images to produce embeddings informed by both global context and local details. It combines:

- **Spatial Grouping Attention (SGA)**: Groups spatially-related features for efficient attention computation
- **Rotary Spatial Embeddings (RoSE)**: Provides spatial awareness through rotary position encoding
- **Multi-resolution Processing**: Handles image pyramids with different resolutions seamlessly

This architecture is particularly effective for:

- **Microscopy imaging**: Processing high-resolution biological images with multiple scales
- **Medical imaging**: Analyzing medical scans with varying levels of detail
- **Computer vision tasks**: Any application requiring both global context and fine-grained spatial details

## Features

- **Multi-scale Processing**: Handles single images or image pyramids with different resolutions
- **Spatial Awareness**: Uses Rotary Spatial Embeddings (RoSE) for position-aware attention
- **Flexible Attention**: Choice between dense and sparse spatial grouping attention
- **Real-world Coordinates**: Supports physical pixel spacing for medical/scientific imaging
- **Masking Support**: Handle irregular regions or missing data with optional masks
- **2D and 3D Support**: Works with both 2D images and 3D volumes
- **GPU Optimized**: Built on PyTorch with efficient attention mechanisms

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `spatial_dims` | Number of spatial dimensions (2 or 3) | 2 |
| `input_features` | Number of input channels | 3 |
| `feature_dims` | Embedding dimension | 128 |
| `num_blocks` | Number of transformer blocks | 4 |
| `sga_attention_type` | "dense" or "sparse" attention | "dense" |
| `num_heads` | Number of attention heads | 16 |
| `kernel_size` | Convolution kernel size for downsampling | 7 |
| `mlp_ratio` | MLP hidden dimension ratio | 4 |
| `learnable_rose` | Use learnable rotary embeddings | True |

## Installation

### From PyPI

```bash
pip install resolution-aware-transformer
```

### From source

```bash
pip install git+https://github.com/rhoadesScholar/resolution-aware-transformer.git
```

## Requirements

- Python ≥ 3.10
- PyTorch
- [spatial-grouping-attention](https://github.com/rhoadesScholar/spatial-grouping-attention)
- [rotary-spatial-embeddings](https://github.com/rhoadesScholar/rotary-spatial-embeddings)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/rhoadesScholar/resolution-aware-transformer.git
cd resolution-aware-transformer

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 .
black .
isort .
```

## Usage

### Basic Usage

```python
import torch
from resolution_aware_transformer import ResolutionAwareTransformer

# Initialize the model
model = ResolutionAwareTransformer(
    spatial_dims=2,          # 2D images (use 3 for 3D volumes)
    input_features=3,        # RGB images (adjust for your data)
    feature_dims=128,        # Embedding dimension
    num_blocks=4,            # Number of transformer blocks
    num_heads=16,            # Attention heads
    sga_attention_type="dense"   # "dense" or "sparse"
)

# Single image input
image = torch.randn(1, 3, 256, 256)  # [batch, channels, height, width]
output = model(image)

# Multi-scale image pyramid input
image_pyramid = [
    torch.randn(1, 3, 256, 256),  # High resolution
    torch.randn(1, 3, 128, 128),  # Medium resolution
    torch.randn(1, 3, 64, 64)     # Low resolution
]
output = model(image_pyramid)
```

### Advanced Usage with Spacing and Masks

```python
# For medical/microscopy images with known pixel spacing
spacing = [0.5, 0.5]  # μm per pixel in x, y dimensions

# Optional masks for irregular regions
mask = torch.ones(1, 256, 256)  # Valid regions = 1, invalid = 0

output = model(
    image,
    input_spacing=spacing,
    mask=mask
)

# Each output contains embeddings and attention maps
for scale_output in output:
    embeddings = scale_output['x_out']        # [batch, num_patches, feature_dims]
    spacing_info = scale_output['out_spacing'] # Pixel spacing for this scale
    grid_shape = scale_output['out_grid_shape'] # Spatial dimensions
```

### 3D Volume Processing

```python
# For 3D medical volumes or microscopy stacks
model_3d = ResolutionAwareTransformer(
    spatial_dims=3,
    input_features=1,        # Grayscale volumes
    feature_dims=256,
    num_blocks=6
)

volume = torch.randn(1, 1, 64, 64, 64)  # [batch, channels, depth, height, width]
output = model_3d(volume)

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite it using the information in [CITATION.cff](CITATION.cff) or use the following BibTeX:

```bibtex
@software{rhoades_resolution_aware_transformer,
  author = {Rhoades, Jeff},
  title = {Resolution Aware Transformer: A PyTorch implementation of a resolution-aware transformer for multi-scale image analysis},
  url = {https://github.com/rhoadesScholar/resolution-aware-transformer},
  version = {2025.8.19.420},
  year = {2025}
}
```

## Acknowledgments

This implementation builds upon research in spatial attention mechanisms and transformer architectures for computer vision. Special thanks to the PyTorch community and contributors to the spatial-grouping-attention and rotary-spatial-embeddings packages.
