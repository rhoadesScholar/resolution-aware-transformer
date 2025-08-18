# resolution-aware-transformer Documentation

Welcome to the resolution-aware-transformer documentation!

## Overview

PyTorch implementation of a Resolution Aware Transformer, designed to take in multiple scales of fixed-resolution images (such as microscopy or medical imaging), with the goal of producing embeddings informed by global context and local details. Utilizes rotary spatial embeddings (RoSE) and spatial grouping attention.

## Table of Contents

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [API Reference](api.md)
- [Examples](examples.md)
- [Contributing](contributing.md)

## Getting Started

A Python package for pytorch implementation of a resolution aware transformer, designed to take in multiple scales of fixed-resolution images (such as microscopy or medical imaging), with the goal of producing embeddings informed by global context and local details. utilizes rotary spatial embeddings (rose) and spatial grouping attention..

### Installation

```bash
pip install resolution-aware-transformer
```

### Quick Example

```python
import resolution_aware_transformer

# Your example code here
obj = resolution_aware_transformer.ResolutionAwareTransformer(param1="example")
result = obj.process()
print(result)
```

## Support

If you encounter any issues or have questions, please:

1. Check the [documentation](https://github.com/rhoadesScholar/resolution-aware-transformer)
2. Search [existing issues](https://github.com/rhoadesScholar/resolution-aware-transformer/issues)
3. Create a [new issue](https://github.com/rhoadesScholar/resolution-aware-transformer/issues/new)

## License

This project is licensed under the BSD 3-Clause License License - see the [LICENSE](../LICENSE) file for details.
