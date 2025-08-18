# {{project_name}}

## {{project_description}}

![PyPI - License](https://img.shields.io/pypi/l/resolution-aware-transformer)
[![CI/CD Pipeline](https://github.com/rhoadesScholar/resolution-aware-transformer/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/rhoadesScholar/resolution-aware-transformer/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/github/rhoadesScholar/resolution-aware-transformer/graph/badge.svg?token=)](https://codecov.io/github/rhoadesScholar/resolution-aware-transformer)
![PyPI - Version](https://img.shields.io/pypi/v/resolution-aware-transformer)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/resolution-aware-transformer)

A Python package for pytorch implementation of a resolution aware transformer, designed to take in multiple scales of fixed-resolution images (such as microscopy or medical imaging), with the goal of producing embeddings informed by global context and local details. utilizes rotary spatial embeddings (rose) and spatial grouping attention..

## Installation

### From PyPI

```bash
pip install resolution-aware-transformer
```

### From source

```bash
pip install git+https://github.com/rhoadesScholar/resolution-aware-transformer.git
```

## Usage

```python
import resolution_aware_transformer

# Example usage
# TODO: Add your usage examples here
```

## Development

### Setting up development environment

```bash
# Clone the repository
git clone https://github.com/rhoadesScholar/resolution-aware-transformer.git
cd resolution-aware-transformer

# Install in development mode with all dependencies
make dev-setup
```

### Running tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run fast tests (stop on first failure)
make test-fast
```

### Code quality

```bash
# Format code
make format

# Lint code
make lint

# Type check
make type-check

# Run all checks
make check-all
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the test suite (`make test`)
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite it using the information in [CITATION.cff](CITATION.cff).
