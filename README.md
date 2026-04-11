<p align="center">
  <img src="https://raw.githubusercontent.com/James-Wirth/pocket-inference/main/assets/logo.png" alt="Pocket Inference" width="400">
</p>

---

<h3 align="center">A lightweight Rust library for running Keras models without the TensorFlow stack.</h3>

<p align="center">
  <a href="https://pypi.org/project/pocket-inference/"><img src="https://img.shields.io/pypi/v/pocket-inference.svg" alt="PyPI"></a>
  <a href="https://github.com/James-Wirth/pocket-inference/actions/workflows/ci.yml"><img src="https://github.com/James-Wirth/pocket-inference/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <strong>Binary size: ~450 KB</strong> · Designed for resource-constrained environments
</p>

## Features

| Category | Supported |
|----------|-----------|
| **Layers** | Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization |
| **Activations** | ReLU, Softmax, Sigmoid, Tanh, Linear |
| **Formats** | `.keras`, `.h5` |

## Installation

```bash
pip install pocket-inference
```

Prebuilt wheels are published for Linux (x86_64, aarch64) and macOS (Apple Silicon), and cover every Python version from 3.7 onwards via the stable ABI.

## Usage

```python
import numpy as np
from pocket_inference import Sequential

# Load a saved Keras model
model = Sequential.load("model.keras")

# Run inference on a single sample
input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
output = model.predict(input_data)
print(output)

# Batch inference
batch = np.random.randn(32, 4).astype(np.float32)
batch_output = model.predict(batch)

# Introspection
print(model.summary())
print(model.num_layers(), model.layer_names())
```

## Roadmap

- Additional layer types (LSTM, GRU, ...)
- More activation functions (LeakyReLU, ELU, GELU, ...)
- GPU acceleration via compute shaders
- Model optimization and pruning tools
- Benchmark suite and performance metrics

## Development

```bash
git clone https://github.com/James-Wirth/pocket-inference.git
cd pocket-inference
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release --features python
```

## License

Dual-licensed under MIT or Apache 2.0.
