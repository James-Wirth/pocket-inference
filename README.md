<p align="center">
  <img src="assets/logo.svg" alt="Pocket Inference" width="400">
</p>

---

<h3 align="center">A lightweight Rust library for running Keras models without the TensorFlow stack.</h3>

<p align="center">
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

## Installation

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
pocket-inference = "0.1.0"
```

### Python

Install via pip (after building):

```bash
pip install maturin
maturin develop --release --features python
```
## Usage

### Python

```python
import numpy as np
import pocket_inference as pi

# Load a saved Keras model
model = pi.Sequential.load("model.keras")

# Run inference
input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
output = model.predict(input_data)
print(f"Output: {output}")

# Batch inference
batch_input = np.random.randn(32, 4).astype(np.float32)
batch_output = model.predict(batch_input)
```

### Rust

```rust
use pocket_inference::{Sequential, Tensor};

fn main() -> pocket_inference::Result<()> {
    // Load a saved Keras model
    let model = Sequential::load("model.keras")?;

    // Run inference
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;
    let output = model.predict(&input)?;
    println!("Output: {:?}", output.to_vec());

    Ok(())
}
```

## Building from Source

```bash
# Clone the repository
git clone https://github.com/James-Wirth/pocket-inference.git
cd pocket-inference

# Build Rust library
cargo build --release

# Build Python bindings
maturin build --release --features python

# Install Python package locally
maturin develop --features python
```

## Roadmap

- Additional layer types (LSTM, GRU, ...)
- More activation functions (LeakyReLU, ELU, GELU, ...)
- GPU acceleration via compute shaders
- Model optimization and pruning tools
- Benchmark suite and performance metrics

