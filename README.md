# Pocket Inference

A lightweight Rust library for running inference on Keras Sequential models, designed for deployment on resource-constrained environments without the full TensorFlow/Keras stack.

## Features

| Category | Supported |
|----------|-----------|
| **Layers** | Dense, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, Dropout |
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

