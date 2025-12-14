//! # Pocket Inference
//!
//! A minimal-size Rust library for running Keras Sequential models.
//! This library focuses on inference only - no training.
//!
//! ## Example
//!
//! ```rust,ignore
//! use pocket_inference::Sequential;
//!
//! let model = Sequential::load("model.keras")?;
//! let output = model.predict(&input)?;
//! ```

pub mod activations;
mod conv2d_impl;
pub mod error;
pub mod layers;
pub mod model;
pub mod tensor;

#[cfg(feature = "python")]
pub mod python;

pub use error::{Error, Result};
pub use model::Sequential;
pub use tensor::Tensor;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn pocket_inference(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(m)?;
    Ok(())
}
