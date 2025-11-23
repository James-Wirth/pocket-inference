pub mod dense;

use crate::{Result, Tensor};

pub trait Layer: std::fmt::Debug + Send + Sync {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn name(&self) -> &str;
    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>>;
}

pub use dense::Dense;


