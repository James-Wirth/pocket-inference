pub mod dense;
pub mod flatten;
pub mod dropout;
pub mod conv2d;
pub mod pooling;

use crate::{Result, Tensor};

pub trait Layer: std::fmt::Debug + Send + Sync {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn name(&self) -> &str;
    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>>;
}

pub use dense::Dense;
pub use flatten::Flatten;
pub use dropout::Dropout;
pub use conv2d::Conv2D;
pub use pooling::{MaxPooling2D, AveragePooling2D};
