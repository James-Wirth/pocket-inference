pub mod batch_normalization;
pub mod conv2d;
pub mod dense;
pub mod dropout;
pub mod flatten;
pub mod pooling;

use crate::{Result, Tensor};

pub trait Layer: std::fmt::Debug + Send + Sync {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn name(&self) -> &str;
    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>>;
}

pub use batch_normalization::BatchNormalization;
pub use conv2d::Conv2D;
pub use dense::Dense;
pub use dropout::Dropout;
pub use flatten::Flatten;
pub use pooling::{AveragePooling2D, MaxPooling2D};
