use crate::{Result, Tensor};

#[derive(Debug, Clone)]
pub struct Dropout {
    name: String,
    rate: f32,
}

impl Dropout {
    pub fn new(name: String, rate: f32) -> Self {
        Self { name, rate }
    }

    pub fn rate(&self) -> f32 {
        self.rate
    }
}

impl super::Layer for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.clone())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        Ok(input_shape.to_vec())
    }
}
