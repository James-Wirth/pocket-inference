use crate::{Result, Tensor};

#[derive(Debug, Clone)]
pub struct Flatten {
    name: String,
}

impl Flatten {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

impl super::Layer for Flatten {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();

        if shape.is_empty() {
            return Ok(input.clone());
        }

        let new_shape = if shape.len() > 1 {
            let batch_size = shape[0];
            let flattened_size: usize = shape[1..].iter().product();
            vec![batch_size, flattened_size]
        } else {
            vec![shape.iter().product()]
        };

        input.reshape(&new_shape)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        if input_shape.is_empty() {
            return Ok(vec![1]);
        }

        if input_shape.len() > 1 {
            let batch_size = input_shape[0];
            let flattened_size: usize = input_shape[1..].iter().product();
            Ok(vec![batch_size, flattened_size])
        } else {
            Ok(vec![input_shape.iter().product()])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::Layer;

    #[test]
    fn test_flatten() {
        let layer = Flatten::new("test_flatten".to_string());

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 2]);

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3]);
    }
}
