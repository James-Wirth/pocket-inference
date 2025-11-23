use crate::{Error, Result, Tensor};
use ndarray::Zip;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    ELU,
    SELU,
    LeakyReLU { alpha: f32 },
}

impl Activation {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "linear" | "none" => Ok(Activation::Linear),
            "relu" => Ok(Activation::ReLU),
            "sigmoid" => Ok(Activation::Sigmoid),
            "tanh" => Ok(Activation::Tanh),
            "softmax" => Ok(Activation::Softmax),
            "elu" => Ok(Activation::ELU),
            "selu" => Ok(Activation::SELU),
            "leaky_relu" => Ok(Activation::LeakyReLU { alpha: 0.3 }),
            _ => Err(Error::UnsupportedActivation(s.to_string())),
        }
    }

    pub fn apply(&self, tensor: &mut Tensor) -> Result<()> {
        match self {
            Activation::Linear => Ok(()),
            Activation::ReLU => {
                Zip::from(tensor.data_mut()).for_each(|x| {
                    *x = x.max(0.0);
                });
                Ok(())
            }
            Activation::Sigmoid => {
                Zip::from(tensor.data_mut()).for_each(|x| {
                    *x = 1.0 / (1.0 + (-*x).exp());
                });
                Ok(())
            }
            Activation::Tanh => {
                Zip::from(tensor.data_mut()).for_each(|x| {
                    *x = x.tanh();
                });
                Ok(())
            }
            Activation::Softmax => {
                self.apply_softmax(tensor)
            }
            Activation::ELU => {
                Zip::from(tensor.data_mut()).for_each(|x| {
                    if *x < 0.0 {
                        *x = x.exp() - 1.0;
                    }
                });
                Ok(())
            }
            Activation::SELU => {
                let alpha = 1.6732632423543772848170429916717;
                let scale = 1.0507009873554804934193349852946;
                Zip::from(tensor.data_mut()).for_each(|x| {
                    *x = if *x > 0.0 {
                        scale * *x
                    } else {
                        scale * alpha * (x.exp() - 1.0)
                    };
                });
                Ok(())
            }
            Activation::LeakyReLU { alpha } => {
                let alpha = *alpha;
                Zip::from(tensor.data_mut()).for_each(|x| {
                    if *x < 0.0 {
                        *x = alpha * *x;
                    }
                });
                Ok(())
            }
        }
    }

    fn apply_softmax(&self, tensor: &mut Tensor) -> Result<()> {
        let data = tensor.data_mut();
        let shape = data.shape();

        if shape.is_empty() {
            return Ok(());
        }

        let last_axis_len = shape[shape.len() - 1];
        let batch_size = data.len() / last_axis_len;

        for batch in 0..batch_size {
            let start = batch * last_axis_len;
            let end = start + last_axis_len;

            let mut max_val = f32::NEG_INFINITY;
            for i in start..end {
                if data.as_slice().unwrap()[i] > max_val {
                    max_val = data.as_slice().unwrap()[i];
                }
            }

            let mut sum = 0.0;
            let slice = data.as_slice_mut().unwrap();
            for i in start..end {
                slice[i] = (slice[i] - max_val).exp();
                sum += slice[i];
            }

            for i in start..end {
                slice[i] /= sum;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_relu() {
        let mut tensor = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]);
        Activation::ReLU.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();
        assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let mut tensor = Tensor::from_vec(vec![0.0], &[1]);
        Activation::Sigmoid.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();
        assert_abs_diff_eq!(result[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        Activation::Softmax.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();

        let sum: f32 = result.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);

        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }
}
