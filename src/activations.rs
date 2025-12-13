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
            Activation::Softmax => self.apply_softmax(tensor),
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

        if let Some(slice) = data.as_slice_mut() {
            for batch in 0..batch_size {
                let start = batch * last_axis_len;
                let end = start + last_axis_len;

                let mut max_val = f32::NEG_INFINITY;
                for i in start..end {
                    if slice[i] > max_val {
                        max_val = slice[i];
                    }
                }

                let mut sum = 0.0;
                for i in start..end {
                    slice[i] = (slice[i] - max_val).exp();
                    sum += slice[i];
                }

                for i in start..end {
                    slice[i] /= sum;
                }
            }
        } else {
            for batch in 0..batch_size {
                let start = batch * last_axis_len;
                let end = start + last_axis_len;

                let mut max_val = f32::NEG_INFINITY;
                for i in start..end {
                    let val = data.as_slice().ok_or_else(|| {
                        Error::Layer("Softmax requires contiguous tensor".to_string())
                    })?[i];
                    if val > max_val {
                        max_val = val;
                    }
                }

                let mut sum = 0.0;
                for i in start..end {
                    let slice = data.as_slice_mut().ok_or_else(|| {
                        Error::Layer("Softmax requires contiguous tensor".to_string())
                    })?;
                    slice[i] = (slice[i] - max_val).exp();
                    sum += slice[i];
                }

                for i in start..end {
                    let slice = data.as_slice_mut().ok_or_else(|| {
                        Error::Layer("Softmax requires contiguous tensor".to_string())
                    })?;
                    slice[i] /= sum;
                }
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
    fn test_linear() {
        let original = vec![1.0, -2.0, 3.0, -4.0];
        let mut tensor = Tensor::from_vec(original.clone(), &[4]).unwrap();
        Activation::Linear.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();
        assert_eq!(
            result, original,
            "Linear activation should not modify values"
        );
    }

    #[test]
    fn test_relu() {
        let mut tensor = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();
        Activation::ReLU.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();
        assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_all_negative() {
        let mut tensor = Tensor::from_vec(vec![-5.0, -3.0, -1.0], &[3]).unwrap();
        Activation::ReLU.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_relu_all_positive() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        Activation::ReLU.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sigmoid() {
        let mut tensor = Tensor::from_vec(vec![0.0], &[1]).unwrap();
        Activation::Sigmoid.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();
        assert_abs_diff_eq!(result[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_sigmoid_large_positive() {
        let mut tensor = Tensor::from_vec(vec![10.0], &[1]).unwrap();
        Activation::Sigmoid.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();
        assert_abs_diff_eq!(result[0], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_sigmoid_large_negative() {
        let mut tensor = Tensor::from_vec(vec![-10.0], &[1]).unwrap();
        Activation::Sigmoid.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_tanh() {
        let mut tensor = Tensor::from_vec(vec![0.0], &[1]).unwrap();
        Activation::Tanh.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_range() {
        let mut tensor = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
        Activation::Tanh.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();

        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-6);
        assert!(result[0] < 0.0 && result[0] > -1.0);
        assert!(result[2] > 0.0 && result[2] < 1.0);
        assert_abs_diff_eq!(result[0], -result[2], epsilon = 1e-6);
    }

    #[test]
    fn test_softmax() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        Activation::Softmax.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();

        let sum: f32 = result.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);

        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_softmax_batch() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        Activation::Softmax.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();

        let sum1: f32 = result[0..3].iter().sum();
        let sum2: f32 = result[3..6].iter().sum();
        assert_abs_diff_eq!(sum1, 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sum2, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let mut tensor = Tensor::from_vec(vec![1000.0, 1001.0, 1002.0], &[3]).unwrap();
        Activation::Softmax.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();

        let sum: f32 = result.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_elu() {
        let mut tensor = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
        Activation::ELU.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();

        assert_abs_diff_eq!(result[0], (-1.0_f32).exp() - 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_selu() {
        let mut tensor = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap();
        Activation::SELU.apply(&mut tensor).unwrap();
        let result = tensor.to_vec();

        let alpha = 1.6732632423543772848170429916717;
        let scale = 1.0507009873554804934193349852946;

        assert_abs_diff_eq!(
            result[0],
            scale * alpha * ((-1.0_f32).exp() - 1.0),
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[2], scale * 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_leaky_relu() {
        let mut tensor = Tensor::from_vec(vec![-2.0, 0.0, 2.0], &[3]).unwrap();
        Activation::LeakyReLU { alpha: 0.3 }
            .apply(&mut tensor)
            .unwrap();
        let result = tensor.to_vec();

        assert_abs_diff_eq!(result[0], -2.0 * 0.3, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[2], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_leaky_relu_custom_alpha() {
        let mut tensor = Tensor::from_vec(vec![-1.0], &[1]).unwrap();
        Activation::LeakyReLU { alpha: 0.01 }
            .apply(&mut tensor)
            .unwrap();
        let result = tensor.to_vec();

        assert_abs_diff_eq!(result[0], -0.01, epsilon = 1e-6);
    }

    #[test]
    fn test_activation_from_str() {
        assert!(matches!(
            Activation::from_str("linear").unwrap(),
            Activation::Linear
        ));
        assert!(matches!(
            Activation::from_str("relu").unwrap(),
            Activation::ReLU
        ));
        assert!(matches!(
            Activation::from_str("sigmoid").unwrap(),
            Activation::Sigmoid
        ));
        assert!(matches!(
            Activation::from_str("tanh").unwrap(),
            Activation::Tanh
        ));
        assert!(matches!(
            Activation::from_str("softmax").unwrap(),
            Activation::Softmax
        ));
        assert!(matches!(
            Activation::from_str("elu").unwrap(),
            Activation::ELU
        ));
        assert!(matches!(
            Activation::from_str("selu").unwrap(),
            Activation::SELU
        ));
        assert!(matches!(
            Activation::from_str("leaky_relu").unwrap(),
            Activation::LeakyReLU { alpha: 0.3 }
        ));
    }

    #[test]
    fn test_activation_from_str_case_insensitive() {
        assert!(matches!(
            Activation::from_str("ReLU").unwrap(),
            Activation::ReLU
        ));
        assert!(matches!(
            Activation::from_str("SIGMOID").unwrap(),
            Activation::Sigmoid
        ));
    }

    #[test]
    fn test_activation_from_str_none() {
        assert!(matches!(
            Activation::from_str("none").unwrap(),
            Activation::Linear
        ));
    }

    #[test]
    fn test_activation_from_str_invalid() {
        let result = Activation::from_str("invalid_activation");
        assert!(result.is_err());
    }
}
