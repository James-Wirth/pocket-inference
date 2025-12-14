use crate::{Error, Result, Tensor};
use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct BatchNormalization {
    name: String,
    gamma: Array1<f32>,
    beta: Array1<f32>,
    moving_mean: Array1<f32>,
    moving_variance: Array1<f32>,
    epsilon: f32,
}

impl BatchNormalization {
    pub fn new(
        name: String,
        gamma: Array1<f32>,
        beta: Array1<f32>,
        moving_mean: Array1<f32>,
        moving_variance: Array1<f32>,
        epsilon: f32,
    ) -> Result<Self> {
        let num_features = gamma.len();

        if beta.len() != num_features {
            return Err(Error::Layer(format!(
                "BatchNormalization: beta length {} doesn't match gamma length {}",
                beta.len(),
                num_features
            )));
        }

        if moving_mean.len() != num_features {
            return Err(Error::Layer(format!(
                "BatchNormalization: moving_mean length {} doesn't match gamma length {}",
                moving_mean.len(),
                num_features
            )));
        }

        if moving_variance.len() != num_features {
            return Err(Error::Layer(format!(
                "BatchNormalization: moving_variance length {} doesn't match gamma length {}",
                moving_variance.len(),
                num_features
            )));
        }

        Ok(Self {
            name,
            gamma,
            beta,
            moving_mean,
            moving_variance,
            epsilon,
        })
    }
}

impl super::Layer for BatchNormalization {
    fn into_forward(&self, input: Tensor) -> Result<Tensor> {
        let input_shape = input.shape().to_vec();

        let num_features = input_shape[input_shape.len() - 1];

        if num_features != self.gamma.len() {
            return Err(Error::Layer(format!(
                "BatchNormalization: input features {} doesn't match layer features {}",
                num_features,
                self.gamma.len()
            )));
        }

        let mut output = input.into_data();

        if let Some(data) = output.as_slice_mut() {
            let total_elements = data.len();
            let elements_per_feature = total_elements / num_features;

            for feature_idx in 0..num_features {
                let mean = self.moving_mean[feature_idx];
                let variance = self.moving_variance[feature_idx];
                let gamma = self.gamma[feature_idx];
                let beta = self.beta[feature_idx];
                let std_inv = 1.0 / (variance + self.epsilon).sqrt();

                for i in 0..elements_per_feature {
                    let idx = i * num_features + feature_idx;
                    data[idx] = gamma * ((data[idx] - mean) * std_inv) + beta;
                }
            }
        } else {
            for mut lane in output.lanes_mut(ndarray::Axis(input_shape.len() - 1)) {
                for (idx, val) in lane.iter_mut().enumerate() {
                    let mean = self.moving_mean[idx];
                    let variance = self.moving_variance[idx];
                    let gamma = self.gamma[idx];
                    let beta = self.beta[idx];
                    let std_inv = 1.0 / (variance + self.epsilon).sqrt();
                    *val = gamma * ((*val - mean) * std_inv) + beta;
                }
            }
        }

        Ok(Tensor::new(output))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        Ok(input_shape.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Layer;
    use ndarray::array;

    #[test]
    fn test_batch_norm_forward_simple() {
        let gamma = array![1.0, 1.0];
        let beta = array![0.0, 0.0];
        let moving_mean = array![0.5, 0.5];
        let moving_variance = array![0.25, 0.25];
        let epsilon = 0.001;

        let layer = BatchNormalization::new(
            "test_bn".to_string(),
            gamma,
            beta,
            moving_mean,
            moving_variance,
            epsilon,
        )
        .unwrap();

        let input = Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap();
        let output = layer.forward(&input).unwrap();
        let result = output.to_vec();

        assert_eq!(result.len(), 2);

        let expected = 0.5 / (0.251_f32).sqrt();
        assert!((result[0] - expected).abs() < 1e-5);
        assert!((result[1] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_batch_norm_with_scale_and_shift() {
        let gamma = array![2.0, 2.0];
        let beta = array![1.0, 1.0];
        let moving_mean = array![0.0, 0.0];
        let moving_variance = array![1.0, 1.0];
        let epsilon = 0.0;

        let layer = BatchNormalization::new(
            "test_bn".to_string(),
            gamma,
            beta,
            moving_mean,
            moving_variance,
            epsilon,
        )
        .unwrap();

        let input = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let output = layer.forward(&input).unwrap();
        let result = output.to_vec();

        assert!((result[0] - 3.0).abs() < 1e-5);
        assert!((result[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_norm_batched_input() {
        let gamma = array![1.0, 1.0];
        let beta = array![0.0, 0.0];
        let moving_mean = array![0.5, 0.5];
        let moving_variance = array![0.25, 0.25];
        let epsilon = 0.001;

        let layer = BatchNormalization::new(
            "test_bn".to_string(),
            gamma,
            beta,
            moving_mean,
            moving_variance,
            epsilon,
        )
        .unwrap();

        let input = Tensor::from_vec(vec![1.0, 1.0, 0.0, 0.0], &[2, 2]).unwrap();
        let output = layer.forward(&input).unwrap();
        let result = output.to_vec();

        assert_eq!(result.len(), 4);
        assert_eq!(output.shape(), &[2, 2]);

        let expected_1 = 0.5 / (0.251_f32).sqrt();
        let expected_0 = -0.5 / (0.251_f32).sqrt();

        assert!((result[0] - expected_1).abs() < 1e-5);
        assert!((result[1] - expected_1).abs() < 1e-5);
        assert!((result[2] - expected_0).abs() < 1e-5);
        assert!((result[3] - expected_0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_norm_validation() {
        let gamma = array![1.0, 1.0];
        let beta = array![0.0];
        let moving_mean = array![0.5, 0.5];
        let moving_variance = array![0.25, 0.25];

        let result = BatchNormalization::new(
            "test_bn".to_string(),
            gamma,
            beta,
            moving_mean,
            moving_variance,
            0.001,
        );

        assert!(result.is_err());
    }
}
