use crate::{activations::Activation, Error, Result, Tensor};
use ndarray::{Array2, IxDyn};

#[derive(Debug, Clone)]
pub struct Dense {
    name: String,
    weights: Array2<f32>,
    bias: Option<Array2<f32>>,
    activation: Activation,
    units: usize,
}

impl Dense {
    pub fn new(
        name: String,
        weights: Array2<f32>,
        bias: Option<Array2<f32>>,
        activation: Activation,
    ) -> Result<Self> {
        let units = weights.ncols();

        if let Some(ref b) = bias {
            if b.len() != units {
                return Err(Error::Layer(format!(
                    "Bias size {} doesn't match units {}",
                    b.len(),
                    units
                )));
            }
        }

        Ok(Self {
            name,
            weights,
            bias,
            activation,
            units,
        })
    }

    pub fn units(&self) -> usize {
        self.units
    }
}

impl super::Layer for Dense {
    fn into_forward(&self, input: Tensor) -> Result<Tensor> {
        let input_shape = input.shape().to_vec();

        let (batch_size, features) = if input_shape.len() == 1 {
            (1, input_shape[0])
        } else if input_shape.len() == 2 {
            (input_shape[0], input_shape[1])
        } else {
            return Err(Error::Layer(format!(
                "Dense layer expects 1D or 2D input, got {:?}",
                input_shape
            )));
        };

        if features != self.weights.nrows() {
            return Err(Error::ShapeMismatch {
                expected: vec![self.weights.nrows()],
                actual: vec![features],
            });
        }

        let input_reshaped = if input_shape.len() == 1 {
            input.into_reshape(&[1, features])?
        } else {
            input.into_reshape(&[batch_size, features])?
        };

        let input_2d = input_reshaped
            .data()
            .to_owned()
            .into_shape_with_order((batch_size, features))
            .map_err(|e| Error::Layer(format!("Reshape failed: {}", e)))?;

        let mut output = input_2d.dot(&self.weights);

        if let Some(ref bias) = self.bias {
            for mut row in output.rows_mut() {
                row += &bias.row(0);
            }
        }

        let output_shape = if input_shape.len() == 1 {
            vec![self.units]
        } else {
            vec![batch_size, self.units]
        };

        let output_dyn = output
            .into_shape_with_order(IxDyn(&output_shape))
            .map_err(|e| Error::Layer(format!("Reshape failed: {}", e)))?;

        let mut tensor = Tensor::new(output_dyn);
        self.activation.apply(&mut tensor)?;

        Ok(tensor)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        if input_shape.len() == 1 {
            Ok(vec![self.units])
        } else if input_shape.len() == 2 {
            Ok(vec![input_shape[0], self.units])
        } else {
            Err(Error::Layer(format!(
                "Dense layer expects 1D or 2D input, got {:?}",
                input_shape
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Layer;
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dense_forward() {
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let bias = Some(array![[0.1, 0.2]]);

        let layer =
            Dense::new("test_dense".to_string(), weights, bias, Activation::Linear).unwrap();

        let input = Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap();
        let output = layer.forward(&input).unwrap();

        let result = output.to_vec();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 4.1).abs() < 1e-6);
        assert!((result[1] - 6.2).abs() < 1e-6);
    }

    #[test]
    fn test_dense_with_relu() {
        let weights = array![[1.0, -2.0], [-3.0, 4.0]];
        let bias = None;

        let layer = Dense::new("test_dense".to_string(), weights, bias, Activation::ReLU).unwrap();

        let input = Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap();
        let output = layer.forward(&input).unwrap();

        let result = output.to_vec();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 2.0);
    }
}
