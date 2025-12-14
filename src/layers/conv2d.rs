use crate::{activations::Activation, conv2d_impl, Error, Result, Tensor};
use ndarray::Array4;

use super::pooling::Padding;

#[derive(Debug, Clone)]
pub struct Conv2D {
    name: String,
    filters: usize,
    kernel_size: (usize, usize),
    strides: (usize, usize),
    padding: Padding,
    weights: Array4<f32>,
    bias: Option<Vec<f32>>,
    activation: Activation,
}

impl Conv2D {
    pub fn new(
        name: String,
        filters: usize,
        kernel_size: (usize, usize),
        strides: (usize, usize),
        padding: Padding,
        weights: Array4<f32>,
        bias: Option<Vec<f32>>,
        activation: Activation,
    ) -> Result<Self> {
        if let Some(ref b) = bias {
            if b.len() != filters {
                return Err(Error::Layer(format!(
                    "Bias size {} doesn't match filters {}",
                    b.len(),
                    filters
                )));
            }
        }

        Ok(Self {
            name,
            filters,
            kernel_size,
            strides,
            padding,
            weights,
            bias,
            activation,
        })
    }

    fn compute_output_size(&self, height: usize, width: usize) -> (usize, usize) {
        let out_height = match self.padding {
            Padding::Valid => (height - self.kernel_size.0) / self.strides.0 + 1,
            Padding::Same => (height + self.strides.0 - 1) / self.strides.0,
        };

        let out_width = match self.padding {
            Padding::Valid => (width - self.kernel_size.1) / self.strides.1 + 1,
            Padding::Same => (width + self.strides.1 - 1) / self.strides.1,
        };

        (out_height, out_width)
    }

    fn compute_padding(&self, height: usize, width: usize) -> ((usize, usize), (usize, usize)) {
        match self.padding {
            Padding::Valid => ((0, 0), (0, 0)),
            Padding::Same => {
                let (out_h, out_w) = self.compute_output_size(height, width);

                let pad_h =
                    ((out_h - 1) * self.strides.0 + self.kernel_size.0).saturating_sub(height);
                let pad_w =
                    ((out_w - 1) * self.strides.1 + self.kernel_size.1).saturating_sub(width);

                let pad_top = pad_h / 2;
                let pad_bottom = pad_h - pad_top;
                let pad_left = pad_w / 2;
                let pad_right = pad_w - pad_left;

                ((pad_top, pad_bottom), (pad_left, pad_right))
            }
        }
    }
}

impl super::Layer for Conv2D {
    fn into_forward(&self, input: Tensor) -> Result<Tensor> {
        let input_shape = input.shape().to_vec();

        let (batch_size, height, width, in_channels, is_batched) = if input_shape.len() == 4 {
            (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
                true,
            )
        } else if input_shape.len() == 3 {
            (1, input_shape[0], input_shape[1], input_shape[2], false)
        } else {
            return Err(Error::Layer(format!(
                "Conv2D expects 3D or 4D input, got {:?}",
                input_shape
            )));
        };

        if in_channels != self.weights.shape()[2] {
            return Err(Error::ShapeMismatch {
                expected: vec![self.weights.shape()[2]],
                actual: vec![in_channels],
            });
        }

        let input_reshaped = if is_batched {
            input.into_reshape(&[batch_size, height, width, in_channels])?
        } else {
            input.into_reshape(&[1, height, width, in_channels])?
        };

        let input_4d = input_reshaped
            .data()
            .to_owned()
            .into_shape_with_order((batch_size, height, width, in_channels))
            .map_err(|e| Error::Layer(format!("Reshape failed: {}", e)))?;

        let (out_height, out_width) = self.compute_output_size(height, width);
        let ((pad_top, _pad_bottom), (pad_left, _pad_right)) = self.compute_padding(height, width);

        let col_matrix = conv2d_impl::im2col(
            &input_4d,
            self.kernel_size.0,
            self.kernel_size.1,
            self.strides.0,
            self.strides.1,
            pad_top,
            pad_left,
            out_height,
            out_width,
        );

        let kernel_flat_size = self.kernel_size.0 * self.kernel_size.1 * in_channels;
        let weights_2d = self
            .weights
            .clone()
            .into_shape_with_order((kernel_flat_size, self.filters))
            .map_err(|e| Error::Layer(format!("Weight reshape failed: {}", e)))?;

        let mut output_2d = col_matrix.dot(&weights_2d);

        if let Some(ref bias) = self.bias {
            for mut row in output_2d.rows_mut() {
                for (i, val) in row.iter_mut().enumerate() {
                    *val += bias[i];
                }
            }
        }

        // Reshape output: (B*OH*OW, F) -> (B, OH, OW, F)
        let output = output_2d
            .into_shape_with_order((batch_size, out_height, out_width, self.filters))
            .map_err(|e| Error::Layer(format!("Output reshape failed: {}", e)))?;

        let output_shape = if is_batched {
            vec![batch_size, out_height, out_width, self.filters]
        } else {
            vec![out_height, out_width, self.filters]
        };

        let output_dyn = output
            .into_shape_with_order(ndarray::IxDyn(&output_shape))
            .map_err(|e| Error::Layer(format!("Reshape failed: {}", e)))?;

        let mut tensor = Tensor::new(output_dyn);
        self.activation.apply(&mut tensor)?;

        Ok(tensor)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        let (height, width, _channels, is_batched) = if input_shape.len() == 4 {
            (input_shape[1], input_shape[2], input_shape[3], true)
        } else if input_shape.len() == 3 {
            (input_shape[0], input_shape[1], input_shape[2], false)
        } else {
            return Err(Error::Layer(format!(
                "Conv2D expects 3D or 4D input, got {:?}",
                input_shape
            )));
        };

        let (out_height, out_width) = self.compute_output_size(height, width);

        if is_batched {
            Ok(vec![input_shape[0], out_height, out_width, self.filters])
        } else {
            Ok(vec![out_height, out_width, self.filters])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::Layer;
    use super::*;

    #[test]
    fn test_conv2d_basic() {
        let weights = Array4::from_shape_fn((1, 1, 1, 1), |_| 1.0);
        let bias = Some(vec![0.0]);

        let layer = Conv2D::new(
            "test_conv".to_string(),
            1,
            (1, 1),
            (1, 1),
            Padding::Valid,
            weights,
            bias,
            Activation::Linear,
        )
        .unwrap();

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2, 1]).unwrap();
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 2, 1]);
    }
}
