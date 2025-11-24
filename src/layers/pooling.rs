use crate::{Error, Result, Tensor};
use ndarray::Array4;

#[derive(Debug, Clone)]
pub struct MaxPooling2D {
    name: String,
    pool_size: (usize, usize),
    strides: (usize, usize),
    padding: Padding,
}

#[derive(Debug, Clone)]
pub struct AveragePooling2D {
    name: String,
    pool_size: (usize, usize),
    strides: (usize, usize),
    padding: Padding,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Padding {
    Valid,
    Same,
}

impl Padding {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "valid" => Ok(Padding::Valid),
            "same" => Ok(Padding::Same),
            _ => Err(Error::Layer(format!("Unknown padding type: {}", s))),
        }
    }
}

impl MaxPooling2D {
    pub fn new(
        name: String,
        pool_size: (usize, usize),
        strides: Option<(usize, usize)>,
        padding: Padding,
    ) -> Self {
        let strides = strides.unwrap_or(pool_size);
        Self {
            name,
            pool_size,
            strides,
            padding,
        }
    }
}

impl super::Layer for MaxPooling2D {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();

        let (batch_size, height, width, channels, is_batched) = if input_shape.len() == 4 {
            (input_shape[0], input_shape[1], input_shape[2], input_shape[3], true)
        } else if input_shape.len() == 3 {
            (1, input_shape[0], input_shape[1], input_shape[2], false)
        } else {
            return Err(Error::Layer(format!(
                "MaxPooling2D expects 3D or 4D input, got {:?}",
                input_shape
            )));
        };

        let input_4d = if is_batched {
            input.data().clone().into_shape_with_order((batch_size, height, width, channels))
                .map_err(|e| Error::Layer(format!("Reshape failed: {}", e)))?
        } else {
            input.data().clone().into_shape_with_order((1, height, width, channels))
                .map_err(|e| Error::Layer(format!("Reshape failed: {}", e)))?
        };

        let (out_height, out_width) = self.compute_output_size(height, width);

        let mut output = Array4::zeros((batch_size, out_height, out_width, channels));

        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let h_start = oh * self.strides.0;
                        let w_start = ow * self.strides.1;
                        let h_end = (h_start + self.pool_size.0).min(height);
                        let w_end = (w_start + self.pool_size.1).min(width);

                        let mut max_val = f32::NEG_INFINITY;
                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                let val = input_4d[[b, h, w, c]];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                        output[[b, oh, ow, c]] = max_val;
                    }
                }
            }
        }

        let output_shape = if is_batched {
            vec![batch_size, out_height, out_width, channels]
        } else {
            vec![out_height, out_width, channels]
        };

        let output_dyn = output.into_shape_with_order(ndarray::IxDyn(&output_shape))
            .map_err(|e| Error::Layer(format!("Reshape failed: {}", e)))?;

        Ok(Tensor::new(output_dyn))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        let (height, width, channels, is_batched) = if input_shape.len() == 4 {
            (input_shape[1], input_shape[2], input_shape[3], true)
        } else if input_shape.len() == 3 {
            (input_shape[0], input_shape[1], input_shape[2], false)
        } else {
            return Err(Error::Layer(format!(
                "MaxPooling2D expects 3D or 4D input, got {:?}",
                input_shape
            )));
        };

        let (out_height, out_width) = self.compute_output_size(height, width);

        if is_batched {
            Ok(vec![input_shape[0], out_height, out_width, channels])
        } else {
            Ok(vec![out_height, out_width, channels])
        }
    }
}

impl MaxPooling2D {
    fn compute_output_size(&self, height: usize, width: usize) -> (usize, usize) {
        let out_height = match self.padding {
            Padding::Valid => (height - self.pool_size.0) / self.strides.0 + 1,
            Padding::Same => (height + self.strides.0 - 1) / self.strides.0,
        };

        let out_width = match self.padding {
            Padding::Valid => (width - self.pool_size.1) / self.strides.1 + 1,
            Padding::Same => (width + self.strides.1 - 1) / self.strides.1,
        };

        (out_height, out_width)
    }
}

impl AveragePooling2D {
    pub fn new(
        name: String,
        pool_size: (usize, usize),
        strides: Option<(usize, usize)>,
        padding: Padding,
    ) -> Self {
        let strides = strides.unwrap_or(pool_size);
        Self {
            name,
            pool_size,
            strides,
            padding,
        }
    }

    fn compute_output_size(&self, height: usize, width: usize) -> (usize, usize) {
        let out_height = match self.padding {
            Padding::Valid => (height - self.pool_size.0) / self.strides.0 + 1,
            Padding::Same => (height + self.strides.0 - 1) / self.strides.0,
        };

        let out_width = match self.padding {
            Padding::Valid => (width - self.pool_size.1) / self.strides.1 + 1,
            Padding::Same => (width + self.strides.1 - 1) / self.strides.1,
        };

        (out_height, out_width)
    }
}

impl super::Layer for AveragePooling2D {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();

        let (batch_size, height, width, channels, is_batched) = if input_shape.len() == 4 {
            (input_shape[0], input_shape[1], input_shape[2], input_shape[3], true)
        } else if input_shape.len() == 3 {
            (1, input_shape[0], input_shape[1], input_shape[2], false)
        } else {
            return Err(Error::Layer(format!(
                "AveragePooling2D expects 3D or 4D input, got {:?}",
                input_shape
            )));
        };

        let input_4d = if is_batched {
            input.data().clone().into_shape_with_order((batch_size, height, width, channels))
                .map_err(|e| Error::Layer(format!("Reshape failed: {}", e)))?
        } else {
            input.data().clone().into_shape_with_order((1, height, width, channels))
                .map_err(|e| Error::Layer(format!("Reshape failed: {}", e)))?
        };

        let (out_height, out_width) = self.compute_output_size(height, width);

        let mut output = Array4::zeros((batch_size, out_height, out_width, channels));

        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let h_start = oh * self.strides.0;
                        let w_start = ow * self.strides.1;
                        let h_end = (h_start + self.pool_size.0).min(height);
                        let w_end = (w_start + self.pool_size.1).min(width);

                        let mut sum = 0.0;
                        let mut count = 0;
                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                sum += input_4d[[b, h, w, c]];
                                count += 1;
                            }
                        }
                        output[[b, oh, ow, c]] = sum / count as f32;
                    }
                }
            }
        }

        let output_shape = if is_batched {
            vec![batch_size, out_height, out_width, channels]
        } else {
            vec![out_height, out_width, channels]
        };

        let output_dyn = output.into_shape_with_order(ndarray::IxDyn(&output_shape))
            .map_err(|e| Error::Layer(format!("Reshape failed: {}", e)))?;

        Ok(Tensor::new(output_dyn))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        let (height, width, channels, is_batched) = if input_shape.len() == 4 {
            (input_shape[1], input_shape[2], input_shape[3], true)
        } else if input_shape.len() == 3 {
            (input_shape[0], input_shape[1], input_shape[2], false)
        } else {
            return Err(Error::Layer(format!(
                "AveragePooling2D expects 3D or 4D input, got {:?}",
                input_shape
            )));
        };

        let (out_height, out_width) = self.compute_output_size(height, width);

        if is_batched {
            Ok(vec![input_shape[0], out_height, out_width, channels])
        } else {
            Ok(vec![out_height, out_width, channels])
        }
    }
}
