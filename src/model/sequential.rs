use crate::{layers::Layer, Error, Result, Tensor};

#[derive(Debug)]
pub struct Sequential {
    name: String,
    layers: Vec<Box<dyn Layer>>,
    input_shape: Option<Vec<usize>>,
}

impl Sequential {
    pub fn new(name: String) -> Self {
        Self {
            name,
            layers: Vec::new(),
            input_shape: None,
        }
    }

    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.input_shape = Some(shape);
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layer_names(&self) -> Vec<&str> {
        self.layers.iter().map(|l| l.name()).collect()
    }

    pub fn predict(&self, input: &Tensor) -> Result<Tensor> {
        if self.layers.is_empty() {
            return Err(Error::ModelLoad(
                "Cannot predict with empty model".to_string(),
            ));
        }

        let mut current = input.clone();

        for (idx, layer) in self.layers.iter().enumerate() {
            current = layer
                .forward(&current)
                .map_err(|e| Error::Layer(format!("Layer {} ({}): {}", idx, layer.name(), e)))?;
        }

        Ok(current)
    }

    pub fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        let mut current_shape = input_shape.to_vec();

        for layer in &self.layers {
            current_shape = layer.output_shape(&current_shape)?;
        }

        Ok(current_shape)
    }

    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Model: {}\n", self.name));
        s.push_str("_________________________________________________________________\n");
        s.push_str("Layer (type)                 Output Shape              \n");
        s.push_str("=================================================================\n");

        let mut current_shape = self.input_shape.clone().unwrap_or_default();

        for layer in &self.layers {
            if !current_shape.is_empty() {
                current_shape = match layer.output_shape(&current_shape) {
                    Ok(shape) => shape,
                    Err(_) => vec![],
                };
            }

            s.push_str(&format!("{:28} {:?}\n", layer.name(), current_shape));
        }

        s.push_str("=================================================================\n");
        s.push_str(&format!("Total layers: {}\n", self.layers.len()));

        s
    }

    pub fn load(path: &str) -> Result<Self> {
        super::loader::load_from_keras(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{activations::Activation, layers::Dense};
    use ndarray::array;

    #[test]
    fn test_sequential_predict() {
        let mut model = Sequential::new("test_model".to_string());

        let weights1 = array![[1.0, 0.5], [0.5, 1.0]];
        let bias1 = Some(array![[0.0, 0.0]]);
        let layer1 = Dense::new("dense1".to_string(), weights1, bias1, Activation::ReLU).unwrap();

        let weights2 = array![[1.0], [1.0]];
        let bias2 = Some(array![[0.0]]);
        let layer2 = Dense::new("dense2".to_string(), weights2, bias2, Activation::Linear).unwrap();

        model.add(Box::new(layer1));
        model.add(Box::new(layer2));

        let input = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let output = model.predict(&input).unwrap();

        assert_eq!(output.shape(), &[1]);
    }
}
