use ndarray::{Array, ArrayD, IxDyn};

#[derive(Clone, Debug)]
pub struct Tensor {
    data: ArrayD<f32>,
}

impl Tensor {
    pub fn new(data: ArrayD<f32>) -> Self {
        Self { data }
    }

    pub fn from_vec(vec: Vec<f32>, shape: &[usize]) -> Self {
        let data = Array::from_shape_vec(IxDyn(shape), vec)
            .expect("Shape and vector length must match");
        Self { data }
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn data(&self) -> &ArrayD<f32> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut ArrayD<f32> {
        &mut self.data
    }

    pub fn into_data(self) -> ArrayD<f32> {
        self.data
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let data = ArrayD::zeros(IxDyn(shape));
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn reshape(&self, new_shape: &[usize]) -> crate::Result<Self> {
        let total_elements: usize = new_shape.iter().product();
        if total_elements != self.len() {
            return Err(crate::Error::ShapeMismatch {
                expected: vec![total_elements],
                actual: vec![self.len()],
            });
        }

        let reshaped = self.data.clone().into_shape_with_order(IxDyn(new_shape))
            .map_err(|e| crate::Error::Layer(format!("Reshape failed: {}", e)))?;
        Ok(Self { data: reshaped })
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data.iter().copied().collect()
    }
}

impl From<ArrayD<f32>> for Tensor {
    fn from(data: ArrayD<f32>) -> Self {
        Self::new(data)
    }
}

impl AsRef<ArrayD<f32>> for Tensor {
    fn as_ref(&self) -> &ArrayD<f32> {
        &self.data
    }
}
