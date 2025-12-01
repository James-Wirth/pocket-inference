use pyo3::prelude::*;
use numpy::{PyArray, PyArrayDyn, PyArrayMethods};
use ndarray::ArrayD;

use crate::{Sequential as RustSequential, Tensor};

#[pyclass(name = "Sequential")]
pub struct PySequential {
    inner: RustSequential,
}

#[pymethods]
impl PySequential {
    #[staticmethod]
    #[pyo3(signature = (_path))]
    fn load(_path: &str) -> PyResult<Self> {
        let model = RustSequential::load(_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load model: {}", e)))?;

        Ok(Self { inner: model })
    }

    #[pyo3(signature = (input))]
    fn predict<'py>(&self, py: Python<'py>, input: &Bound<'py, PyArrayDyn<f32>>) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
        let input_array: ArrayD<f32> = input.readonly().as_array().to_owned();
        let input_tensor = Tensor::new(input_array);

        let output = self.inner.predict(&input_tensor)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Prediction failed: {}", e)))?;

        let output_array = output.into_data();
        Ok(PyArray::from_owned_array(py, output_array))
    }

    fn summary(&self) -> PyResult<String> {
        Ok(self.inner.summary())
    }

    #[pyo3(signature = ())]
    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    #[pyo3(signature = ())]
    fn layer_names(&self) -> Vec<String> {
        self.inner.layer_names().iter().map(|s| s.to_string()).collect()
    }

    fn __repr__(&self) -> String {
        format!("<Sequential model with {} layers>", self.inner.num_layers())
    }

    fn __str__(&self) -> String {
        self.inner.summary()
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySequential>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
