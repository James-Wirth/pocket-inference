use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Model loading error: {0}")]
    ModelLoad(String),

    #[error("Layer error: {0}")]
    Layer(String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Unsupported layer type: {0}")]
    UnsupportedLayer(String),

    #[error("Unsupported activation: {0}")]
    UnsupportedActivation(String),

    #[error("ZIP error: {0}")]
    Zip(#[from] zip::result::ZipError),

    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
