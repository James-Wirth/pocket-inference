use crate::{
    activations::Activation,
    layers::{pooling::Padding, *},
    Error, Result,
};
use hdf5::File as H5File;
use ndarray::Array;
use serde_json::Value;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use zip::ZipArchive;

use super::Sequential;

pub fn load_from_keras(path: &str) -> Result<Sequential> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(BufReader::new(file))
        .map_err(|e| Error::ModelLoad(format!("Failed to open .keras file as ZIP: {}", e)))?;

    let config = read_config(&mut archive)?;
    let model_name = config["config"]["name"]
        .as_str()
        .unwrap_or("model")
        .to_string();

    let mut model = Sequential::new(model_name);

    let layer_configs = config["config"]["layers"]
        .as_array()
        .ok_or_else(|| Error::ModelLoad("No layers found in config".to_string()))?;

    let _weights_temp = extract_weights_h5(&mut archive)?;

    let h5_file = H5File::open(_weights_temp.path())?;

    let mut dense_layer_idx = 0;
    let mut conv_layer_idx = 0;
    let mut bn_layer_idx = 0;

    for layer_config in layer_configs {
        let class_name = layer_config["class_name"]
            .as_str()
            .ok_or_else(|| Error::ModelLoad("Layer missing class_name".to_string()))?;

        let layer_name = layer_config["config"]["name"]
            .as_str()
            .ok_or_else(|| Error::ModelLoad("Layer missing name".to_string()))?
            .to_string();

        let layer: Box<dyn Layer> = match class_name {
            "Dense" => {
                let h5_layer_name = if dense_layer_idx == 0 {
                    "dense".to_string()
                } else {
                    format!("dense_{}", dense_layer_idx)
                };
                dense_layer_idx += 1;
                load_dense_layer(&h5_file, &h5_layer_name, &layer_name, layer_config)?
            }
            "Conv2D" => {
                let h5_layer_name = if conv_layer_idx == 0 {
                    "conv2d".to_string()
                } else {
                    format!("conv2d_{}", conv_layer_idx)
                };
                conv_layer_idx += 1;
                load_conv2d_layer(&h5_file, &h5_layer_name, &layer_name, layer_config)?
            }
            "MaxPooling2D" => load_maxpooling2d_layer(layer_config)?,
            "AveragePooling2D" => load_averagepooling2d_layer(layer_config)?,
            "Flatten" => Box::new(Flatten::new(layer_name)),
            "Dropout" => {
                let rate = layer_config["config"]["rate"]
                    .as_f64()
                    .unwrap_or(0.5) as f32;
                Box::new(Dropout::new(layer_name, rate))
            }
            "BatchNormalization" => {
                let h5_layer_name = if bn_layer_idx == 0 {
                    "batch_normalization".to_string()
                } else {
                    format!("batch_normalization_{}", bn_layer_idx)
                };
                bn_layer_idx += 1;
                load_batch_normalization_layer(&h5_file, &h5_layer_name, &layer_name, layer_config)?
            }
            "InputLayer" => {
                if let Some(batch_shape) = layer_config["config"]["batch_shape"].as_array() {
                    let input_shape: Vec<usize> = batch_shape
                        .iter()
                        .skip(1)
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect();
                    if !input_shape.is_empty() {
                        model.set_input_shape(input_shape);
                    }
                }
                continue;
            }
            _ => return Err(Error::UnsupportedLayer(class_name.to_string())),
        };

        model.add(layer);
    }

    Ok(model)
}

fn read_config(archive: &mut ZipArchive<BufReader<File>>) -> Result<Value> {
    let mut config_file = archive
        .by_name("config.json")
        .map_err(|e| Error::ModelLoad(format!("config.json not found in .keras file: {}", e)))?;

    let mut config_str = String::new();
    config_file
        .read_to_string(&mut config_str)
        .map_err(|e| Error::ModelLoad(format!("Failed to read config.json: {}", e)))?;

    let config: Value = serde_json::from_str(&config_str)?;
    Ok(config)
}

fn extract_weights_h5(archive: &mut ZipArchive<BufReader<File>>) -> Result<tempfile::NamedTempFile> {
    let mut weights_zip = archive
        .by_name("model.weights.h5")
        .map_err(|e| Error::ModelLoad(format!("model.weights.h5 not found in .keras file: {}", e)))?;

    let mut temp_file = tempfile::NamedTempFile::new()?;
    let mut buffer = Vec::new();
    weights_zip.read_to_end(&mut buffer)?;
    temp_file.write_all(&buffer)?;
    temp_file.flush()?;

    Ok(temp_file)
}

fn load_dense_layer(
    h5_file: &H5File,
    h5_layer_name: &str,
    layer_name: &str,
    config: &Value,
) -> Result<Box<dyn Layer>> {
    let _units = config["config"]["units"]
        .as_u64()
        .ok_or_else(|| Error::ModelLoad("Dense layer missing units".to_string()))?
        as usize;

    let activation_str = config["config"]["activation"]
        .as_str()
        .unwrap_or("linear");
    let activation = Activation::from_str(activation_str)?;

    let use_bias = config["config"]["use_bias"].as_bool().unwrap_or(true);

    let layer_group = h5_file
        .group(&format!("layers/{}", h5_layer_name))
        .or_else(|_| {
            h5_file.group(h5_layer_name)
        })
        .map_err(|_| Error::ModelLoad(format!("Layer weights not found: {} (tried layers/{})", layer_name, h5_layer_name)))?;

    let vars_group = layer_group
        .group("vars")
        .map_err(|_| Error::ModelLoad(format!("vars group not found for layer: {}", layer_name)))?;

    let kernel_dataset = vars_group
        .dataset("0")
        .map_err(|_| Error::ModelLoad(format!("kernel not found for layer: {}", layer_name)))?;

    let kernel_shape = kernel_dataset.shape();
    let kernel_data: Vec<f32> = kernel_dataset
        .read_raw()
        .map_err(|e| Error::ModelLoad(format!("Failed to read kernel: {}", e)))?;

    let weights = Array::from_shape_vec((kernel_shape[0], kernel_shape[1]), kernel_data)
        .map_err(|e| Error::ModelLoad(format!("Failed to create weights array: {}", e)))?;

    let bias = if use_bias {
        let bias_dataset = vars_group
            .dataset("1")
            .map_err(|_| Error::ModelLoad(format!("bias not found for layer: {}", layer_name)))?;

        let bias_data: Vec<f32> = bias_dataset
            .read_raw()
            .map_err(|e| Error::ModelLoad(format!("Failed to read bias: {}", e)))?;

        Some(
            Array::from_shape_vec((1, bias_data.len()), bias_data)
                .map_err(|e| Error::ModelLoad(format!("Failed to create bias array: {}", e)))?,
        )
    } else {
        None
    };

    Ok(Box::new(Dense::new(layer_name.to_string(), weights, bias, activation)?))
}

fn load_conv2d_layer(
    h5_file: &H5File,
    h5_layer_name: &str,
    layer_name: &str,
    config: &Value,
) -> Result<Box<dyn Layer>> {
    let filters = config["config"]["filters"]
        .as_u64()
        .ok_or_else(|| Error::ModelLoad("Conv2D layer missing filters".to_string()))?
        as usize;

    let kernel_size_arr = config["config"]["kernel_size"]
        .as_array()
        .ok_or_else(|| Error::ModelLoad("Conv2D layer missing kernel_size".to_string()))?;
    let kernel_size = (
        kernel_size_arr
            .get(0)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::ModelLoad("Invalid kernel_size[0] in config".to_string()))? as usize,
        kernel_size_arr
            .get(1)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::ModelLoad("Invalid kernel_size[1] in config".to_string()))? as usize,
    );

    let strides_arr = config["config"]["strides"].as_array();
    let strides = if let Some(s) = strides_arr {
        (
            s.get(0)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| Error::ModelLoad("Invalid strides[0] in config".to_string()))? as usize,
            s.get(1)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| Error::ModelLoad("Invalid strides[1] in config".to_string()))? as usize,
        )
    } else {
        (1, 1)
    };

    let padding_str = config["config"]["padding"].as_str().unwrap_or("valid");
    let padding = Padding::from_str(padding_str)?;

    let activation_str = config["config"]["activation"]
        .as_str()
        .unwrap_or("linear");
    let activation = Activation::from_str(activation_str)?;

    let use_bias = config["config"]["use_bias"].as_bool().unwrap_or(true);

    let layer_group = h5_file
        .group(&format!("layers/{}", h5_layer_name))
        .or_else(|_| {
            h5_file.group(h5_layer_name)
        })
        .map_err(|_| Error::ModelLoad(format!("Layer weights not found: {} (tried layers/{})", layer_name, h5_layer_name)))?;

    let vars_group = layer_group
        .group("vars")
        .map_err(|_| Error::ModelLoad(format!("vars group not found for layer: {}", layer_name)))?;

    let kernel_dataset = vars_group
        .dataset("0")
        .map_err(|_| Error::ModelLoad(format!("kernel not found for layer: {}", layer_name)))?;

    let kernel_shape = kernel_dataset.shape();
    let kernel_data: Vec<f32> = kernel_dataset
        .read_raw()
        .map_err(|e| Error::ModelLoad(format!("Failed to read kernel: {}", e)))?;

    let weights = Array::from_shape_vec(
        (
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
            kernel_shape[3],
        ),
        kernel_data,
    )
    .map_err(|e| Error::ModelLoad(format!("Failed to create weights array: {}", e)))?;

    let bias = if use_bias {
        let bias_dataset = vars_group
            .dataset("1")
            .map_err(|_| Error::ModelLoad(format!("bias not found for layer: {}", layer_name)))?;

        let bias_data: Vec<f32> = bias_dataset
            .read_raw()
            .map_err(|e| Error::ModelLoad(format!("Failed to read bias: {}", e)))?;

        Some(bias_data)
    } else {
        None
    };

    Ok(Box::new(Conv2D::new(
        layer_name.to_string(),
        filters,
        kernel_size,
        strides,
        padding,
        weights,
        bias,
        activation,
    )?))
}

fn load_maxpooling2d_layer(config: &Value) -> Result<Box<dyn Layer>> {
    let layer_name = config["config"]["name"]
        .as_str()
        .ok_or_else(|| Error::ModelLoad("MaxPooling2D layer missing name".to_string()))?
        .to_string();

    let pool_size_arr = config["config"]["pool_size"]
        .as_array()
        .ok_or_else(|| Error::ModelLoad("MaxPooling2D layer missing pool_size".to_string()))?;
    let pool_size = (
        pool_size_arr
            .get(0)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::ModelLoad("Invalid pool_size[0] in config".to_string()))? as usize,
        pool_size_arr
            .get(1)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::ModelLoad("Invalid pool_size[1] in config".to_string()))? as usize,
    );

    let strides = if let Some(strides_arr) = config["config"]["strides"].as_array() {
        Some((
            strides_arr
                .get(0)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| Error::ModelLoad("Invalid strides[0] in config".to_string()))? as usize,
            strides_arr
                .get(1)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| Error::ModelLoad("Invalid strides[1] in config".to_string()))? as usize,
        ))
    } else {
        None
    };

    let padding_str = config["config"]["padding"].as_str().unwrap_or("valid");
    let padding = Padding::from_str(padding_str)?;

    Ok(Box::new(MaxPooling2D::new(
        layer_name, pool_size, strides, padding,
    )))
}

fn load_averagepooling2d_layer(config: &Value) -> Result<Box<dyn Layer>> {
    let layer_name = config["config"]["name"]
        .as_str()
        .ok_or_else(|| Error::ModelLoad("AveragePooling2D layer missing name".to_string()))?
        .to_string();

    let pool_size_arr = config["config"]["pool_size"]
        .as_array()
        .ok_or_else(|| Error::ModelLoad("AveragePooling2D layer missing pool_size".to_string()))?;
    let pool_size = (
        pool_size_arr
            .get(0)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::ModelLoad("Invalid pool_size[0] in config".to_string()))? as usize,
        pool_size_arr
            .get(1)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| Error::ModelLoad("Invalid pool_size[1] in config".to_string()))? as usize,
    );

    let strides = if let Some(strides_arr) = config["config"]["strides"].as_array() {
        Some((
            strides_arr
                .get(0)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| Error::ModelLoad("Invalid strides[0] in config".to_string()))? as usize,
            strides_arr
                .get(1)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| Error::ModelLoad("Invalid strides[1] in config".to_string()))? as usize,
        ))
    } else {
        None
    };

    let padding_str = config["config"]["padding"].as_str().unwrap_or("valid");
    let padding = Padding::from_str(padding_str)?;

    Ok(Box::new(AveragePooling2D::new(
        layer_name, pool_size, strides, padding,
    )))
}

fn load_batch_normalization_layer(
    h5_file: &H5File,
    h5_layer_name: &str,
    layer_name: &str,
    config: &Value,
) -> Result<Box<dyn Layer>> {
    let epsilon = config["config"]["epsilon"]
        .as_f64()
        .unwrap_or(0.001) as f32;

    let center = config["config"]["center"].as_bool().unwrap_or(true);
    let scale = config["config"]["scale"].as_bool().unwrap_or(true);

    let layer_group = h5_file
        .group(&format!("layers/{}", h5_layer_name))
        .or_else(|_| h5_file.group(h5_layer_name))
        .map_err(|_| {
            Error::ModelLoad(format!(
                "Layer weights not found: {} (tried layers/{})",
                layer_name, h5_layer_name
            ))
        })?;

    let vars_group = layer_group.group("vars").map_err(|_| {
        Error::ModelLoad(format!("vars group not found for layer: {}", layer_name))
    })?;

    let mut dataset_idx = 0;

    let gamma = if scale {
        let gamma_dataset = vars_group.dataset(&dataset_idx.to_string()).map_err(|_| {
            Error::ModelLoad(format!("gamma not found for layer: {}", layer_name))
        })?;
        let gamma_data: Vec<f32> = gamma_dataset
            .read_raw()
            .map_err(|e| Error::ModelLoad(format!("Failed to read gamma: {}", e)))?;
        dataset_idx += 1;
        Array::from_shape_vec(gamma_data.len(), gamma_data)
            .map_err(|e| Error::ModelLoad(format!("Failed to create gamma array: {}", e)))?
    } else {
        let moving_mean_dataset = vars_group.dataset("0").map_err(|_| {
            Error::ModelLoad(format!("moving_mean not found for layer: {}", layer_name))
        })?;
        let shape = moving_mean_dataset.shape();
        Array::from_elem(shape[0], 1.0)
    };

    let beta = if center {
        let beta_dataset = vars_group.dataset(&dataset_idx.to_string()).map_err(|_| {
            Error::ModelLoad(format!("beta not found for layer: {}", layer_name))
        })?;
        let beta_data: Vec<f32> = beta_dataset
            .read_raw()
            .map_err(|e| Error::ModelLoad(format!("Failed to read beta: {}", e)))?;
        dataset_idx += 1;
        Array::from_shape_vec(beta_data.len(), beta_data)
            .map_err(|e| Error::ModelLoad(format!("Failed to create beta array: {}", e)))?
    } else {
        Array::from_elem(gamma.len(), 0.0)
    };

    let moving_mean_dataset = vars_group.dataset(&dataset_idx.to_string()).map_err(|_| {
        Error::ModelLoad(format!("moving_mean not found for layer: {}", layer_name))
    })?;
    let moving_mean_data: Vec<f32> = moving_mean_dataset
        .read_raw()
        .map_err(|e| Error::ModelLoad(format!("Failed to read moving_mean: {}", e)))?;
    let moving_mean = Array::from_shape_vec(moving_mean_data.len(), moving_mean_data)
        .map_err(|e| Error::ModelLoad(format!("Failed to create moving_mean array: {}", e)))?;
    dataset_idx += 1;

    let moving_variance_dataset = vars_group.dataset(&dataset_idx.to_string()).map_err(|_| {
        Error::ModelLoad(format!(
            "moving_variance not found for layer: {}",
            layer_name
        ))
    })?;
    let moving_variance_data: Vec<f32> = moving_variance_dataset
        .read_raw()
        .map_err(|e| Error::ModelLoad(format!("Failed to read moving_variance: {}", e)))?;
    let moving_variance =
        Array::from_shape_vec(moving_variance_data.len(), moving_variance_data).map_err(|e| {
            Error::ModelLoad(format!("Failed to create moving_variance array: {}", e))
        })?;

    Ok(Box::new(BatchNormalization::new(
        layer_name.to_string(),
        gamma,
        beta,
        moving_mean,
        moving_variance,
        epsilon,
    )?))
}
