use approx::assert_abs_diff_eq;
use ndarray::array;
use pocket_inference::layers::{
    dense::Dense,
    flatten::Flatten,
    pooling::{AveragePooling2D, MaxPooling2D, Padding},
    Layer,
};
use pocket_inference::{activations::Activation, Tensor};

#[test]
fn test_dense_forward_single_input() {
    let weights = array![[1.0, 2.0], [3.0, 4.0]];
    let bias = Some(array![[0.1, 0.2]]);

    let layer = Dense::new("test_dense".to_string(), weights, bias, Activation::Linear).unwrap();

    let input = Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap();
    let output = layer.forward(&input).unwrap();

    let result = output.to_vec();
    assert_eq!(result.len(), 2);
    assert_abs_diff_eq!(result[0], 4.1, epsilon = 1e-6);
    assert_abs_diff_eq!(result[1], 6.2, epsilon = 1e-6);
}

#[test]
fn test_dense_batch_forward() {
    let weights = array![[1.0, 2.0], [3.0, 4.0]];
    let bias = Some(array![[0.5, 1.0]]);

    let layer = Dense::new("test_dense".to_string(), weights, bias, Activation::Linear).unwrap();

    let input = Tensor::from_vec(vec![1.0, 1.0, 2.0, 2.0], &[2, 2]).unwrap();

    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2]);
}

#[test]
fn test_dense_with_activation() {
    let weights = array![[1.0, -2.0], [-3.0, 4.0]];
    let bias = None;

    let layer = Dense::new("test_dense".to_string(), weights, bias, Activation::ReLU).unwrap();

    let input = Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap();
    let output = layer.forward(&input).unwrap();

    let result = output.to_vec();
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 2.0);
}

#[test]
fn test_dense_wrong_input_shape() {
    let weights = array![[1.0, 2.0], [3.0, 4.0]];
    let layer = Dense::new("test_dense".to_string(), weights, None, Activation::Linear).unwrap();

    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let result = layer.forward(&input);

    assert!(result.is_err());
}

#[test]
fn test_flatten_2d() {
    let layer = Flatten::new("test_flatten".to_string());

    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_flatten_3d_to_2d() {
    let layer = Flatten::new("test_flatten".to_string());

    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]).unwrap();

    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 3]);
}

#[test]
fn test_flatten_4d_conv_output() {
    let layer = Flatten::new("test_flatten".to_string());

    let input_vec: Vec<f32> = (0..36).map(|i| i as f32).collect();
    let input = Tensor::from_vec(input_vec, &[2, 3, 3, 2]).unwrap();

    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 18]);
}

#[test]
fn test_maxpool_2x2_valid() {
    let layer = MaxPooling2D::new("test_maxpool".to_string(), (2, 2), None, Padding::Valid);

    let input = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[4, 4, 1],
    )
    .unwrap();

    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2, 1]);

    let result = output.to_vec();
    assert_eq!(result[0], 6.0);
    assert_eq!(result[1], 8.0);
    assert_eq!(result[2], 14.0);
    assert_eq!(result[3], 16.0);
}

#[test]
fn test_maxpool_batch() {
    let layer = MaxPooling2D::new("test_maxpool".to_string(), (2, 2), None, Padding::Valid);

    let input_vec: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let input = Tensor::from_vec(input_vec, &[2, 4, 4, 1]).unwrap();

    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2, 2, 1]);
}

#[test]
fn test_maxpool_multiple_channels() {
    let layer = MaxPooling2D::new("test_maxpool".to_string(), (2, 2), None, Padding::Valid);

    let input_vec: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let input = Tensor::from_vec(input_vec, &[4, 4, 2]).unwrap();

    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2, 2]);
}

#[test]
fn test_avgpool_2x2_valid() {
    let layer = AveragePooling2D::new("test_avgpool".to_string(), (2, 2), None, Padding::Valid);

    let input = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[4, 4, 1],
    )
    .unwrap();

    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2, 1]);

    let result = output.to_vec();
    assert_abs_diff_eq!(result[0], 3.5, epsilon = 1e-6);
    assert_abs_diff_eq!(result[1], 5.5, epsilon = 1e-6);
    assert_abs_diff_eq!(result[2], 11.5, epsilon = 1e-6);
    assert_abs_diff_eq!(result[3], 13.5, epsilon = 1e-6);
}

#[test]
fn test_avgpool_batch() {
    let layer = AveragePooling2D::new("test_avgpool".to_string(), (2, 2), None, Padding::Valid);

    let input_vec: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let input = Tensor::from_vec(input_vec, &[2, 4, 4, 1]).unwrap();

    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2, 2, 1]);
}

#[test]
fn test_padding_from_str() {
    assert_eq!(Padding::from_str("valid").unwrap(), Padding::Valid);
    assert_eq!(Padding::from_str("same").unwrap(), Padding::Same);
    assert_eq!(Padding::from_str("VALID").unwrap(), Padding::Valid);
    assert_eq!(Padding::from_str("Same").unwrap(), Padding::Same);
}

#[test]
fn test_padding_from_str_invalid() {
    let result = Padding::from_str("invalid");
    assert!(result.is_err());
}

#[test]
fn test_dense_output_shape() {
    let weights = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let layer = Dense::new("test".to_string(), weights, None, Activation::Linear).unwrap();

    let shape = layer.output_shape(&[2]).unwrap();
    assert_eq!(shape, vec![3]);

    let shape = layer.output_shape(&[10, 2]).unwrap();
    assert_eq!(shape, vec![10, 3]);
}

#[test]
fn test_flatten_output_shape() {
    let layer = Flatten::new("test".to_string());

    let shape = layer.output_shape(&[2, 3, 4]).unwrap();
    assert_eq!(shape, vec![2, 12]);

    let shape = layer.output_shape(&[5, 5, 3]).unwrap();
    assert_eq!(shape, vec![5, 15]);
}

#[test]
fn test_maxpool_output_shape() {
    let layer = MaxPooling2D::new("test".to_string(), (2, 2), None, Padding::Valid);

    let shape = layer.output_shape(&[4, 4, 1]).unwrap();
    assert_eq!(shape, vec![2, 2, 1]);

    let shape = layer.output_shape(&[8, 4, 4, 2]).unwrap();
    assert_eq!(shape, vec![8, 2, 2, 2]);
}
