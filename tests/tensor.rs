use pocket_inference::Tensor;

#[test]
fn test_tensor_creation_from_vec() {
    let vec = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_vec(vec.clone(), &[4]).expect("Failed to create tensor");

    assert_eq!(tensor.shape(), &[4]);
    assert_eq!(tensor.to_vec(), vec);
}

#[test]
fn test_tensor_creation_2d() {
    let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(vec, &[2, 3]).expect("Failed to create tensor");

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.len(), 6);
}

#[test]
fn test_tensor_creation_3d() {
    let vec: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let tensor = Tensor::from_vec(vec, &[2, 3, 4]).expect("Failed to create tensor");

    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(tensor.len(), 24);
}

#[test]
fn test_tensor_shape_mismatch() {
    let vec = vec![1.0, 2.0, 3.0];
    let result = Tensor::from_vec(vec, &[4]);

    assert!(
        result.is_err(),
        "Should fail when shape doesn't match vector length"
    );
}

#[test]
fn test_tensor_zeros() {
    let tensor = Tensor::zeros(&[3, 2]);

    assert_eq!(tensor.shape(), &[3, 2]);
    assert_eq!(tensor.len(), 6);

    let vec = tensor.to_vec();
    assert!(vec.iter().all(|&x| x == 0.0), "All values should be zero");
}

#[test]
fn test_tensor_reshape() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("Failed to create tensor");

    let reshaped = tensor.reshape(&[3, 2]).expect("Reshape failed");

    assert_eq!(reshaped.shape(), &[3, 2]);
    assert_eq!(reshaped.len(), 6);
}

#[test]
fn test_tensor_reshape_1d_to_2d() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("Failed to create tensor");

    let reshaped = tensor.reshape(&[2, 2]).expect("Reshape failed");

    assert_eq!(reshaped.shape(), &[2, 2]);
}

#[test]
fn test_tensor_reshape_invalid() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("Failed to create tensor");

    let result = tensor.reshape(&[2, 3]);

    assert!(
        result.is_err(),
        "Should fail when total elements don't match"
    );
}

#[test]
fn test_tensor_len() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("Failed to create tensor");

    assert_eq!(tensor.len(), 3);
    assert!(!tensor.is_empty());
}

#[test]
fn test_tensor_empty() {
    let tensor = Tensor::from_vec(vec![], &[0]).expect("Failed to create tensor");

    assert_eq!(tensor.len(), 0);
    assert!(tensor.is_empty());
}

#[test]
fn test_tensor_to_vec() {
    let original = vec![1.5, 2.5, 3.5, 4.5];
    let tensor = Tensor::from_vec(original.clone(), &[4]).expect("Failed to create tensor");

    let retrieved = tensor.to_vec();

    assert_eq!(retrieved, original);
}

#[test]
fn test_tensor_clone() {
    let tensor1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("Failed to create tensor");

    let tensor2 = tensor1.clone();

    assert_eq!(tensor1.shape(), tensor2.shape());
    assert_eq!(tensor1.to_vec(), tensor2.to_vec());
}
