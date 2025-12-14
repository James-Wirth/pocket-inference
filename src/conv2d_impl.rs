use ndarray::{Array2, Array4};

/// Transform 4D input tensor into 2D matrix for GEMM convolution.
/// This is the im2col (image-to-column) transformation.

pub fn im2col(
    input: &Array4<f32>,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    out_h: usize,
    out_w: usize,
) -> Array2<f32> {
    let (batch, height, width, channels) = input.dim();
    let col_h = batch * out_h * out_w;
    let col_w = kernel_h * kernel_w * channels;

    let mut col_matrix = Array2::zeros((col_h, col_w));

    for b in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let row_idx = b * (out_h * out_w) + oh * out_w + ow;
                let mut col_idx = 0;

                for kh in 0..kernel_h {
                    for kw in 0..kernel_w {

                        let ih = (oh * stride_h + kh).wrapping_sub(pad_top);
                        let iw = (ow * stride_w + kw).wrapping_sub(pad_left);

                        if ih < height && iw < width {
                            for c in 0..channels {
                                col_matrix[[row_idx, col_idx + c]] = input[[b, ih, iw, c]];
                            }
                        }

                        col_idx += channels;
                    }
                }
            }
        }
    }

    col_matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_im2col_basic() {
        let input = Array4::from_shape_vec((1, 2, 2, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let col = im2col(&input, 2, 2, 1, 1, 0, 0, 1, 1);

        assert_eq!(col.shape(), &[1, 4]);
        assert_eq!(col.row(0).to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_im2col_with_stride() {

        let input = Array4::from_shape_vec(
            (1, 3, 3, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();

        let col = im2col(&input, 2, 2, 2, 2, 0, 0, 1, 1);

        assert_eq!(col.shape(), &[1, 4]);
        assert_eq!(col.row(0).to_vec(), vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn test_im2col_with_padding() {

        let input = Array4::from_shape_vec((1, 2, 2, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let col = im2col(&input, 3, 3, 1, 1, 1, 1, 2, 2);

        assert_eq!(col.shape(), &[4, 9]);

        let first_patch = col.row(0).to_vec();
        assert_eq!(first_patch[4], 1.0); 
    }

    #[test]
    fn test_im2col_multiple_channels() {
        let input = Array4::from_shape_vec(
            (1, 2, 2, 2),
            vec![
                1.0, 5.0,
                2.0, 6.0, 
                3.0, 7.0, 
                4.0, 8.0, 
            ],
        )
        .unwrap();

        let col = im2col(&input, 2, 2, 1, 1, 0, 0, 1, 1);

        assert_eq!(col.shape(), &[1, 8]);
        assert_eq!(
            col.row(0).to_vec(),
            vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]
        );
    }

    #[test]
    fn test_im2col_batch() {
        let input = Array4::from_shape_vec(
            (2, 2, 2, 1),
            vec![
                1.0, 2.0, 3.0, 4.0, 
                5.0, 6.0, 7.0, 8.0, 
            ],
        )
        .unwrap();

        let col = im2col(&input, 2, 2, 1, 1, 0, 0, 1, 1);

        assert_eq!(col.shape(), &[2, 4]);

        assert_eq!(col.row(0).to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(col.row(1).to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
    }
}
