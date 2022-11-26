pub fn generate_strides<const N: usize>(dim: &[usize; N]) -> [usize; N] {
    let mut strides = [1; N];
    for i in (0..dim.len() - 1).rev() {
        strides[i] = dim[i + 1] * strides[i + 1];
    }
    strides 
}