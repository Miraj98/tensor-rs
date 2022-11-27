use num_integer::Integer;

pub fn generate_strides<const N: usize>(dim: &[usize; N]) -> [usize; N] {
    let mut strides = [1; N];
    for i in (0..dim.len() - 1).rev() {
        strides[i] = dim[i + 1] * strides[i + 1];
    }
    strides
}

pub fn tnsr_idx<const N: usize>(vec_id: usize, strides: &[usize; N]) -> [usize; N] {
    let mut idx = [0; N];
    let mut r = vec_id;
    for (is, s) in strides.iter().enumerate() {
        let (q, m) = r.div_rem(s);
        idx[is] = q;
        if m == 0 {
            break;
        }
        r = m;
    }
    idx
}

pub fn vec_id<const N: usize>(
    tnsr_idx: [usize; N],
    padded_dims: &[usize; N],
    padded_strides: &[usize; N],
) -> usize {
    let id = tnsr_idx.iter().enumerate().fold(0, |acc, (i, val)| {
        acc + padded_strides[i] * (val % padded_dims[i])
    });

    id
}
