use num_integer::Integer;

use super::dim::Dimension;

pub fn generate_strides<S>(dim: &S) -> S where S: Dimension {
    let mut strides = S::ones();
    for i in (0..dim.ndim() - 1).rev() {
        strides[i] = dim[i + 1] * strides[i + 1];
    }
    strides
}

pub fn tnsr_idx<S>(vec_id: usize, strides: &S) -> S where S: Dimension{
    let mut idx = S::zeros();
    let mut r = vec_id;
    for (is, s) in strides.get_iter().enumerate() {
        let (q, m) = r.div_rem(s);
        idx[is] = q;
        if m == 0 {
            break;
        }
        r = m;
    }
    idx
}

pub fn vec_id<S>(
    tnsr_idx: S,
    padded_dims: &S,
    padded_strides: &S,
) -> usize where S: Dimension {
    let id = tnsr_idx.get_iter().enumerate().fold(0, |acc, (i, val)| {
        acc + padded_strides[i] * (val % padded_dims[i])
    });

    id
}
