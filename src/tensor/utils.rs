use crate::prelude::{BackwardOps, Merge};
use num_integer::Integer;
use std::mem::{size_of, ManuallyDrop};

use super::{dim::Dimension, TensorBase};

pub fn generate_strides<S>(dim: &S) -> S
where
    S: Dimension,
{
    let mut strides = S::ones();
    for i in (0..dim.ndim() - 1).rev() {
        strides[i] = dim[i + 1] * strides[i + 1];
    }
    strides
}

pub fn tnsr_idx<S>(vec_id: usize, strides: &S) -> S
where
    S: Dimension,
{
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

pub fn vec_id<S>(tnsr_idx: S, padded_dims: &S, padded_strides: &S) -> usize
where
    S: Dimension,
{
    let id = tnsr_idx.get_iter().enumerate().fold(0, |acc, (i, val)| {
        acc + padded_strides[i] * (val % padded_dims[i])
    });

    id
}

pub unsafe fn unlimited_transmute<A, B>(data: A) -> B {
    // safe when sizes are equal and caller guarantees that representations are equal
    assert_eq!(size_of::<A>(), size_of::<B>());
    let old_data = ManuallyDrop::new(data);
    (&*old_data as *const A as *const B).read()
}

pub fn merge_backward_ops<L, R, Dtype>(
    lhs: &TensorBase<L, Dtype>,
    rhs: &TensorBase<R, Dtype>,
) -> Option<BackwardOps>
where
    L: Dimension,
    R: Dimension,
{
    let lhs_ops = lhs.detach_backward_ops();
    let rhs_ops = rhs.detach_backward_ops();
    let merged = lhs_ops.merge(rhs_ops);
    merged
}
