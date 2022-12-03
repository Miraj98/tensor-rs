use crate::{
    num_taits::{One, Zero},
    prelude::{BackwardOps, Merge},
};
use num_integer::Integer;
use std::mem::{size_of, ManuallyDrop};

use super::{dim::Dimension, impl_constructors::TensorConstructors, Tensor, TensorBase};

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

pub fn vec_id<S>(tnsr_idx: S, dims: &S, strides: &S) -> usize
where
    S: Dimension,
{
    let id = tnsr_idx
        .get_iter()
        .enumerate()
        .fold(0, |acc, (i, val)| acc + strides[i] * (val % dims[i]));

    id
}

pub unsafe fn unlimited_transmute<A, B>(data: A) -> B {
    // safe when sizes are equal and caller guarantees that representations are equal
    assert_eq!(size_of::<A>(), size_of::<B>());
    let old_data = ManuallyDrop::new(data);
    (&*old_data as *const A as *const B).read()
}

pub fn merge_backward_ops<L, R, Dtype>(
    lhs: &Tensor<L, Dtype>,
    rhs: &Tensor<R, Dtype>,
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

pub fn reduced_grad<L, R, Dtype>(reduce_to: L, incoming_grad: &Tensor<R, Dtype>) -> Tensor<L, Dtype>
where
    L: Dimension,
    R: Dimension,
    Dtype: One + Zero + std::ops::Add<Dtype, Output = Dtype>,
{
    let mut t = vec![Dtype::zero(); reduce_to.count()];
    if reduce_to.shape() != incoming_grad.shape() {
        let mut padded = R::ones();
        reduce_to.get_iter().enumerate().for_each(|(i, val)| {
            padded[incoming_grad.ndim() - reduce_to.ndim() + i] = *val;
        });
        let padded_strides = generate_strides(&padded);
        let broadcasted_strides = generate_strides(&incoming_grad.dim());
        let broadcasted_count = incoming_grad.dim().count();

        for i in 0..broadcasted_count {
            let incoming_grad_idx = tnsr_idx(i, &broadcasted_strides);
            let idx = vec_id(incoming_grad_idx.clone(), &padded, &padded_strides);
            t[idx] = t[idx] + incoming_grad[incoming_grad_idx];
        }
    }

    TensorBase::from_vec(t, reduce_to)
}
