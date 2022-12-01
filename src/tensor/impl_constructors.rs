use crate::num_taits::{One, Zero};

use super::{dim::Dimension, TensorBase, Tensor};

pub trait TensorConstructors {
    type S: Dimension;
    fn ones(dim: Self::S) -> Self;
    fn zeros(dim: Self::S) -> Self;
}

impl<S, Dtype> TensorConstructors for Tensor<S, Dtype>
where
    S: Dimension,
    Dtype: One + Zero,
{
    type S = S;

    fn ones(dim: S) -> Self {
        let total_len: usize = dim.get_iter().fold(1, |acc, val| acc * *val);
        let a = vec![Dtype::one(); total_len];
        TensorBase::from_vec(a, dim)
    }

    fn zeros(dim: S) -> Self {
        let total_len = dim.get_iter().fold(1, |acc, val| acc * *val);
        let a = vec![Dtype::zero(); total_len];
        TensorBase::from_vec(a, dim)
    }
}
