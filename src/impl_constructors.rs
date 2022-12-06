use crate::{prelude::dim::Dimension, Tensor, DataElement, TensorBase};


pub trait TensorConstructors {
    type S: Dimension;
    fn ones(dim: Self::S) -> Self;
    fn zeros(dim: Self::S) -> Self;
}

impl<S, Dtype> TensorConstructors for Tensor<S, Dtype>
where
    S: Dimension,
    Dtype: DataElement,
{
    type S = S;

    fn ones(dim: S) -> Self {
        let total_len: usize = dim.get_iter().fold(1, |acc, val| acc * *val);
        println!("total_len: {}", total_len);
        let a = vec![Dtype::one(); total_len];
        TensorBase::from_vec(a, dim)
    }

    fn zeros(dim: S) -> Self {
        let total_len = dim.get_iter().fold(1, |acc, val| acc * *val);
        let a = vec![Dtype::zero(); total_len];
        TensorBase::from_vec(a, dim)
    }
}