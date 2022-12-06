use crate::{prelude::dim::Dimension, Tensor, DataElement, TensorBase};


pub trait TensorConstructors<Dtype> where Dtype: DataElement {
    type S: Dimension;
    fn ones(dim: Self::S) -> Self;
    fn zeros(dim: Self::S) -> Self;
    fn from_elem(dim: Self::S, elem: Dtype) -> Self;
}

impl<S, Dtype> TensorConstructors<Dtype> for Tensor<S, Dtype>
where
    S: Dimension,
    Dtype: DataElement,
{
    type S = S;

    fn ones(dim: S) -> Self {
        let a = vec![Dtype::one(); dim.count()];
        TensorBase::from_vec(a, dim)
    }

    fn zeros(dim: S) -> Self {
        let a = vec![Dtype::zero(); dim.count()];
        TensorBase::from_vec(a, dim)
    }

    fn from_elem(dim: S, elem: Dtype) -> Self {
        let a = vec![elem; dim.count()];
        TensorBase::from_vec(a, dim)
    }
}