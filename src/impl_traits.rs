use crate::{
    prelude::{dim::Dimension, utils::nd_index},
    unique_id::unique_id,
    DataBuffer, TensorBase, DataElement, Tensor, TensorView,
};
use std::{cell::RefCell, ops::Index};

impl<S, A> Index<S> for TensorBase<S, A>
where
    A: DataBuffer,
    S: Dimension,
{
    type Output = <A as Index<usize>>::Output;

    fn index(&self, index: S) -> &Self::Output {
        assert_eq!(self.strides.slice().len(), index.slice().len());
        let i = self
            .strides
            .slice()
            .iter()
            .zip(index.slice().iter())
            .fold(0, |acc, (stride, index)| acc + stride * index);
        &self.data[i]
    }
}

impl<S, A> Clone for TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer,
{
    fn clone(&self) -> Self {
        TensorBase {
            id: unique_id(),
            data: self.data.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            is_leaf: self.is_leaf,
            requires_grad: self.requires_grad,
            backward_ops: RefCell::new(None),
        }
    }
}

impl<S, S2, A> PartialEq<Tensor<S2, A>> for Tensor<S, A>
where
    S: Dimension,
    S2: Dimension,
    A: DataElement,
{
    fn eq(&self, other: &Tensor<S2, A>) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        if let Some(self_s) = self.as_slice() {
            if let Some(rhs_s) = other.as_slice() {
                return self_s == rhs_s;
            }
        }

        for i in 0..self.len() {
            let l = nd_index(i, &self.dim);
            let r = nd_index(i, &other.dim);
            if self[l] != other[r] {
                return false;
            }
        }

        true
    }
}

impl<'a, S, S2, A> PartialEq<TensorView<'a, S2, A>> for Tensor<S, A>
where
    S: Dimension,
    S2: Dimension,
    A: DataElement,
{
    fn eq(&self, other: &TensorView<'a, S2, A>) -> bool {
        println!("eq being called");
        if self.shape() != other.shape() {
            println!("Shapes are not equal {:?}, {:?}", self.shape(), other.shape());
            return false;
        }

        if let Some(self_s) = self.as_slice() {
            if let Some(rhs_s) = other.as_slice() {
                return self_s == rhs_s;
            }
        }

        for i in 0..self.len() {
            let l = nd_index(i, &self.dim);
            let r = nd_index(i, &other.dim);
            if self[l] != other[r] {
                return false;
            }
        }

        true
    }
}

impl<'a, S, S2, A> PartialEq<TensorView<'a, S2, A>> for TensorView<'a, S, A>
where
    S: Dimension,
    S2: Dimension,
    A: DataElement,
{
    fn eq(&self, other: &TensorView<'a, S2, A>) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        if let Some(self_s) = self.as_slice() {
            if let Some(rhs_s) = other.as_slice() {
                return self_s == rhs_s;
            }
        }

        for i in 0..self.len() {
            let l = nd_index(i, &self.dim);
            let r = nd_index(i, &other.dim);
            if self[l] != other[r] {
                return false;
            }
        }

        true
    }
}

impl<'a, S, S2, A> PartialEq<Tensor<S2, A>> for TensorView<'a, S, A>
where
    S: Dimension,
    S2: Dimension,
    A: DataElement,
{
    fn eq(&self, other: &Tensor<S2, A>) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        if let Some(self_s) = self.as_slice() {
            if let Some(rhs_s) = other.as_slice() {
                return self_s == rhs_s;
            }
        }

        for i in 0..self.len() {
            let l = nd_index(i, &self.dim);
            let r = nd_index(i, &other.dim);
            if self[l] != other[r] {
                return false;
            }
        }

        true
    }
}
