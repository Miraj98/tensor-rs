use crate::{
    prelude::{dim::Dimension, utils::nd_index},
    unique_id::unique_id,
    DataBuffer, DataElement, OwnedData, Tensor, TensorBase,
};
use std::{
    cell::RefCell,
    ops::{Add, Index, Mul, Sub, Div},
};

macro_rules! impl_binary_ops {
    ($math: ident, $trait: ident) => {
        impl<S, A, E> $trait<TensorBase<S, A>> for TensorBase<S, A>
        where
            S: Dimension,
            E: DataElement,
            A: DataBuffer<Item = E> + Index<usize, Output = E>,
        {
            type Output = Tensor<S, E>;

            fn $math(self, rhs: TensorBase<S, A>) -> Self::Output {
                assert_eq!(self.shape(), rhs.shape());
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                for i in 0..self.len() {
                    let idx = nd_index(i, &self.dim, &strides);
                    out_vec[i] = self[idx.clone()].$math(rhs[idx]);
                }

                TensorBase {
                    id: unique_id(),
                    data: OwnedData::new(out_vec),
                    dim: self.dim.clone(),
                    strides,
                    is_leaf: false,
                    requires_grad: self.requires_grad,
                    backward_ops: RefCell::new(None),
                }
            }
        }

        impl<'a, S, A, E> $trait<&'a TensorBase<S, A>> for TensorBase<S, A>
        where
            S: Dimension,
            E: DataElement,
            A: DataBuffer<Item = E> + Index<usize, Output = E>,
        {
            type Output = Tensor<S, E>;

            fn $math(self, rhs: &'a TensorBase<S, A>) -> Self::Output {
                assert_eq!(self.shape(), rhs.shape());
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                for i in 0..self.len() {
                    let idx = nd_index(i, &self.dim, &strides);
                    out_vec[i] = self[idx.clone()].$math(rhs[idx]);
                }

                TensorBase {
                    id: unique_id(),
                    data: OwnedData::new(out_vec),
                    dim: self.dim.clone(),
                    strides,
                    is_leaf: false,
                    requires_grad: self.requires_grad,
                    backward_ops: RefCell::new(None),
                }
            }
        }

        impl<'a, S, A, E> $trait<TensorBase<S, A>> for &'a TensorBase<S, A>
        where
            S: Dimension,
            E: DataElement,
            A: DataBuffer<Item = E> + Index<usize, Output = E>,
        {
            type Output = Tensor<S, E>;

            fn $math(self, rhs: TensorBase<S, A>) -> Self::Output {
                assert_eq!(self.shape(), rhs.shape());
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                for i in 0..self.len() {
                    let idx = nd_index(i, &self.dim, &strides);
                    out_vec[i] = self[idx.clone()].$math(rhs[idx]);
                }

                TensorBase {
                    id: unique_id(),
                    data: OwnedData::new(out_vec),
                    dim: self.dim.clone(),
                    strides,
                    is_leaf: false,
                    requires_grad: self.requires_grad,
                    backward_ops: RefCell::new(None),
                }
            }
        }

        impl<'a, S, A, E> $trait<&'a TensorBase<S, A>> for &'a TensorBase<S, A>
        where
            S: Dimension,
            E: DataElement,
            A: DataBuffer<Item = E> + Index<usize, Output = E>,
        {
            type Output = Tensor<S, E>;

            fn $math(self, rhs: &'a TensorBase<S, A>) -> Self::Output {
                assert_eq!(self.shape(), rhs.shape());
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                for i in 0..self.len() {
                    let idx = nd_index(i, &self.dim, &strides);
                    out_vec[i] = self[idx.clone()].$math(rhs[idx]);
                }

                TensorBase {
                    id: unique_id(),
                    data: OwnedData::new(out_vec),
                    dim: self.dim.clone(),
                    strides,
                    is_leaf: false,
                    requires_grad: self.requires_grad,
                    backward_ops: RefCell::new(None),
                }
            }
        }
    };
}

impl_binary_ops!(add, Add);
impl_binary_ops!(mul, Mul);
impl_binary_ops!(sub, Sub);
impl_binary_ops!(div, Div);
