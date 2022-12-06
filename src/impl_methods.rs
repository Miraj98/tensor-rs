use std::{cell::RefCell, ptr::NonNull};

use crate::{
    prelude::{dim::Dimension, utils::generate_strides},
    unique_id::unique_id,
    DataBuffer, DataElement, OwnedData, Tensor, TensorBase, TensorView, ViewData,
};

impl<S, A> TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer,
{
    pub fn from_vec<Dtype: DataElement>(a: Vec<Dtype>, dim: S) -> Tensor<S, Dtype> {
        let total_len = dim.count();
        assert_eq!(total_len, a.len());
        let strides = generate_strides(&dim);
        TensorBase {
            id: unique_id(),
            data: OwnedData::new(a),
            dim,
            strides,
            is_leaf: true,
            requires_grad: false,
            backward_ops: RefCell::new(None),
        }
    }

    pub fn as_slice(&self) -> &[A::Item] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.dim.count()) }
    }

    pub fn view(&self) -> TensorView<'_, S, A::Item> {
        TensorBase {
            id: self.id,
            data: ViewData {
                ptr: NonNull::new(self.data.as_mut_ptr()).unwrap(),
                marker: std::marker::PhantomData,
            },
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            is_leaf: self.is_leaf,
            requires_grad: self.requires_grad,
            backward_ops: RefCell::new(None),
        }
    }

    pub fn t(&self) -> TensorView<'_, S, A::Item> {
        let mut self_view = self.view();
        let strides = self.strides.rev();
        let dim = self.dim.rev();
        self_view.dim = dim;
        self_view.strides = strides;

        self_view
    }

    pub fn reshape(&self, dim: S) -> TensorView<'_, S, A::Item> {
        assert_eq!(self.dim.count(), dim.count());
        let mut self_view = self.view();
        let strides = generate_strides(&dim);
        self_view.dim = dim;
        self_view.strides = strides;
        self_view
    }

    pub fn broadcast<K>(&self, dim: K) -> TensorView<'_, K, A::Item>
    where
        K: Dimension,
    {
        assert!(self.dim.ndim() <= dim.ndim());
        let mut new_strides = dim.clone();
        let mut new_strides_iter = new_strides.slice_mut().iter_mut().rev();

        for ((er, es), tr) in self
            .dim
            .slice()
            .iter()
            .rev()
            .zip(self.strides.slice().iter().rev())
            .zip(new_strides_iter.by_ref())
        {
            if *er == *tr {
                *tr = *es;
            } else {
                assert_eq!(*er, 1);
                *tr = 0;
            }
        }

        for tr in new_strides_iter {
            *tr = 0;
        }

        TensorBase {
            id: self.id,
            data: ViewData {
                ptr: NonNull::new(self.data.as_mut_ptr()).unwrap(),
                marker: std::marker::PhantomData,
            },
            dim,
            strides: new_strides,
            is_leaf: self.is_leaf,
            requires_grad: self.requires_grad,
            backward_ops: RefCell::new(None),
        }
    }
}

impl<S, A> Clone for TensorBase<S, A> where S: Dimension, A: DataBuffer {
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
