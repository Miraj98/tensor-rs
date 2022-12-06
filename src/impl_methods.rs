use std::{cell::RefCell, ptr::NonNull};

use crate::{
    prelude::{dim::Dimension, utils::generate_strides},
    unique_id::unique_id,
    DataBuffer, DataElement, OwnedData, Tensor, TensorBase, TensorView, ViewData,
};

impl<S, Dtype> Tensor<S, Dtype>
where
    S: Dimension,
    Dtype: DataElement,
{
    pub fn from_vec(a: Vec<Dtype>, dim: S) -> Tensor<S, Dtype> {
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
}

impl<S, A> TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer,
{
    pub fn len(&self) -> usize {
        self.dim.count()
    }

    pub fn as_slice(&self) -> Option<&[A::Item]> {
        if self.is_standard_layout() {
            unsafe {
                Some(std::slice::from_raw_parts(
                    self.data.as_ptr(),
                    self.dim.count(),
                ))
            }
        } else {
            None
        }
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

    pub fn is_standard_layout(&self) -> bool {
        self.strides.slice() == generate_strides(&self.dim).slice()
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

    pub fn shape(&self) -> &[usize] {
        self.dim.slice()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_test() {
        let avec: Vec<f32> = vec![3., 4.];
        let a = TensorBase::from_vec(avec, [2, 1]);
        let broadcasted = a.broadcast([3, 2, 5]);
        let similar = TensorBase::from_vec(
            vec![
                3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0,
                4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
            ],
            [3, 2, 5],
        );
        assert_eq!(broadcasted, similar);
    }
}
