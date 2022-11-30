pub mod dim;
pub mod impl_index;
pub mod ops;
pub mod utils;
pub mod impl_constructors;

use crate::num_taits::Zero;
use crate::prelude::BackwardOps;
use crate::unique_id::{unique_id, UniqueId};
use std::cell::RefCell;
use std::cmp::{max, min};
use std::marker::PhantomData;
use std::rc::Rc;
use std::usize;
use utils::{generate_strides, tnsr_idx, vec_id};
use self::dim::Dimension;

#[derive(Debug)]
pub struct TensorBase<S, Dtype = f32>
where
    S: Dimension,
{
    id: UniqueId,
    data: Rc<Vec<Dtype>>,
    dim: S,
    strides: S,
    stride_reps: S,
    marker: PhantomData<Dtype>,
    backward_ops: RefCell<Option<BackwardOps>>,
    is_leaf: bool,
    requires_grad: bool,
}

impl<S, Dtype> TensorBase<S, Dtype>
where
    S: Dimension,
{
    pub fn from_vec(a: Vec<Dtype>, dim: S) -> TensorBase<S, Dtype> {
        let total_len = dim.get_iter().fold(1, |acc, val| acc * *val);
        assert_eq!(total_len, a.len());
        let strides = generate_strides(&dim);
        TensorBase {
            id: unique_id(),
            data: Rc::new(a),
            dim,
            strides,
            stride_reps: S::ones(),
            marker: PhantomData,
            is_leaf: true,
            requires_grad: false,
            backward_ops: RefCell::new(None),
        }
    }

    pub fn requires_grad(mut self, b: bool) -> Self {
        self.requires_grad = b;
        if b && self.is_leaf && self.backward_ops.borrow().is_none() {
            *self.backward_ops.borrow_mut() = Some(BackwardOps(Vec::new()));
        }

        self
    }

    pub fn with_backops(self, backops: Option<BackwardOps>) -> Self {
        *self.backward_ops.borrow_mut() = backops;
        self
    }

    pub fn dim(&self) -> S {
        self.dim.clone()
    }

    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }

    pub fn len(&self) -> usize {
        self.dim.get_iter().fold(1, |acc, val| acc * *val)
    }

    pub fn shape(&self) -> &[usize] {
        self.dim.slice()
    }

    pub fn strides(&self) -> S {
        self.strides.clone()
    }

    pub fn id(&self) -> &UniqueId {
        &self.id
    }

    pub(crate) fn detach_backward_ops(&self) -> Option<BackwardOps> {
        self.backward_ops.borrow_mut().take()
    }

    pub fn update_stride_reps(&mut self, a: S) {
        self.stride_reps = a;
    }

    pub fn view(&self) -> TensorBase<S, &Dtype> {
        let a = self.data.iter().collect();

        TensorBase {
            id: self.id.clone(),
            data: Rc::new(a),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            stride_reps: S::ones(),
            marker: PhantomData,
            is_leaf: self.is_leaf,
            requires_grad: false,
            backward_ops: RefCell::new(None),
        }
    }

    pub fn broadcast<K>(&self, to_dim: K) -> TensorBase<K, &Dtype>
    where
        K: Dimension,
    {
        assert!(self.dim.ndim() <= to_dim.ndim());
        let num_l = self.ndim();
        let num_r = to_dim.ndim();

        // New dimensions
        let mut extended_dims = K::ones();
        self.dim
            .get_iter()
            .enumerate()
            .for_each(|(i, val)| extended_dims[num_r - num_l + i] = *val);
        // Old dimensions and strides but padding the extra dims as 1
        let padded_dims = extended_dims.clone();
        let padded_strides = generate_strides(&padded_dims);

        let mut stride_reps = K::ones();
        self.stride_reps
            .get_iter()
            .enumerate()
            .for_each(|(i, val)| stride_reps[num_r - num_l + i] = *val);

        for i in 0..num_r {
            if extended_dims[num_r - 1 - i] != to_dim[num_r - 1 - i] {
                if min(to_dim[num_r - 1 - i], extended_dims[num_r - 1 - i]) == 1 {
                    extended_dims[num_r - 1 - i] =
                        max(to_dim[num_r - 1 - i], extended_dims[num_r - 1 - i]);
                    if extended_dims[num_r - 1 - i] == 1 {
                        stride_reps[num_r - 1 - i] *= extended_dims[num_r - 1 - i]
                    }
                } else {
                    panic!("Incompatible for broadcasting");
                }
            } else {
                extended_dims[num_r - 1 - i] = extended_dims[num_r - 1 - i];
            }
        }

        // Traverse all data points to generate the broadcasted view
        let new_len = extended_dims.get_iter().fold(1, |acc, val| acc * *val);
        let new_strides = generate_strides(&extended_dims);
        let mut broadcasted_data = Vec::<&Dtype>::with_capacity(new_len);
        for i in 0..new_len {
            let id = vec_id(tnsr_idx(i, &new_strides), &padded_dims, &padded_strides);
            broadcasted_data.push(&self.data[id]);
        }

        TensorBase {
            id: self.id.clone(),
            data: Rc::new(broadcasted_data),
            dim: extended_dims.clone(),
            strides: generate_strides(&extended_dims),
            marker: PhantomData,
            stride_reps,
            is_leaf: false,
            requires_grad: self.requires_grad,
            backward_ops: RefCell::new(self.backward_ops.borrow_mut().take()), // TODO: Not sure about this right now
        }
    }
}


impl<const D: usize, Dtype> Clone for TensorBase<[usize; D], Dtype> {
    fn clone(&self) -> Self {
        TensorBase {
            id: self.id.clone(),
            data: Rc::clone(&self.data),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            stride_reps: self.stride_reps.clone(),
            marker: self.marker,
            is_leaf: self.is_leaf,
            requires_grad: self.requires_grad,
            backward_ops: RefCell::new(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::zip;

    #[test]
    fn broadcast_test() {
        let a = vec![3, 4];
        let t = TensorBase::from_vec(a, [2, 1]);
        let a = t.broadcast([3, 1, 5]);

        assert_eq!(a.strides, [10, 5, 1]);
        assert_eq!(a.dim, [3, 2, 5]);
        let expected_out = vec![
            3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 4, 4, 4, 4,
            4,
        ];

        for (eo, ao) in zip(expected_out.iter(), a.data.iter()) {
            assert_eq!(eo, *ao);
        }
    }

    #[test]
    fn test_backward_ops_take() {
        let a = vec![3, 4];
        let t = TensorBase::from_vec(a, [2, 1]).requires_grad(true);
        assert!(t.backward_ops.borrow().is_some());
        let ops = t.detach_backward_ops();
        assert!(t.backward_ops.borrow().is_none());
        assert!(ops.is_some());
        assert!(ops.unwrap().0.len() == 0);
    }
}
