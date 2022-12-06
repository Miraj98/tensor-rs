pub mod dim;
pub mod impl_constructors;
pub mod impl_eq;
pub mod impl_index;
pub mod ops;
pub mod utils;

use self::dim::Dimension;
use self::impl_constructors::TensorConstructors;
use self::utils::unlimited_transmute;
use crate::num_taits::{One, Zero};
use crate::prelude::{BackwardOps, GradientMap};
use crate::unique_id::{unique_id, UniqueId};
use std::cell::RefCell;
use std::cmp::{max, min};
use std::iter::zip;
use std::ops::{Deref, Div};
use std::rc::Rc;
use std::usize;
use utils::{generate_strides, tnsr_idx, vec_id};

pub trait Data: Clone + PartialEq {
    type Dtype: PartialEq + Copy;

    fn as_ptr(&self) -> *const Self::Dtype;
    fn map(&self, f: impl Fn(Self::Dtype) -> Self::Dtype) -> OwnedData<Self::Dtype>;
    fn sum(&self) -> Self::Dtype
    where
        Self::Dtype: Zero;
    fn mean(&self) -> Self::Dtype
    where
        Self::Dtype: Zero + Div<f32, Output = Self::Dtype>;
}

#[derive(Debug)]
pub struct OwnedData<Dtype>
where
    Dtype: PartialEq,
{
    data: Rc<Vec<Dtype>>,
}

impl<Dtype: PartialEq> PartialEq for OwnedData<Dtype> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<Dtype: PartialEq + Copy> Data for OwnedData<Dtype> {
    type Dtype = Dtype;

    fn as_ptr(&self) -> *const Self::Dtype {
        self.data.as_ptr()
    }

    fn map(&self, f: impl Fn(Self::Dtype) -> Self::Dtype) -> OwnedData<Self::Dtype> {
        let mut result = self.clone();
        result.data = Rc::new(self.data.iter().map(move |x| f(*x)).collect());
        result
    }

    fn sum(&self) -> Self::Dtype
    where
        Self::Dtype: Zero,
    {
        self.data
            .iter()
            .fold(Self::Dtype::zero(), |acc, x| acc + *x)
    }

    fn mean(&self) -> Self::Dtype
    where
        Self::Dtype: Zero + Div<f32, Output = Self::Dtype>,
    {
        self.data
            .iter()
            .fold(Self::Dtype::zero(), |acc, x| acc + *x)
            / self.data.len() as f32
    }
}

impl<Dtype: PartialEq> Deref for OwnedData<Dtype> {
    type Target = Rc<Vec<Dtype>>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<Dtype: PartialEq> Clone for OwnedData<Dtype> {
    fn clone(&self) -> Self {
        Self {
            data: Rc::clone(&self.data),
        }
    }
}

#[derive(Debug)]
pub struct ViewData<'a, Dtype> {
    data: Rc<Vec<&'a Dtype>>,
}

impl<Dtype: PartialEq> PartialEq for ViewData<'_, Dtype> {
    fn eq(&self, other: &Self) -> bool {
        let mut is_eq = self.data.len() == other.data.len();
        for (s, o) in zip(self.data.iter(), other.data.iter()) {
            is_eq = is_eq && **s == **o;
        }

        is_eq
    }
}

impl<Dtype: PartialEq> PartialEq<OwnedData<Dtype>> for ViewData<'_, Dtype> {
    fn eq(&self, other: &OwnedData<Dtype>) -> bool {
        let mut is_eq = self.data.len() == other.data.len();
        for (s, o) in zip(self.data.iter(), other.data.iter()) {
            is_eq = is_eq && **s == *o;
        }

        is_eq
    }
}

impl<Dtype: PartialEq> PartialEq<ViewData<'_, Dtype>> for OwnedData<Dtype> {
    fn eq(&self, other: &ViewData<'_, Dtype>) -> bool {
        let mut is_eq = self.data.len() == other.data.len();
        for (s, o) in zip(self.data.iter(), other.data.iter()) {
            is_eq = is_eq && *s == **o;
        }

        is_eq
    }
}

impl<'a, Dtype: PartialEq> Data for ViewData<'a, Dtype>
where
    Dtype: Copy,
{
    type Dtype = Dtype;

    fn as_ptr(&self) -> *const Self::Dtype {
        let a = self.data.iter().map(|x| **x).collect::<Vec<_>>();
        a.as_ptr()
    }
    fn map(&self, f: impl Fn(Self::Dtype) -> Self::Dtype) -> OwnedData<Self::Dtype> {
        OwnedData {
            data: Rc::new(self.data.iter().map(|x| f(**x)).collect()),
        }
    }

    fn sum(&self) -> Self::Dtype
    where
        Self::Dtype: Zero,
    {
        self.data
            .iter()
            .fold(Self::Dtype::zero(), |acc, x| acc + **x)
    }

    fn mean(&self) -> Self::Dtype
    where
        Self::Dtype: Zero + Div<f32, Output = Self::Dtype>,
    {
        self.data
            .iter()
            .fold(Self::Dtype::zero(), |acc, x| acc + **x)
            / self.data.len() as f32
    }
}

impl<'a, Dtype> Deref for ViewData<'a, Dtype> {
    type Target = Rc<Vec<&'a Dtype>>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a, Dtype> Clone for ViewData<'a, Dtype> {
    fn clone(&self) -> Self {
        Self {
            data: Rc::clone(&self.data),
        }
    }
}

#[derive(Debug)]
pub struct TensorBase<S, A>
where
    S: Dimension,
    A: Data,
{
    id: UniqueId,
    data: A,
    dim: S,
    strides: S,
    stride_reps: S,
    backward_ops: RefCell<Option<BackwardOps>>,
    is_leaf: bool,
    requires_grad: bool,
}

impl<S, A> TensorBase<S, A>
where
    S: Dimension,
    A: Data,
{
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
        self.dim.shape()
    }

    pub fn strides(&self) -> S {
        self.strides.clone()
    }
    pub fn map(&self, f: impl Fn(A::Dtype) -> A::Dtype) -> Tensor<S, A::Dtype> {
        TensorBase {
            id: unique_id(),
            data: self.data.map(f),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            stride_reps: self.stride_reps.clone(),
            backward_ops: RefCell::new(None),
            is_leaf: false,
            requires_grad: self.requires_grad,
        }
    }

    pub fn id(&self) -> &UniqueId {
        &self.id
    }

    pub fn into_dimensionality<D2: Dimension>(&self) -> &TensorBase<D2, A> {
        unsafe { unlimited_transmute(self) }
    }
}

impl<S, Dtype> TensorBase<S, OwnedData<Dtype>>
where
    S: Dimension,
    Dtype: PartialEq + Copy,
{
    pub fn from_vec(a: Vec<Dtype>, dim: S) -> TensorBase<S, OwnedData<Dtype>> {
        let total_len = dim.count();
        assert_eq!(total_len, a.len());
        let strides = generate_strides(&dim);
        TensorBase {
            id: unique_id(),
            data: OwnedData { data: Rc::new(a) },
            dim,
            strides,
            stride_reps: S::ones(),
            is_leaf: true,
            requires_grad: false,
            backward_ops: RefCell::new(None),
        }
    }

    pub fn _sum(&self) -> Tensor<[usize; 0], Dtype>
    where
        Dtype: Zero,
    {
        let data = self.data.sum();

        TensorBase {
            id: unique_id(),
            data: OwnedData { data: Rc::new(vec![data]) },
            dim: [],
            strides: [],
            stride_reps: [],
            backward_ops: RefCell::new(None),
            is_leaf: false,
            requires_grad: self.requires_grad,
        }
    }

    pub fn _mean(&self) -> Tensor<[usize; 0], Dtype>
    where
        Dtype: Zero + Div<f32, Output = Dtype>,
    {
        let data = self.data.mean();

        TensorBase {
            id: unique_id(),
            data: OwnedData { data: Rc::new(vec![data]) },
            dim: [],
            strides: [],
            stride_reps: [],
            backward_ops: RefCell::new(None),
            is_leaf: false,
            requires_grad: self.requires_grad,
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

    pub(crate) fn detach_backward_ops(&self) -> Option<BackwardOps> {
        self.backward_ops.borrow_mut().take()
    }

    pub(crate) fn put_backward_ops(&self, backops: Option<BackwardOps>) {
        *self.backward_ops.borrow_mut() = backops;
    }

    pub fn update_stride_reps(&mut self, a: S) {
        self.stride_reps = a;
    }

    pub fn view(&self) -> TensorBase<S, ViewData<'_, Dtype>>
    where
        Dtype: Copy,
    {
        let a = self.data.iter().collect();
        let view_data = ViewData { data: Rc::new(a) };

        TensorBase {
            id: self.id.clone(),
            data: view_data,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            stride_reps: S::ones(),
            is_leaf: self.is_leaf,
            requires_grad: false,
            backward_ops: RefCell::new(None),
        }
    }

    pub fn t(&self) -> TensorBase<S, ViewData<'_, Dtype>>
    where
        Dtype: Copy,
    {
        let strides = self.strides.rev();
        let dim = self.dim.rev();
        let a = self.data.iter().collect();
        TensorBase {
            id: self.id,
            data: ViewData { data: Rc::new(a) },
            dim,
            strides,
            stride_reps: self.stride_reps.clone(),
            backward_ops: RefCell::new(None),
            is_leaf: false,
            requires_grad: false,
        }
    }

    pub fn broadcast<K>(&self, to_dim: K) -> TensorBase<K, ViewData<Dtype>>
    where
        K: Dimension,
        Dtype: Copy,
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
            data: ViewData {
                data: Rc::new(broadcasted_data),
            },
            dim: extended_dims.clone(),
            strides: generate_strides(&extended_dims),
            stride_reps,
            is_leaf: false,
            requires_grad: self.requires_grad,
            backward_ops: RefCell::new(self.backward_ops.borrow_mut().take()), // TODO: Not sure about this right now
        }
    }

    // pub fn backward(&self) -> GradientMap
    // where
    //     Dtype: Zero + One + 'static,
    //     S: Dimension + 'static,
    // {
    //     if self.backward_ops.borrow().is_none() {
    //         panic!("Use requires_grad(true) to enable gradient computation");
    //     }

    //     let mut backops = self.detach_backward_ops().unwrap();
    //     let id = self.id;
    //     let dim = self.dim();
    //     backops.add_backward_op(move |grad| {
    //         let mut_ref: &mut Tensor<S, Dtype> = grad.mut_grad_by_id(id, dim.clone());
    //         *mut_ref = Tensor::ones(dim);
    //     });

    //     let grads = backops.execute();
    //     grads
    // }
}

impl<S, A> Clone for TensorBase<S, A>
where
    S: Dimension,
    A: Data,
{
    fn clone(&self) -> Self {
        TensorBase {
            id: self.id.clone(),
            data: self.data.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            stride_reps: self.stride_reps.clone(),
            is_leaf: self.is_leaf,
            requires_grad: self.requires_grad,
            backward_ops: RefCell::new(None),
        }
    }
}

pub type Tensor<S, A> = TensorBase<S, OwnedData<A>>;
pub type TensorView<'a, S, A> = TensorBase<S, ViewData<'a, A>>;

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
