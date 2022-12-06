pub mod unique_id;
pub mod tensor;
pub mod gradient;

pub mod impl_methods;

use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign},
    ptr::NonNull,
    rc::Rc, cell::RefCell,
};

use crate::unique_id::UniqueId;
use crate::tensor::dim::Dimension;
use crate::gradient::BackwardOps;

pub mod num_taits;

pub mod prelude {
    pub use crate::gradient::*;
    pub use crate::tensor::*;
    pub use crate::unique_id::*;
}

#[derive(Debug)]
pub struct TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer,
{
    id: UniqueId,
    data: A,
    dim: S,
    strides: S,
    backward_ops: RefCell<Option<BackwardOps>>,
    is_leaf: bool,
    requires_grad: bool,
}

pub type Tensor<S, Dtype> = TensorBase<S, OwnedData<Dtype>>;
pub type TensorView<'a, S, Dtype> = TensorBase<S, ViewData<'a, Dtype>>;

#[derive(Debug)]
pub struct OwnedData<E>
where
    E: DataElement,
{
    data: Rc<Vec<E>>,
}

impl<E: DataElement> OwnedData<E> {
    pub fn new(data: Vec<E>) -> Self {
        OwnedData { data: Rc::new(data) }
    }
}

impl<E: DataElement> Clone for OwnedData<E> {
    fn clone(&self) -> Self {
        OwnedData { data: Rc::clone(&self.data) }
    }
}

impl<E: DataElement> DataBuffer for OwnedData<E> {
    type Item = E;

    fn as_ptr(&self) -> *const Self::Item {
        self.data.as_ptr()
    }

    fn as_mut_ptr(&self) -> *mut Self::Item {
        self.data.as_ptr() as *mut Self::Item    
    }
}

pub struct ViewData<'a, E>
where
    E: DataElement,
{
    ptr: NonNull<E>,
    marker: PhantomData<&'a E>,
}

impl<'a, E: DataElement> Clone for ViewData<'a, E> {
    fn clone(&self) -> Self {
        ViewData {
            ptr: self.ptr.clone(),
            marker: PhantomData,
        }
    }
}

impl<'a, E: DataElement> DataBuffer for ViewData<'a, E> {
    type Item = E;

    fn as_ptr(&self) -> *const Self::Item {
        self.ptr.as_ptr() as *const Self::Item
    }

    fn as_mut_ptr(&self) -> *mut Self::Item {
        self.ptr.as_ptr()
    }
}

pub trait DataBuffer: Clone {
    type Item: DataElement;

    fn as_ptr(&self) -> *const Self::Item;
    fn as_mut_ptr(&self) -> *mut Self::Item;
}

pub trait DataElement:
    PartialEq
    + Copy
    + Add<Output = Self>
    + AddAssign<Self>
    + Sub<Output = Self>
    + SubAssign<Self>
    + Mul<Output = Self>
    + MulAssign<Self>
    + Div<Output = Self>
{
    type Dtype;

    fn one() -> Self;
    fn is_one(&self) -> bool;
    fn zero() -> Self;
    fn is_zero(&self) -> bool;
    fn sigmoid(&self) -> Self;
    fn from_usize(x: usize) -> Self;
}

impl DataElement for f32 {
    type Dtype = f32;
    fn one() -> Self {
        1.
    }
    fn is_one(&self) -> bool {
        *self == Self::one()
    }
    fn zero() -> Self {
        0.
    }
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
    fn sigmoid(&self) -> Self {
        1. / (1. + (-*self).exp())
    }
    fn from_usize(x: usize) -> Self {
        x as f32
    }
}

impl DataElement for f64 {
    type Dtype = f64;
    fn one() -> Self {
        1.
    }
    fn is_one(&self) -> bool {
        *self == Self::one()
    }
    fn zero() -> Self {
        0.
    }
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
    fn sigmoid(&self) -> Self {
        1. / (1. + (-*self).exp())
    }
    fn from_usize(x: usize) -> Self {
        x as f64
    }
}
