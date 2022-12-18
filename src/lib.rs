pub mod dim;
pub mod gradient;
pub mod unique_id;
pub mod utils;

pub mod impl_binary_ops;
pub mod impl_constructors;
pub mod impl_methods;
pub mod impl_processing_ops;
pub mod impl_reduce_ops;
pub mod impl_traits;
pub mod impl_unary_ops;

pub mod neural_nets;
pub mod mnist;

use std::{
    cell::RefCell,
    fmt::Debug,
    marker::PhantomData,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign},
    ptr::NonNull,
    rc::Rc,
};

use rand::{thread_rng, Rng};
use rand_distr::StandardNormal;

use crate::dim::Dimension;
use crate::gradient::BackwardOps;
use crate::unique_id::UniqueId;

pub mod num_taits;

pub mod prelude {
    pub use crate::gradient::*;
    pub use crate::unique_id::*;
}

#[derive(Debug)]
pub struct TensorBase<S, A = OwnedData<f32>>
where
    S: Dimension,
    A: DataBuffer,
{
    id: UniqueId,
    ptr: NonNull<A::Item>,
    data: A,
    dim: S,
    strides: S,
    pub backward_ops: RefCell<Option<BackwardOps>>,
    is_leaf: bool,
    requires_grad: bool,
}

pub type Tensor<S, Dtype = f32> = TensorBase<S, OwnedData<Dtype>>;
pub type TensorView<'a, S, E> = TensorBase<S, ViewData<&'a E>>;
pub type TensorViewMut<'a, S, E> = TensorBase<S, ViewData<&'a mut E>>;

#[derive(Debug)]
pub struct OwnedData<E>
where
    E: DataElement,
{
    data: Rc<Vec<E>>,
}

impl<E> DataBuffer for OwnedData<E>
where
    E: DataElement,
{
    type Item = E;
}

impl<E: DataElement> OwnedData<E> {
    pub fn new(data: Vec<E>) -> Self {
        OwnedData {
            data: Rc::new(data),
        }
    }

    pub fn from(data: Rc<Vec<E>>) -> Self {
        OwnedData { data }
    }
}

impl<E: DataElement> Clone for OwnedData<E> {
    fn clone(&self) -> Self {
        OwnedData {
            data: Rc::clone(&self.data),
        }
    }
}

#[derive(Debug)]
pub struct ViewData<E> {
    marker: PhantomData<E>,
}

impl<E> Clone for ViewData<E> {
    fn clone(&self) -> Self {
        ViewData {
            marker: PhantomData::<E>,
        }
    }
}

impl<E> DataBuffer for ViewData<&E>
where
    E: DataElement,
{
    type Item = E;
}

impl<E> DataBuffer for ViewData<&mut E>
where
    E: DataElement,
{
    type Item = E;
}

pub trait DataBuffer: Clone {
    type Item: DataElement;
}

pub trait DataElement:
    PartialEq
    + Debug
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
    fn relu(&self) -> Self;
    fn from_usize(x: usize) -> Self;
    fn randn() -> Self;
}

pub trait IntoFloat32 {
    fn f32(self) -> f32;
}
pub trait IntoFloat64 {
    fn f64(self) -> f64;
}

macro_rules! impl_into_float32 {
    ($type: ty) => {
        impl IntoFloat32 for $type {
            fn f32(self) -> f32 {
                self.into()
            }
        }
    };
}

macro_rules! impl_into_float64 {
    ($type: ty) => {
        impl IntoFloat64 for $type {
            fn f64(self) -> f64 {
                self.into()
            }
        }
    };
}

impl_into_float32!(u8);
impl_into_float32!(i8);
impl_into_float32!(u16);
impl_into_float32!(i16);

impl_into_float64!(u8);
impl_into_float64!(i8);
impl_into_float64!(u16);
impl_into_float64!(i16);
impl_into_float64!(u32);
impl_into_float64!(i32);
impl_into_float64!(f32);

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
    fn relu(&self) -> Self {
        if *self > 0. {
            *self
        } else {
            0.
        }
    }
    fn from_usize(x: usize) -> Self {
        x as f32
    }
    fn randn() -> Self {
        thread_rng().sample(StandardNormal)
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
    fn relu(&self) -> Self {
        if *self > 0. {
            *self
        } else {
            0.
        }
    }
    fn from_usize(x: usize) -> Self {
        x as f64
    }
    fn randn() -> Self {
        thread_rng().sample(StandardNormal)
    }
}
