use std::{ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign, Div}, rc::Rc, slice};

pub mod gradient;
pub mod num_taits;
pub mod tensor;
pub mod unique_id;

pub mod prelude {
    pub use crate::gradient::*;
    pub use crate::tensor::*;
    pub use crate::unique_id::*;
}

#[derive(Debug)]
pub struct OwnedData<E>
where
    E: DataElement,
{
    data: Rc<Vec<E>>,
}

impl<E: DataElement> DataBuffer for OwnedData<E> {
    type Item = E;

    fn len(&self) -> usize { self.data.len() }
    fn as_ptr(&self) -> *const Self::Item { self.data.as_ptr() }
}

pub struct ViewData<'a, E>
where
    E: DataElement,
{
    data:Rc<Vec<&'a E>>,
}

impl <'a, E: DataElement> DataBuffer for ViewData<'a, E> {
    type Item = E;

    fn len(&self) -> usize { self.data.len() }
    fn as_ptr(&self) -> *const Self::Item { self.data.as_ptr() as *const Self::Item }
}


pub trait DataBuffer {
    type Item: DataElement;

    fn len(&self) -> usize;
    fn as_ptr(&self) -> *const Self::Item;
    fn as_slice(&self) -> &[Self::Item] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
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
    fn one() -> Self {1.}
    fn is_one(&self) -> bool { *self == Self::one() }
    fn zero() -> Self {0.}
    fn is_zero(&self) -> bool { *self == Self::zero() }
    fn sigmoid(&self) -> Self { 1. / (1. + (-*self).exp()) }
    fn from_usize(x: usize) -> Self { x as f32 }
}

impl DataElement for f64 {
    type Dtype = f64;
    fn one() -> Self {1.}
    fn is_one(&self) -> bool { *self == Self::one() }
    fn zero() -> Self {0.}
    fn is_zero(&self) -> bool { *self == Self::zero() }
    fn sigmoid(&self) -> Self { 1. / (1. + (-*self).exp()) }
    fn from_usize(x: usize) -> Self { x as f64 }
}

