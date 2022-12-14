use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
    slice::Iter,
};

use crate::utils::unlimited_transmute;

pub type Ix = usize;
pub type Ix0 = [Ix; 0];
pub type Ix1 = [Ix; 1];
pub type Ix2 = [Ix; 2];
pub type Ix3 = [Ix; 3];
pub type Ix4 = [Ix; 4];
pub type Ix5 = [Ix; 5];
pub type Ix6 = [Ix; 6];
pub type Ix7 = [Ix; 7];
pub type DimMaxOf<A, B> = <A as DimMax<B>>::Output;

pub trait DimMax<S>
where
    S: Dimension,
{
    type Output: Dimension;
}

pub trait ShapePattern
{
    type IOutput;
    type UOutput;

    fn ipattern(&self) -> Self::IOutput;
    fn upattern(&self) -> Self::UOutput;
}

pub trait Dimension:
    Clone + Eq + Index<usize, Output = Ix> + IndexMut<usize, Output = Ix> + Debug + ShapePattern
{
    const NDIM: usize;
    type Smaller: Dimension;
    type Larger: Dimension;

    fn into_dimensionality<D2>(&self) -> D2 where D2: Dimension;
    fn expand(self, head: usize) -> Self::Larger;
    fn slice_mut(&mut self) -> &mut [Ix];
    fn get_iter(&self) -> Iter<'_, Ix>;
    fn shape(&self) -> &[Ix];
    fn slice(&self) -> &[Ix];
    fn count(&self) -> usize;
    fn ndim(&self) -> usize;
    fn rev(&self) -> Self;
    fn zeros() -> Self;
    fn ones() -> Self;
}

macro_rules! impl_dimension {
    ($dim: ty, $larger: ty, $smaller: ty, $ndim: expr) => {
        impl Dimension for $dim {
            const NDIM: usize = $ndim;
            type Larger = $larger;
            type Smaller = $smaller;

            fn ndim(&self) -> usize { $ndim }
            fn shape(&self) -> &[Ix] { &self[..] }
            fn slice(&self) -> &[Ix] { &self[..] }
            fn slice_mut(&mut self) -> &mut [Ix] { &mut self[..] }
            fn get_iter(&self) -> Iter<'_, Ix> { self.iter() }
            fn ones() -> Self { [1; $ndim] }
            fn zeros() -> Self { [0; $ndim] }
            fn count(&self) -> usize { self.get_iter().fold(1, |acc, val| acc * *val) }
            fn into_dimensionality<D2>(&self) -> D2 where D2: Dimension {
                unsafe { unlimited_transmute::<Self, D2>(self.clone()) }
            }
            fn rev(&self) -> Self {
                let mut a = self.clone();
                a.reverse();
                a
            }
            fn expand(self, head: usize) -> Self::Larger {
                let mut larger = Self::Larger::zeros();
                let mut dim_iter = larger.slice_mut().iter_mut();
                *dim_iter.next().unwrap() = head;
                for (i, j) in dim_iter.zip(self.slice().iter()) {
                    *i = *j;
                }
                larger
            }
        }
    };
}

pub trait IntoDimension {
    type Dim: Dimension;
    fn into_dimension(self) -> Self::Dim;
}

impl IntoDimension for () {
    type Dim = [usize; 0];
    fn into_dimension(self) -> Self::Dim { [] }
}
impl IntoDimension for usize {
    type Dim = [usize; 1];
    fn into_dimension(self) -> Self::Dim { [self] }
}
impl IntoDimension for (usize, usize) {
    type Dim = [usize; 2];
    fn into_dimension(self) -> Self::Dim { [self.0, self.1] }
}
impl IntoDimension for (usize, usize, usize) {
    type Dim = [usize; 3];
    fn into_dimension(self) -> Self::Dim { [self.0, self.1, self.2] }
}
impl IntoDimension for (usize, usize, usize, usize) {
    type Dim = [usize; 4];
    fn into_dimension(self) -> Self::Dim { [self.0, self.1, self.2, self.3] }
}
impl IntoDimension for (usize, usize, usize, usize, usize) {
    type Dim = [usize; 5];
    fn into_dimension(self) -> Self::Dim { [self.0, self.1, self.2, self.3, self.4] }
}
impl IntoDimension for (usize, usize, usize, usize, usize, usize) {
    type Dim = [usize; 6];
    fn into_dimension(self) -> Self::Dim { [self.0, self.1, self.2, self.3, self.4, self.5] }
}
impl IntoDimension for (usize, usize, usize, usize, usize, usize, usize) {
    type Dim = [usize; 7];
    fn into_dimension(self) -> Self::Dim { [self.0, self.1, self.2, self.3, self.4, self.5, self.6] }
}

macro_rules! impl_into_dimension {
    ($type: ty) => {
        impl IntoDimension for $type {
            type Dim = Self;
            fn into_dimension(self) -> Self::Dim { self }
        }
    }
}

impl_into_dimension!(Ix0);
impl_into_dimension!(Ix1);
impl_into_dimension!(Ix2);
impl_into_dimension!(Ix3);
impl_into_dimension!(Ix4);
impl_into_dimension!(Ix5);
impl_into_dimension!(Ix6);
impl_into_dimension!(Ix7);

impl ShapePattern for [Ix; 0] {
    type IOutput = ();
    type UOutput = ();

    fn ipattern(&self) -> Self::IOutput {
        ()
    }

    fn upattern(&self) -> Self::UOutput {
        ()
    }
}

impl ShapePattern for [Ix; 1] {
    type IOutput = isize;
    type UOutput = Ix;

    fn ipattern(&self) -> Self::IOutput {
        self[0] as isize
    }

    fn upattern(&self) -> Self::UOutput {
        self[0]
    }
}

impl ShapePattern for [Ix; 2] {
    type IOutput = (isize, isize);
    type UOutput = (Ix, Ix);

    fn ipattern(&self) -> Self::IOutput {
        (self[0] as isize, self[1] as isize)
    }

    fn upattern(&self) -> Self::UOutput {
        (self[0], self[1])
    }
}

impl ShapePattern for [Ix; 3] {
    type IOutput = (isize, isize, isize);
    type UOutput = (Ix, Ix, Ix);

    fn ipattern(&self) -> Self::IOutput {
        (self[0] as isize, self[1] as isize, self[2] as isize)
    }

    fn upattern(&self) -> Self::UOutput {
        (self[0], self[1], self[2])
    }
}

impl ShapePattern for [Ix; 4] {
    type IOutput = (isize, isize, isize, isize);
    type UOutput = (Ix, Ix, Ix, Ix);

    fn ipattern(&self) -> Self::IOutput {
        (
            self[0] as isize,
            self[1] as isize,
            self[2] as isize,
            self[3] as isize,
        )
    }

    fn upattern(&self) -> Self::UOutput {
        (self[0], self[1], self[2], self[3])
    }
}

impl ShapePattern for [Ix; 5] {
    type IOutput = (isize, isize, isize, isize, isize);
    type UOutput = (Ix, Ix, Ix, Ix, Ix);

    fn ipattern(&self) -> Self::IOutput {
        (
            self[0] as isize,
            self[1] as isize,
            self[2] as isize,
            self[3] as isize,
            self[4] as isize,
        )
    }

    fn upattern(&self) -> Self::UOutput {
        (self[0], self[1], self[2], self[3], self[4])
    }
}

impl ShapePattern for [Ix; 6] {
    type IOutput = (isize, isize, isize, isize, isize, isize);
    type UOutput = (Ix, Ix, Ix, Ix, Ix, Ix);

    fn ipattern(&self) -> Self::IOutput {
        (
            self[0] as isize,
            self[1] as isize,
            self[2] as isize,
            self[3] as isize,
            self[4] as isize,
            self[5] as isize,
        )
    }

    fn upattern(&self) -> Self::UOutput {
        (self[0], self[1], self[2], self[3], self[4], self[5])
    }
}

impl ShapePattern for [Ix; 7] {
    type IOutput = (isize, isize, isize, isize, isize, isize, isize);
    type UOutput = (Ix, Ix, Ix, Ix, Ix, Ix, Ix);

    fn ipattern(&self) -> Self::IOutput {
        (
            self[0] as isize,
            self[1] as isize,
            self[2] as isize,
            self[3] as isize,
            self[4] as isize,
            self[5] as isize,
            self[6] as isize,
        )
    }

    fn upattern(&self) -> Self::UOutput {
        (self[0], self[1], self[2], self[3], self[4], self[5], self[6])
    }
}



impl_dimension!(Ix0, Ix1, Ix0, 0);
impl_dimension!(Ix1, Ix2, Ix0, 1);
impl_dimension!(Ix2, Ix3, Ix1, 2);
impl_dimension!(Ix3, Ix4, Ix2, 3);
impl_dimension!(Ix4, Ix5, Ix3, 4);
impl_dimension!(Ix5, Ix6, Ix4, 5);
impl_dimension!(Ix6, Ix7, Ix5, 6);
impl_dimension!(Ix7, Ix7, Ix6, 7);

impl<D> DimMax<D> for D
where
    D: Dimension,
{
    type Output = D;
}

macro_rules! impl_broadcast_distinct_fixed {
    ($smaller:ty, $larger:ty) => {
        impl DimMax<$larger> for $smaller {
            type Output = $larger;
        }

        impl DimMax<$smaller> for $larger {
            type Output = $larger;
        }
    };
}

impl_broadcast_distinct_fixed!(Ix0, Ix1);
impl_broadcast_distinct_fixed!(Ix0, Ix2);
impl_broadcast_distinct_fixed!(Ix0, Ix3);
impl_broadcast_distinct_fixed!(Ix0, Ix4);
impl_broadcast_distinct_fixed!(Ix0, Ix5);
impl_broadcast_distinct_fixed!(Ix0, Ix6);
impl_broadcast_distinct_fixed!(Ix1, Ix2);
impl_broadcast_distinct_fixed!(Ix1, Ix3);
impl_broadcast_distinct_fixed!(Ix1, Ix4);
impl_broadcast_distinct_fixed!(Ix1, Ix5);
impl_broadcast_distinct_fixed!(Ix1, Ix6);
impl_broadcast_distinct_fixed!(Ix2, Ix3);
impl_broadcast_distinct_fixed!(Ix2, Ix4);
impl_broadcast_distinct_fixed!(Ix2, Ix5);
impl_broadcast_distinct_fixed!(Ix2, Ix6);
impl_broadcast_distinct_fixed!(Ix3, Ix4);
impl_broadcast_distinct_fixed!(Ix3, Ix5);
impl_broadcast_distinct_fixed!(Ix3, Ix6);
impl_broadcast_distinct_fixed!(Ix4, Ix5);
impl_broadcast_distinct_fixed!(Ix4, Ix6);
impl_broadcast_distinct_fixed!(Ix5, Ix6);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expand_dimensions() {
        let a = [3, 4, 5];
        let b = a.expand(10);
        assert_eq!(b, [10, 3, 4, 5]);
    }
}
