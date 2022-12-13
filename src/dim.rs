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

pub trait Dimension:
    Clone + Eq + Index<usize, Output = usize> + IndexMut<usize, Output = usize> + Debug
{
    const NDIM: usize;
    type Smaller: Dimension;
    type Larger: Dimension;

    fn ndim(&self) -> usize;
    fn shape(&self) -> &[usize];
    fn slice(&self) -> &[usize];
    fn slice_mut(&mut self) -> &mut [usize];
    fn rev(&self) -> Self;
    fn count(&self) -> usize;
    fn into_dimensionality<D2>(&self) -> D2
    where
        D2: Dimension;
    fn get_iter(&self) -> Iter<'_, usize>;
    fn ones() -> Self;
    fn zeros() -> Self;
}

macro_rules! impl_dimension {
    ($dim: ty, $larger: ty, $smaller: ty, $ndim: expr) => {
        impl Dimension for $dim {
            const NDIM: usize = $ndim;
            type Larger = $larger;
            type Smaller = $smaller;

            fn ndim(&self) -> usize {
                $ndim
            }

            fn shape(&self) -> &[usize] {
                &self[..]
            }

            fn slice(&self) -> &[usize] {
                &self[..]
            }

            fn slice_mut(&mut self) -> &mut [usize] {
                &mut self[..]
            }

            fn rev(&self) -> Self {
                let mut a = self.clone();
                a.reverse();
                a
            }

            fn count(&self) -> usize {
                self.get_iter().fold(1, |acc, val| acc * *val)
            }

            fn into_dimensionality<D2>(&self) -> D2
            where
                D2: Dimension,
            {
                unsafe { unlimited_transmute::<Self, D2>(self.clone()) }
            }

            fn get_iter(&self) -> Iter<'_, usize> {
                self.iter()
            }

            fn ones() -> Self {
                [1; $ndim]
            }

            fn zeros() -> Self {
                [0; $ndim]
            }
        }
    };
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
