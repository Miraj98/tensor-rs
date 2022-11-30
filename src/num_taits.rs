use core::ops::{Add, Mul};

pub trait One: Sized + Mul<Self, Output = Self> + Copy {
    fn one() -> Self;
    fn is_one(&self) -> bool;
}

impl One for usize {
    fn one() -> Self { 1 }
    fn is_one(&self) -> bool { *self == Self::one() }
}
impl One for i32 {
    fn one() -> Self { 1 }
    fn is_one(&self) -> bool { *self == Self::one() }
}
impl One for f32 {
    fn one() -> Self { 1. }
    fn is_one(&self) -> bool { *self == Self::one() }
}
impl One for f64 {
    fn one() -> Self { 1. }
    fn is_one(&self) -> bool { *self == Self::one() }
}

pub trait Zero: Sized + Add<Self, Output = Self> + Copy {
    fn zero() -> Self;
    fn is_zero(&self) -> bool;
}

impl Zero for usize {
    fn zero() -> Self { 0 }
    fn is_zero(&self) -> bool { *self == Self::zero() }
}
impl Zero for i32 {
    fn zero() -> Self { 0 }
    fn is_zero(&self) -> bool { *self == Self::zero() }
}
impl Zero for f32 {
    fn zero() -> Self { 0. }
    fn is_zero(&self) -> bool { *self == Self::zero() }
}
impl Zero for f64 {
    fn zero() -> Self { 0. }
    fn is_zero(&self) -> bool { *self == Self::zero() }
}