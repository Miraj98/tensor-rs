use crate::prelude::TensorBase;

impl<const D: usize, Dtype> std::ops::Add for TensorBase<D, Dtype> {
    type Output = TensorBase<D, Dtype>;

    fn add(self, rhs: Self) -> Self::Output {
        todo!()         
    }
}