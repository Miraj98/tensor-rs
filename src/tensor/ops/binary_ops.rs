use crate::prelude::{
    dim::{DimMax, Dimension},
    TensorBase,
};

impl<L, R, Dtype> std::ops::Add<&TensorBase<R, Dtype>> for &TensorBase<L, Dtype>
where
    R: Dimension,
    L: DimMax<R> + Dimension,
    Dtype: std::ops::Add
{
    type Output = TensorBase<<L as DimMax<R>>::Output, Dtype>;

    fn add(self, rhs: &TensorBase<R, Dtype>) -> Self::Output {
        if self.shape() == rhs.shape() {
        }
        todo!()
    }
}
