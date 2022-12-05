use crate::{
    num_taits::{One, Zero},
    prelude::{dim::Dimension, Tensor, impl_constructors::TensorConstructors},
};
use std::ops::{Add, Div};

pub trait ReduceOps {
    type Output;
    fn sum(&self) -> Self::Output;
    fn mean(&self) -> Self::Output;
}

impl<S, Dtype> ReduceOps for Tensor<S, Dtype>
where
    S: Dimension + 'static,
    Dtype: PartialEq
        + Copy
        + Zero
        + One
        + Add<Dtype, Output = Dtype>
        + Div<f32, Output = Dtype>
        + 'static,
{
    type Output = Tensor<[usize; 0], Dtype>;

    fn sum(&self) -> Self::Output {
        let sum = self._sum();
        let backops = self.detach_backward_ops();
        if backops.is_some() {
            let mut backops = backops.unwrap();
            let lhs_clone = self.clone();
            let out_id = sum.id;
            backops.add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, Dtype>, &Tensor<_, Dtype>) =
                    grad.mr_grad((lhs_clone.id, lhs_clone.dim()), out_id);
                let t = Tensor::<S, Dtype>::ones(lhs_clone.dim());
                let out_data = out.data[0];
                *input = input.clone() + t * out_data;
            });
        }
        sum
    }

    fn mean(&self) -> Self::Output {
        let mean = self._mean();
        let backops = self.detach_backward_ops();
        if backops.is_some() {
            let mut backops = backops.unwrap();
            let lhs_clone = self.clone();
            let out_id = mean.id;
            backops.add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, Dtype>, &Tensor<_, Dtype>) =
                    grad.mr_grad((lhs_clone.id, lhs_clone.dim()), out_id);
                let t = Tensor::<S, Dtype>::ones(lhs_clone.dim());
                let out_data = out.data[0] / lhs_clone.len() as f32;
                *input = input.clone() + t * out_data;
            });
        }
        mean
    }
}
