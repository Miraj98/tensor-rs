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
        let mut backops = self.detach_backward_ops();
        if backops.is_some() {
            let lhs_clone = self.clone();
            let out_id = sum.id;
            backops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, Dtype>, &Tensor<[usize; 0], Dtype>) =
                    grad.mr_grad((lhs_clone.id, lhs_clone.dim()), out_id);
                let t = Tensor::<S, Dtype>::ones(lhs_clone.dim());
                let out_data = out.data[0];
                *input = input.clone() + t * out_data;
            });
        }
        *sum.backward_ops.borrow_mut() = backops;
        sum
    }

    fn mean(&self) -> Self::Output {
        let mean = self._mean();
        let mut backops = self.detach_backward_ops();
        if backops.is_some() {
            let lhs_clone = self.clone();
            let out_id = mean.id;
            backops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, Dtype>, &Tensor<[usize; 0], Dtype>) =
                    grad.mr_grad((lhs_clone.id, lhs_clone.dim()), out_id);
                let t = Tensor::<S, Dtype>::ones(lhs_clone.dim());
                let out_data = out.data[0] / lhs_clone.len() as f32;
                *input = input.clone() + t * out_data;
            });
        }
        *mean.backward_ops.borrow_mut() = backops;
        mean
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::Tensor;
    use crate::tensor::ops::reduce_ops::ReduceOps;

    #[test]
    fn test_sum() {
        let t = Tensor::from_vec(vec![1., 2., 3., 4., 5.], [5, 1]).requires_grad(true);
        let sum = t.sum();
        let grad = sum.backward();
        let t_grad = grad.grad(&t);
        println!("{:?}", t_grad);
        assert_eq!(sum.data[0], 15.);
    }

    #[test]
    fn test_mean() {
        let t = Tensor::from_vec(vec![1., 2., 3., 4., 5.], [5, 1]).requires_grad(true);
        let mean = t.mean();
        let grad = mean.backward();
        let t_grad = grad.grad(&t);
        println!("{:?}", t_grad);
        assert_eq!(mean.data[0], 3.);
    }
}
