use std::{iter::Sum, ops::Index};

use crate::{
    impl_constructors::TensorConstructors, prelude::{dim::Dimension, utils::nd_index}, DataBuffer, DataElement,
    Tensor, TensorBase,
};

pub trait ReduceOps {
    type Output;
    fn sum(&self) -> Self::Output;
    fn mean(&self) -> Self::Output;
}

impl<'a, S, A, Dtype> ReduceOps for TensorBase<S, A>
where
    S: Dimension + 'static,
    A: DataBuffer<Item = Dtype> + Index<usize, Output = Dtype> + 'static,
    Dtype: DataElement + Sum<&'a Dtype> + 'static,
{
    type Output = Tensor<[usize; 0], Dtype>;

    fn sum(&self) -> Self::Output {
        let sum = {
            if self.is_standard_layout() {
                let s = self.as_slice().unwrap().iter().fold(Dtype::zero(), |acc, val| acc + *val);
                Tensor::from_vec(vec![s], [])
            } else {
                let mut s = Dtype::zero();
                let strides = self.default_strides();

                for i in 0..self.len() {
                    let idx = nd_index(i, &strides);
                    s += self[idx]
                }
                Tensor::from_vec(vec![s], []) 
            }
        };
        let mut backops = self.detach_backward_ops();
        if backops.is_some() {
            let lhs_clone = self.clone();
            let out_id = sum.id;
            backops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, Dtype>, &Tensor<[usize; 0], Dtype>) =
                    grad.mr_grad((lhs_clone.id, lhs_clone.dim()), out_id);
                let out_data = out.data[0];
                let t = Tensor::<S, Dtype>::from_elem(lhs_clone.dim(), out_data);
                *input = input.clone() + t;
            });
        }
        *sum.backward_ops.borrow_mut() = backops;
        sum
    }

    fn mean(&self) -> Self::Output {
        let mean = {
            if self.is_standard_layout() {
                let s = self.as_slice().unwrap().iter().fold(Dtype::zero(), |acc, val| acc + *val);
                let n = Dtype::from_usize(self.len());
                Tensor::from_vec(vec![s / n], [])
            } else {
                let mut s = Dtype::zero();
                let n = Dtype::from_usize(self.len());
                let strides = self.default_strides();

                for i in 0..self.len() {
                    let idx = nd_index(i, &strides);
                    s += self[idx]
                }
                Tensor::from_vec(vec![s / n], []) 
            }
        };
        let mut backops = self.detach_backward_ops();
        if backops.is_some() {
            let lhs_clone = self.clone();
            let out_id = mean.id;
            backops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, Dtype>, &Tensor<[usize; 0], Dtype>) =
                    grad.mr_grad((lhs_clone.id, lhs_clone.dim()), out_id);
                let n = Dtype::from_usize(lhs_clone.len());
                let out_data = out.data[0] / n;
                let t = Tensor::<S, Dtype>::from_elem(lhs_clone.dim(), out_data);
                *input = input.clone() + t;
            });
        }
        *mean.backward_ops.borrow_mut() = backops;
        mean
    }
}

#[cfg(test)]
mod tests {
    use crate::{Tensor, impl_reduce_ops::ReduceOps, impl_constructors::TensorConstructors};

    #[test]
    fn test_sum() {
        let t = Tensor::from_vec(vec![1., 2., 3., 4., 5.], [5, 1]).requires_grad(true);
        let sum = t.sum();
        let grad = sum.backward();
        let t_grad = grad.grad(&t);
        let a = Tensor::ones([5, 1]);
        assert_eq!(a, t_grad);
        assert_eq!(sum.data[0], 15.);
    }

    #[test]
    fn test_mean() {
        let t = Tensor::from_vec(vec![1., 2., 3., 4., 5.], [5, 1]).requires_grad(true);
        let mean = t.mean();
        let grad = mean.backward();
        let t_grad = grad.grad(&t);
        let a = Tensor::from_elem([5, 1], 0.2);
        assert_eq!(a, t_grad);
        assert_eq!(mean.data[0], 3.);
    }
}
