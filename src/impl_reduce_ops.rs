use crate::{
    dim::Dimension, impl_constructors::TensorConstructors, utils::nd_index, DataBuffer,
    DataElement, Tensor, TensorBase,
};

pub trait ReduceOps {
    type Output;
    fn sum(&self) -> Self::Output;
    fn mean(&self) -> Self::Output;
}

impl<'a, S, A, E> ReduceOps for TensorBase<S, A>
where
    S: Dimension + 'static,
    A: DataBuffer<Item = E> + 'static,
    E: DataElement + 'static,
{
    type Output = Tensor<[usize; 0], E>;

    fn sum(&self) -> Self::Output {
        let sum = {
            if self.is_standard_layout() {
                let s = self
                    .as_slice()
                    .unwrap()
                    .iter()
                    .fold(E::zero(), |acc, val| acc + *val);
                Tensor::from_vec(vec![s], [])
            } else {
                let mut s = E::zero();
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
                let (input, out): (&mut Tensor<_, E>, &Tensor<[usize; 0], E>) =
                    grad.mr_grad((lhs_clone.id, lhs_clone.dim()), out_id);
                let out_data = unsafe { out.ptr.as_ptr().read() };
                let t = Tensor::<S, E>::from_elem(lhs_clone.dim(), out_data);
                *input = input.clone() + t;
            });
        }
        *sum.backward_ops.borrow_mut() = backops;
        sum
    }

    fn mean(&self) -> Self::Output {
        let mean = {
            if self.is_standard_layout() {
                let s = self
                    .as_slice()
                    .unwrap()
                    .iter()
                    .fold(E::zero(), |acc, val| acc + *val);
                let n = E::from_usize(self.len());
                Tensor::from_vec(vec![s / n], [])
            } else {
                let mut s = E::zero();
                let n = E::from_usize(self.len());
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
                let (input, out): (&mut Tensor<_, E>, &Tensor<[usize; 0], E>) =
                    grad.mr_grad((lhs_clone.id, lhs_clone.dim()), out_id);
                let n = E::from_usize(lhs_clone.len());
                let out_data = unsafe { out.ptr.as_ptr().read() } / n;
                let t = Tensor::<S, E>::from_elem(lhs_clone.dim(), out_data);
                *input = input.clone() + t;
            });
        }
        *mean.backward_ops.borrow_mut() = backops;
        mean
    }
}

#[cfg(test)]
mod tests {
    use crate::{impl_constructors::TensorConstructors, impl_reduce_ops::ReduceOps, Tensor};

    #[test]
    fn test_sum() {
        let t = Tensor::from_vec(vec![1., 2., 3., 4., 5.], [5, 1]).requires_grad(true);
        let sum = t.sum();
        let grad = sum.backward();
        let t_grad = grad.grad(&t);
        let a = Tensor::ones([5, 1]);
        let sum = unsafe { sum.ptr.as_ptr().read() };
        assert_eq!(a, t_grad);
        assert_eq!(sum, 15.);
    }

    #[test]
    fn test_mean() {
        let t = Tensor::from_vec(vec![1., 2., 3., 4., 5.], [5, 1]).requires_grad(true);
        let mean = t.mean();
        let grad = mean.backward();
        let t_grad = grad.grad(&t);
        let a = Tensor::from_elem([5, 1], 0.2);
        assert_eq!(a, t_grad);
        let mean = unsafe { mean.ptr.as_ptr().read() };
        assert_eq!(mean, 3.);
    }
}
