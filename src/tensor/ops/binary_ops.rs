use std::iter::zip;

use matrixmultiply::sgemm;

use crate::{
    num_taits::{One, Zero},
    prelude::{
        dim::{DimMax, DimMaxOf, Dimension},
        utils::{merge_backward_ops, reduced_grad, generate_strides},
        Tensor, TensorView, TensorBase, Data,
    },
};

macro_rules! impl_std_binary_ops {
    ($op: ident, $op_trait: ident, $op_symbol: tt) => {
        impl<'a, S, Dtype> std::ops::$op_trait<&'a Tensor<S, Dtype>> for &'a Tensor<S, Dtype>
        where
            S: Dimension,
            Dtype: Copy + std::ops::$op_trait<Dtype, Output = Dtype>,
        {
            type Output = Tensor<S, Dtype>;

            fn $op(self, other: &'a Tensor<S, Dtype>) -> Self::Output {
                assert!(self.shape() == other.shape());
                let mut data = Vec::with_capacity(self.data.len());
                for (a, b) in zip(self.data.iter(), other.data.iter()) {
                    data.push(*a $op_symbol *b);
                }
                let tensor = Tensor::from_vec(data, self.dim.clone());
                tensor
            }
        }

        impl<'a, S, Dtype> std::ops::$op_trait<&'a Tensor<S, Dtype>> for Tensor<S, Dtype>
        where
            S: Dimension,
            Dtype: Copy + std::ops::$op_trait<Dtype, Output = Dtype>,
        {
            type Output = Tensor<S, Dtype>;

            fn $op(self, other: &'a Tensor<S, Dtype>) -> Self::Output {
                assert!(self.shape() == other.shape());
                let mut data = Vec::with_capacity(self.data.len());
                for (a, b) in zip(self.data.iter(), other.data.iter()) {
                    data.push(*a $op_symbol *b);
                }
                let tensor = Tensor::from_vec(data, self.dim.clone());
                tensor
            }
        }

        impl<S, Dtype> std::ops::$op_trait<Tensor<S, Dtype>> for Tensor<S, Dtype>
        where
            S: Dimension,
            Dtype: Copy + std::ops::$op_trait<Dtype, Output = Dtype>,
        {
            type Output = Tensor<S, Dtype>;

            fn $op(self, other: Tensor<S, Dtype>) -> Self::Output {
                assert!(self.shape() == other.shape());
                let mut data = Vec::with_capacity(self.data.len());
                for (a, b) in zip(self.data.iter(), other.data.iter()) {
                    data.push(*a $op_symbol *b);
                }
                let tensor = Tensor::from_vec(data, self.dim.clone());
                tensor
            }
        }

        impl<'a, S, Dtype> std::ops::$op_trait<TensorView<'a, S, Dtype>> for TensorView<'a, S, Dtype>
        where
            S: Dimension,
            Dtype: Copy + std::ops::$op_trait<Dtype, Output = Dtype>,
        {
            type Output = Tensor<S, Dtype>;

            fn $op(self, other: TensorView<S, Dtype>) -> Self::Output {
                assert!(self.shape() == other.shape());
                let mut data = Vec::with_capacity(self.data.len());
                for (a, b) in zip(self.data.iter(), other.data.iter()) {
                    data.push(**a $op_symbol **b);
                }
                let tensor = Tensor::from_vec(data, self.dim.clone());
                tensor
            }
        }

        impl<'a, S, Dtype> std::ops::$op_trait<TensorView<'a, S, Dtype>> for &Tensor<S, Dtype>
        where
            S: Dimension,
            Dtype: Copy + std::ops::$op_trait<Dtype, Output = Dtype>,
        {
            type Output = Tensor<S, Dtype>;

            fn $op(self, other: TensorView<S, Dtype>) -> Self::Output {
                assert!(self.shape() == other.shape());
                let mut data = Vec::with_capacity(self.data.len());
                for (a, b) in zip(self.data.iter(), other.data.iter()) {
                    data.push(*a $op_symbol **b);
                }
                let tensor = Tensor::from_vec(data, self.dim.clone());
                tensor
            }
        }
    };
}

impl_std_binary_ops!(add, Add, +);
impl_std_binary_ops!(mul, Mul, *);
impl_std_binary_ops!(sub, Sub, -);

macro_rules! impl_binary_ops_with_broadcast {
    ($lhs: ident, $rhs: ident, $symbol: tt) => {
        {
            let out: Tensor<<L as DimMax<R>>::Output, Dtype>;
            let backops = merge_backward_ops($lhs, $rhs);

            if $lhs.shape() == $rhs.shape() {
                let l = $lhs.into_dimensionality::<DimMaxOf<L, R>>();
                let r = $rhs.into_dimensionality::<DimMaxOf<L, R>>();
                out = l $symbol r;
            } else {
                let v1: TensorView<'_, DimMaxOf<L, R>, Dtype>;
                let v2: TensorView<'_, DimMaxOf<L, R>, Dtype>;
                let dim: DimMaxOf<L, R>;
                if $lhs.ndim() >= $rhs.ndim() {
                    dim = $lhs.dim().into_dimensionality::<DimMaxOf<L, R>>();
                } else {
                    dim = $rhs.dim().into_dimensionality::<DimMaxOf<L, R>>();
                }
                v1 = $lhs.broadcast(dim.clone());
                v2 = $rhs.broadcast(v1.dim());
                out = v1.clone() $symbol v2.clone();
            }
            (out, backops)
        }
    }
}

pub trait TensorBinaryOps<Rhs> {
    type Output;
    fn add(&self, rhs: &Rhs) -> Self::Output;
    fn sub(&self, rhs: &Rhs) -> Self::Output;
    fn mul(&self, rhs: &Rhs) -> Self::Output;
}

pub trait Matmul<Rhs> {
    type Output;
    fn matmul(&self, rhs: &Rhs) -> Self::Output;
}

impl<L, R, Dtype> TensorBinaryOps<Tensor<R, Dtype>> for Tensor<L, Dtype>
where
    R: Dimension + 'static,
    L: Dimension + DimMax<R> + 'static,
    Dtype: One + Zero + 'static,
{
    type Output = Tensor<DimMaxOf<L, R>, Dtype>;

    fn add(&self, rhs: &Tensor<R, Dtype>) -> Self::Output {
        let (out, mut backops) = impl_binary_ops_with_broadcast!(self, rhs, +);
        let o_id = out.id;

        let l = (self.id, self.dim());
        let r = (rhs.id, rhs.dim());
        if backops.is_some() {
            backops.as_mut().unwrap().add_backward_op(move |grad| {
                let (grad_lhs, grad_rhs, grad_out): (
                    &mut Tensor<_, Dtype>,
                    &mut Tensor<_, Dtype>,
                    &Tensor<DimMaxOf<L, R>, Dtype>,
                ) = grad.mmr_grad(l.clone(), r.clone(), o_id);

                if l.1.shape() == r.1.shape() {
                    let g_out = grad_out.into_dimensionality::<L>();
                    *grad_lhs = grad_lhs.clone() + g_out;
                    let g_out = grad_out.into_dimensionality::<R>();
                    *grad_rhs = grad_rhs.clone() + g_out;
                } else {
                    *grad_lhs = grad_lhs.clone() + reduced_grad(l.1, grad_out);
                    *grad_rhs = grad_rhs.clone() + reduced_grad(r.1, grad_out);
                }
            });
        }

        out.put_backward_ops(backops);
        out
    }

    fn sub(&self, rhs: &Tensor<R, Dtype>) -> Self::Output {
        let (out, mut backops) = impl_binary_ops_with_broadcast!(self, rhs, -);
        let o_id = out.id;

        let l = (self.id, self.dim());
        let r = (rhs.id, rhs.dim());
        if backops.is_some() {
            backops.as_mut().unwrap().add_backward_op(move |grad| {
                let (grad_lhs, grad_rhs, grad_out): (
                    &mut Tensor<_, Dtype>,
                    &mut Tensor<_, Dtype>,
                    &Tensor<DimMaxOf<L, R>, Dtype>,
                ) = grad.mmr_grad(l.clone(), r.clone(), o_id);
                if l.1.shape() == r.1.shape() {
                    let g_out = grad_out.into_dimensionality::<L>();
                    *grad_lhs = grad_lhs.clone() + g_out;
                    let g_out = grad_out.into_dimensionality::<R>();
                    *grad_rhs = grad_rhs.clone() - g_out;
                } else {
                    *grad_lhs = grad_lhs.clone() + reduced_grad(l.1, grad_out);
                    *grad_rhs = grad_rhs.clone() - reduced_grad(r.1, grad_out);
                }
            });
        }

        out.put_backward_ops(backops);
        out
    }

    fn mul(&self, rhs: &Tensor<R, Dtype>) -> Self::Output {
        let (out, mut backops) = impl_binary_ops_with_broadcast!(self, rhs, *);
        let o_id = out.id;

        let l = (self.id, self.dim());
        let r = (rhs.id, rhs.dim());
        let lhs_clone = self.clone();
        let rhs_clone = rhs.clone();
        if backops.is_some() {
            backops.as_mut().unwrap().add_backward_op(move |grad| {
                let (grad_lhs, grad_rhs, grad_out): (
                    &mut Tensor<_, Dtype>,
                    &mut Tensor<_, Dtype>,
                    &Tensor<DimMaxOf<L, R>, Dtype>,
                ) = grad.mmr_grad(l.clone(), r.clone(), o_id);
                if l.1.shape() == r.1.shape() {
                    let g_out = grad_out.into_dimensionality::<L>();
                    let rhs_clone = rhs_clone.into_dimensionality::<L>();
                    *grad_lhs = grad_lhs.clone() + g_out * rhs_clone;
                    let g_out = grad_out.into_dimensionality::<R>();
                    let lhs_clone = lhs_clone.into_dimensionality::<R>();
                    *grad_rhs = grad_rhs.clone() + g_out * lhs_clone;
                } else {
                    let rhs_broadcast = rhs_clone.broadcast(grad_out.dim());
                    let lhs_broadcast = lhs_clone.broadcast(grad_out.dim());

                    let lhs_local_grad = grad_out * rhs_broadcast;
                    *grad_lhs = grad_lhs.clone() + reduced_grad(l.1, &lhs_local_grad);
                    let rhs_local_grad = grad_out * lhs_broadcast;
                    *grad_rhs = grad_rhs.clone() + reduced_grad(r.1, &rhs_local_grad);
                }
            });
        }

        out.put_backward_ops(backops);
        out
    }
}

impl<A> Matmul<TensorBase<[usize; 2], A>> for TensorBase<[usize; 2], A> where A: Data<Dtype = f32> {
    type Output = Tensor<[usize; 2], A::Dtype>;

    fn matmul(&self, rhs: &TensorBase<[usize; 2], A>) -> Self::Output {
        assert!(self.shape()[1] == rhs.shape()[0]);
        let out_dim = [self.shape()[0], rhs.shape()[1]];
        let out_strides = generate_strides(&out_dim);
        let mut o = vec![0.; out_dim[0] * out_dim[1]];

        unsafe {
            sgemm(
                self.shape()[0],
                self.shape()[1],
                rhs.shape()[1],
                1.,
                self.data.as_ptr(),
                self.strides()[0] as isize,
                self.strides()[1] as isize,
                rhs.data.as_ptr(),
                rhs.strides()[0] as isize,
                rhs.strides()[1] as isize,
                0.,
                o.as_mut_ptr(),
                out_strides[0] as isize,
                out_strides[1] as isize,
            )
        }

        Tensor::from_vec(o, out_dim)
    }
}

impl<A> Matmul<TensorBase<[usize; 2], A>> for &TensorBase<[usize; 2], A> where A: Data<Dtype = f32> {
    type Output = Tensor<[usize; 2], A::Dtype>;

    fn matmul(&self, rhs: &TensorBase<[usize; 2], A>) -> Self::Output {
        assert!(self.shape()[1] == rhs.shape()[0]);
        let out_dim = [self.shape()[0], rhs.shape()[1]];
        let out_strides = generate_strides(&out_dim);
        let mut o = vec![0.; out_dim[0] * out_dim[1]];

        unsafe {
            sgemm(
                self.shape()[0],
                self.shape()[1],
                rhs.shape()[1],
                1.,
                self.data.as_ptr(),
                self.strides()[0] as isize,
                self.strides()[1] as isize,
                rhs.data.as_ptr(),
                rhs.strides()[0] as isize,
                rhs.strides()[1] as isize,
                0.,
                o.as_mut_ptr(),
                out_strides[0] as isize,
                out_strides[1] as isize,
            )
        }

        Tensor::from_vec(o, out_dim)
    }
}


#[cfg(test)]
mod tests {
    use crate::prelude::{ops::binary_ops::TensorBinaryOps, Tensor};

    #[test]
    fn add_tensors() {
        let t1 = Tensor::from_vec(vec![5, 1, 3], [3, 1]).requires_grad(true);
        let t2 = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], [2, 1, 3]).requires_grad(true);
        let c = t1.add(&t2);
        let grads = c.backward();

        println!("t1 grad\n{:#?}\n\n", grads.grad(&t1));
        println!("t2 grad\n{:#?}\n\n", grads.grad(&t2));
    }

    #[test]
    fn mul_tensors() {
        let t1 = Tensor::from_vec(vec![5, 1, 3], [3, 1]).requires_grad(true);
        let t2 = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], [2, 1, 3]).requires_grad(true);
        let c = t1.mul(&t2);
        let grads = c.backward();

        println!("t1 grad\n{:#?}\n\n", grads.grad(&t1));
        println!("t2 grad\n{:#?}\n\n", grads.grad(&t2));
    }

    #[test]
    fn sub_tensors() {
        let t1 = Tensor::from_vec(vec![5, 1, 3], [1, 3]);
        let t2 = Tensor::from_vec(vec![1, 2, 3], [1, 3]);
        let c = t1.view() - t2.view();
        println!("{:?}", c);
    }
}
