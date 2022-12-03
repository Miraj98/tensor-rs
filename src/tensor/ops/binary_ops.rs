use std::iter::zip;

use crate::{
    num_taits::{One, Zero},
    prelude::{
        dim::{DimMax, DimMaxOf, Dimension},
        utils::{merge_backward_ops, reduced_grad}, Tensor, TensorView,
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
    // fn mul(&self, rhs: &Self) -> Self::Output;
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

        let lhs = (self.id, self.dim());
        let rhs = (rhs.id, rhs.dim());
        if backops.is_some() {
            backops.as_mut().unwrap().add_backward_op(move |grad| {
                let (grad_lhs, grad_rhs, grad_out): (
                    &mut Tensor<_, Dtype>,
                    &mut Tensor<_, Dtype>,
                    &Tensor<DimMaxOf<L, R>, Dtype>,
                ) = grad.mmr_grad(lhs.clone(), rhs.clone(), o_id);

                if lhs.1.shape() == rhs.1.shape() {
                    let g_out = grad_out.into_dimensionality::<L>();
                    *grad_lhs = grad_lhs.clone() + g_out;
                    let g_out = grad_out.into_dimensionality::<R>();
                    *grad_rhs = grad_rhs.clone() + g_out;
                } else {
                    *grad_lhs = grad_lhs.clone() + reduced_grad(lhs.1, grad_out);
                    *grad_rhs = grad_rhs.clone() + reduced_grad(rhs.1, grad_out);
                }
            });
        }

        out.put_backward_ops(backops);
        out
    }

    fn sub(&self, rhs: &Tensor<R, Dtype>) -> Self::Output {
        let (out, mut backops) = impl_binary_ops_with_broadcast!(self, rhs, -);
        let o_id = out.id;

        let lhs = (self.id, self.dim());
        let rhs = (rhs.id, rhs.dim());
        if backops.is_some() {
            backops.as_mut().unwrap().add_backward_op(move |grad| {
                let (grad_lhs, grad_rhs, grad_out): (
                    &mut Tensor<_, Dtype>,
                    &mut Tensor<_, Dtype>,
                    &Tensor<DimMaxOf<L, R>, Dtype>,
                ) = grad.mmr_grad(lhs.clone(), rhs.clone(), o_id);
                if lhs.1.shape() == rhs.1.shape() {
                    let g_out = grad_out.into_dimensionality::<L>();
                    *grad_lhs = grad_lhs.clone() + g_out;
                    let g_out = grad_out.into_dimensionality::<R>();
                    *grad_rhs = grad_rhs.clone() - g_out;
                } else {
                    *grad_lhs = grad_lhs.clone() + reduced_grad(lhs.1, grad_out);
                    *grad_rhs = grad_rhs.clone() - reduced_grad(rhs.1, grad_out);
                }
            });
        }

        out.put_backward_ops(backops);
        out
    }

    // fn mul(&self, rhs: &Self) -> Self::Output {
    //     let (out, mut backops) = impl_binary_ops_with_broadcast!(self, rhs, *);
    //     let o_id = out.id;

    //     let lhs = (self.id, self.dim());
    //     let rhs = (rhs.id, rhs.dim());
    //     if backops.is_some() {
    //         backops.as_mut().unwrap().add_backward_op(move |grad| {
    //             let (grad_lhs, grad_rhs, grad_out): (
    //                 &mut Tensor<_, Dtype>,
    //                 &mut Tensor<_, Dtype>,
    //                 &Tensor<_, Dtype>,
    //             ) = grad.mmr_grad(l, r, o_id);
    //             if l.1.is_owned() {
    //                 let _lhs = l.1.take_owned();
    //                 let _rhs = r.1.take_owned();
    //                 *grad_lhs = grad_lhs.clone() + grad_out * &_rhs;
    //                 *grad_rhs = grad_rhs.clone() + grad_out * &_lhs;
    //             } else {
    //                 let _lhs = l.1.take_view();
    //                 let _rhs = r.1.take_view();
    //                 *grad_lhs = grad_lhs.clone() + grad_out * _rhs;
    //                 *grad_rhs = grad_rhs.clone() + grad_out * _lhs;
    //             }
    //         });
    //     }

    //     out.put_backward_ops(backops);
    //     out
    // }
}

#[cfg(test)]
mod tests {
    use crate::prelude::{Tensor, ops::binary_ops::TensorBinaryOps};

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
    fn sub_tensors() {
        let t1 = Tensor::from_vec(vec![5, 1, 3], [1, 3]);
        let t2 = Tensor::from_vec(vec![1, 2, 3], [1, 3]);
        let c = t1.view() - t2.view();
        println!("{:?}", c);
    }
}
