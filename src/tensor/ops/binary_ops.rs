use std::iter::zip;

use crate::{
    num_taits::{One, Zero},
    prelude::{
        dim::{DimMax, DimMaxOf, Dimension},
        impl_constructors::TensorConstructors,
        utils::merge_backward_ops,
        Tensor, TensorView,
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

impl<'a, S, Dtype> std::ops::Sub<&'a Tensor<S, Dtype>> for &'a Tensor<S, Dtype>
where
    S: Dimension,
    Dtype: Copy + std::ops::Sub<Dtype, Output = Dtype>,
{
    type Output = Tensor<S, Dtype>;

    fn sub(self, other: &'a Tensor<S, Dtype>) -> Self::Output {
        assert!(self.shape() == other.shape());
        let mut data = Vec::with_capacity(self.data.len());
        for (a, b) in zip(self.data.iter(), other.data.iter()) {
            data.push(*a - *b);
        }
        let tensor = Tensor::from_vec(data, self.dim.clone());
        tensor
    }
}

impl<'a, S, Dtype> std::ops::Sub<TensorView<'a, S, Dtype>> for TensorView<'a, S, Dtype>
where
    S: Dimension,
    Dtype: Copy + std::ops::Sub<Dtype, Output = Dtype>,
{
    type Output = Tensor<S, Dtype>;

    fn sub(self, other: TensorView<S, Dtype>) -> Self::Output {
        assert!(self.shape() == other.shape());
        let mut data = Vec::with_capacity(self.data.len());
        for (a, b) in zip(self.data.iter(), other.data.iter()) {
            data.push(**a - **b);
        }
        let tensor = Tensor::from_vec(data, self.dim.clone());
        tensor
    }
}

impl_std_binary_ops!(add, Add, +);
impl_std_binary_ops!(mul, Mul, *);
// impl_std_binary_ops!(sub, Sub, -);

macro_rules! impl_binary_ops_with_broadcast {
    ($lhs: ident, $rhs: ident, $symbol: tt) => {
        {
            let out: Tensor<<L as DimMax<R>>::Output, Dtype>;
            let _lhs: (crate::prelude::UniqueId, <L as DimMax<R>>::Output);
            let _rhs: (crate::prelude::UniqueId, <L as DimMax<R>>::Output);
            let backops = merge_backward_ops($lhs, $rhs);

            if $lhs.shape() == $rhs.shape() {
                let l = $lhs.into_dimensionality::<DimMaxOf<L, R>>();
                let r = $rhs.into_dimensionality::<DimMaxOf<L, R>>();
                out = l $symbol r;

                _lhs = (l.id, l.dim());
                _rhs = (r.id, r.dim());
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

                _lhs = (v1.id, v1.dim());
                _rhs = (v2.id, v2.dim());
            }
            (out, backops, _lhs, _rhs)
        }
    }
}

pub trait TensorBinaryOps<Rhs> {
    type Output;
    // fn add(&self, rhs: &Rhs) -> Self::Output;
    // fn sub(&self, rhs: &Rhs) -> Self::Output;
    fn mul(&self, rhs: &Self) -> Self::Output;
}

impl<L, R, Dtype> TensorBinaryOps<Tensor<R, Dtype>> for Tensor<L, Dtype>
where
    R: Dimension + 'static,
    L: Dimension + DimMax<R> + 'static,
    Dtype: One + Zero + 'static 
{
    type Output = Tensor<DimMaxOf<L, R>, Dtype>;

    // fn add(&self, rhs: &Tensor<R, Dtype>) -> Self::Output {
    //     let (out, mut backops, lhs, rhs) = impl_binary_ops_with_broadcast!(self, rhs, +);
    //     let o_id = out.id;

    //     if backops.is_some() {
    //         backops.as_mut().unwrap().add_backward_op(move |grad| {
    //             let (grad_lhs, grad_rhs, grad_out): (
    //                 &mut Tensor<_, Dtype>,
    //                 &mut Tensor<_, Dtype>,
    //                 &Tensor<_, Dtype>,
    //             ) = grad.mmr_grad(lhs, rhs, o_id);
    //             *grad_lhs = grad_lhs.clone() + grad_out;
    //             *grad_rhs = grad_rhs.clone() + grad_out;
    //         });
    //     }

    //     out.put_backward_ops(backops);
    //     out
    // }

    // fn sub(&self, rhs: &Tensor<R, Dtype>) -> Self::Output {
    //     let (out, mut backops, lhs, rhs) = {
    //         let out: Tensor<DimMaxOf<L, R>, Dtype>;
    //         let _lhs: (crate::prelude::UniqueId, DimMaxOf<L, R>);
    //         let _rhs: (crate::prelude::UniqueId,DimMaxOf<L, R>);
    //         let backops = merge_backward_ops(self, rhs);

    //         if self.shape() == rhs.shape() {
    //             let l = self.into_dimensionality::<DimMaxOf<L, R>>();
    //             let r = rhs.into_dimensionality::<DimMaxOf<L, R>>();
    //             out = l - r;

    //             _lhs = (l.id, l.dim());
    //             _rhs = (r.id, r.dim());
    //         } else {
    //             let v1: TensorView<'_, DimMaxOf<L, R>, Dtype>;
    //             let v2: TensorView<'_, DimMaxOf<L, R>, Dtype>;
    //             let dim: DimMaxOf<L, R>;
    //             if self.ndim() >= rhs.ndim() {
    //                 dim = self.dim().into_dimensionality::<DimMaxOf<L, R>>();
    //             } else {
    //                 dim = rhs.dim().into_dimensionality::<DimMaxOf<L, R>>();
    //             }
    //             v1 = self.broadcast(dim.clone());
    //             v2 = rhs.broadcast(v1.dim());
    //             out = v1.clone() - v2.clone();

    //             _lhs = (v1.id, v1.dim());
    //             _rhs = (v2.id, v2.dim());
    //         }
    //         (out, backops, _lhs, _rhs)
    //     };
    //     let o_id = out.id;

    //     if backops.is_some() {
    //         backops.as_mut().unwrap().add_backward_op(move |grad| {
    //             let (grad_lhs, grad_rhs, grad_out): (
    //                 &mut Tensor<_, Dtype>,
    //                 &mut Tensor<_, Dtype>,
    //                 &Tensor<_, Dtype>,
    //             ) = grad.mmr_grad(lhs, rhs, o_id);
    //             let ones = Tensor::ones(grad_out.dim().clone());
    //             // *grad_lhs = grad_lhs.clone() + grad_out;
    //             // *grad_rhs = grad_rhs.clone() - grad_out;
    //         });
    //     }

    //     out.put_backward_ops(backops);
    //     out
    // }

    fn mul(&self, rhs: &Self) -> Self::Output {
        let (out, mut backops, l, r) = impl_binary_ops_with_broadcast!(self, rhs, *);
        let o_id = out.id;

        let rhs_clone = rhs.clone();
        let lhs_clone = self.clone();
        if backops.is_some() {
            backops.as_mut().unwrap().add_backward_op(move |grad| {
                let (grad_lhs, grad_rhs, grad_out): (
                    &mut Tensor<_, Dtype>,
                    &mut Tensor<_, Dtype>,
                    &Tensor<_, Dtype>,
                ) = grad.mmr_grad(l, r, o_id);
                // *grad_lhs = grad_lhs.clone() + grad_out * rhs_clone;
                // *grad_rhs = grad_rhs.clone() + grad_out * lhs_clone;
            });
        }

        out.put_backward_ops(backops);
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::{
        impl_constructors::TensorConstructors, ops::binary_ops::TensorBinaryOps, Tensor,
    };

    // #[test]
    // fn add_tensors() {
    //     let t1 = Tensor::from_vec(vec![1, 2, 3], [1, 3]);
    //     let t2 = Tensor::from_vec(vec![1, 2, 3], [1, 3]);
    //     let c = &t1 + &t2;
    //     println!("{:?}", c);
    // }

    #[test]
    fn sub_tensors() {
        let t1 = Tensor::from_vec(vec![1, 2, 3], [1, 3]);
        let t2 = Tensor::from_vec(vec![1, 2, 3], [1, 3]);
        let c = &t1 - &t2;
        println!("{:?}", c);
    }

    // #[test]
    // fn add_tensors_with_grads() {
    //     let t1 = Tensor::from_vec(vec![1, 2, 3], [1, 3]).requires_grad(true);
    //     let t2 = Tensor::from_vec(vec![1, 2, 3], [1, 3]).requires_grad(true);
    //     let c = t1.add(&t2);
    //     let d = c.add(&t2);
    //     let mut grad = d.detach_backward_ops().unwrap();
    //     let c_id = (d.id, d.dim());
    //     grad.add_backward_op(move |_grad| {
    //         let mut _c: &mut Tensor<_, i32> = _grad.mut_grad_by_id(c_id.0, c_id.1);
    //         *_c = Tensor::ones(c_id.1);
    //     });

    //     let grads = grad.execute();
    //     let t1_grad = grads.grad(&t1);
    //     let t2_grad = grads.grad(&t2);
    //     println!("{:?}", c);
    //     println!("{:?}", d);
    //     println!("{:?}", t1_grad);
    //     println!("{:?}", t2_grad);

    //     // todo!("Compelte the assertions for the correct gradients");
    // }
}
