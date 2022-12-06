use matrixmultiply::sgemm;

use crate::{
    prelude::{
        dim::{DimMax, DimMaxOf, Dimension},
        utils::{merge_backward_ops, nd_index, reduced_grad, generate_strides},
    },
    unique_id::unique_id,
    DataBuffer, DataElement, OwnedData, Tensor, TensorView, TensorBase,
};
use std::{
    cell::RefCell,
    ops::{Add, Div, Index, Mul, Sub},
};

macro_rules! impl_binary_ops {
    ($math: ident, $trait: ident) => {
        impl<S, A, B, E> $trait<TensorBase<S, A>> for TensorBase<S, B>
        where
            S: Dimension,
            E: DataElement,
            A: DataBuffer<Item = E> + Index<usize, Output = E>,
            B: DataBuffer<Item = E> + Index<usize, Output = E>,
        {
            type Output = Tensor<S, E>;

            fn $math(self, rhs: TensorBase<S, A>) -> Self::Output {
                assert_eq!(self.shape(), rhs.shape());
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                for i in 0..self.len() {
                    let idx = nd_index(i, &self.dim, &strides);
                    out_vec[i] = self[idx.clone()].$math(rhs[idx]);
                }

                TensorBase {
                    id: unique_id(),
                    data: OwnedData::new(out_vec),
                    dim: self.dim.clone(),
                    strides,
                    is_leaf: false,
                    requires_grad: self.requires_grad,
                    backward_ops: RefCell::new(None),
                }
            }
        }

        impl<'a, S, A, B, E> $trait<&'a TensorBase<S, A>> for TensorBase<S, B>
        where
            S: Dimension,
            E: DataElement,
            A: DataBuffer<Item = E> + Index<usize, Output = E>,
            B: DataBuffer<Item = E> + Index<usize, Output = E>,
        {
            type Output = Tensor<S, E>;

            fn $math(self, rhs: &'a TensorBase<S, A>) -> Self::Output {
                assert_eq!(self.shape(), rhs.shape());
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                for i in 0..self.len() {
                    let idx = nd_index(i, &self.dim, &strides);
                    out_vec[i] = self[idx.clone()].$math(rhs[idx]);
                }

                TensorBase {
                    id: unique_id(),
                    data: OwnedData::new(out_vec),
                    dim: self.dim.clone(),
                    strides,
                    is_leaf: false,
                    requires_grad: self.requires_grad,
                    backward_ops: RefCell::new(None),
                }
            }
        }

        impl<'a, S, A, B, E> $trait<TensorBase<S, A>> for &'a TensorBase<S, B>
        where
            S: Dimension,
            E: DataElement,
            A: DataBuffer<Item = E> + Index<usize, Output = E>,
            B: DataBuffer<Item = E> + Index<usize, Output = E>,
        {
            type Output = Tensor<S, E>;

            fn $math(self, rhs: TensorBase<S, A>) -> Self::Output {
                assert_eq!(self.shape(), rhs.shape());
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                for i in 0..self.len() {
                    let idx = nd_index(i, &self.dim, &strides);
                    out_vec[i] = self[idx.clone()].$math(rhs[idx]);
                }

                TensorBase {
                    id: unique_id(),
                    data: OwnedData::new(out_vec),
                    dim: self.dim.clone(),
                    strides,
                    is_leaf: false,
                    requires_grad: self.requires_grad,
                    backward_ops: RefCell::new(None),
                }
            }
        }

        impl<'a, S, A, B, E> $trait<&'a TensorBase<S, A>> for &'a TensorBase<S, B>
        where
            S: Dimension,
            E: DataElement,
            A: DataBuffer<Item = E> + Index<usize, Output = E>,
            B: DataBuffer<Item = E> + Index<usize, Output = E>,
        {
            type Output = Tensor<S, E>;

            fn $math(self, rhs: &'a TensorBase<S, A>) -> Self::Output {
                assert_eq!(self.shape(), rhs.shape());
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                for i in 0..self.len() {
                    let idx = nd_index(i, &self.dim, &strides);
                    out_vec[i] = self[idx.clone()].$math(rhs[idx]);
                }

                TensorBase {
                    id: unique_id(),
                    data: OwnedData::new(out_vec),
                    dim: self.dim.clone(),
                    strides,
                    is_leaf: false,
                    requires_grad: self.requires_grad,
                    backward_ops: RefCell::new(None),
                }
            }
        }

        impl<S, A> $trait<f32> for TensorBase<S, A>
        where
            S: Dimension,
            A: DataBuffer<Item = f32> + Index<usize, Output = f32>,
        {
            type Output = Tensor<S, f32>;

            fn $math(self, rhs: f32) -> Self::Output {
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                for i in 0..self.len() {
                    let idx = nd_index(i, &self.dim, &strides);
                    out_vec[i] = self[idx].$math(rhs);
                }

                TensorBase {
                    id: unique_id(),
                    data: OwnedData::new(out_vec),
                    dim: self.dim.clone(),
                    strides,
                    is_leaf: false,
                    requires_grad: self.requires_grad,
                    backward_ops: RefCell::new(None),
                }
            }
        }

        impl<S, A> $trait<f64> for TensorBase<S, A>
        where
            S: Dimension,
            A: DataBuffer<Item = f64> + Index<usize, Output = f64>,
        {
            type Output = Tensor<S, f64>;

            fn $math(self, rhs: f64) -> Self::Output {
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                for i in 0..self.len() {
                    let idx = nd_index(i, &self.dim, &strides);
                    out_vec[i] = self[idx].$math(rhs);
                }

                TensorBase {
                    id: unique_id(),
                    data: OwnedData::new(out_vec),
                    dim: self.dim.clone(),
                    strides,
                    is_leaf: false,
                    requires_grad: self.requires_grad,
                    backward_ops: RefCell::new(None),
                }
            }
        }

        impl<'a, S, A> $trait<f32> for &'a TensorBase<S, A>
        where
            S: Dimension,
            A: DataBuffer<Item = f32> + Index<usize, Output = f32>,
        {
            type Output = Tensor<S, f32>;

            fn $math(self, rhs: f32) -> Self::Output {
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                for i in 0..self.len() {
                    let idx = nd_index(i, &self.dim, &strides);
                    out_vec[i] = self[idx].$math(rhs);
                }

                TensorBase {
                    id: unique_id(),
                    data: OwnedData::new(out_vec),
                    dim: self.dim.clone(),
                    strides,
                    is_leaf: false,
                    requires_grad: self.requires_grad,
                    backward_ops: RefCell::new(None),
                }
            }
        }

        impl<'a, S, A> $trait<f64> for &'a TensorBase<S, A>
        where
            S: Dimension,
            A: DataBuffer<Item = f64> + Index<usize, Output = f64>,
        {
            type Output = Tensor<S, f64>;

            fn $math(self, rhs: f64) -> Self::Output {
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                for i in 0..self.len() {
                    let idx = nd_index(i, &self.dim, &strides);
                    out_vec[i] = self[idx].$math(rhs);
                }

                TensorBase {
                    id: unique_id(),
                    data: OwnedData::new(out_vec),
                    dim: self.dim.clone(),
                    strides,
                    is_leaf: false,
                    requires_grad: self.requires_grad,
                    backward_ops: RefCell::new(None),
                }
            }
        }
    };
}

impl_binary_ops!(add, Add);
impl_binary_ops!(mul, Mul);
impl_binary_ops!(sub, Sub);
impl_binary_ops!(div, Div);

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
    Dtype: DataElement + 'static,
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

                    let lhs_local_grad = rhs_broadcast * grad_out;
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

impl<A> TensorBase<[usize; 2], A>
where
    A: DataBuffer<Item = f32>,
{
    fn dot<B>(&self, rhs: &TensorBase<[usize; 2], B>) -> Tensor<[usize; 2], A::Item>
    where
        B: DataBuffer<Item = f32>,
    {
        let a = self.shape()[1];
        let b = self.shape()[1];
        println!("a: {}, b: {}", a, b);
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
