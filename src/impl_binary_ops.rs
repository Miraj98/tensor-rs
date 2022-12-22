use crate::{
    dim::{DimMax, DimMaxOf, Dimension},
    utils::{merge_backward_ops, nd_index, reduced_grad, vec_ptr_offset},
    DataBuffer, DataElement, Tensor, TensorBase, TensorView,
};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};

macro_rules! impl_binary_ops {
    ($math: ident, $trait: ident) => {
        impl<S, A, B, E> $trait<TensorBase<S, A>> for TensorBase<S, B>
        where
            S: Dimension,
            E: DataElement,
            A: DataBuffer<Item = E>,
            B: DataBuffer<Item = E>,
        {
            type Output = Tensor<S, E>;

            #[inline]
            fn $math(self, rhs: TensorBase<S, A>) -> Self::Output {
                assert_eq!(self.len(), rhs.len());
                let s_ptr = self.ptr.as_ptr();
                let o_ptr = rhs.ptr.as_ptr();
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                if (self.is_standard_layout() && rhs.is_standard_layout()) {
                    unsafe {
                        for i in 0..self.len() {
                            out_vec.push(s_ptr.add(i).read().$math(o_ptr.add(i).read()));
                        }
                    }
                } else {
                    for i in 0..self.len() {
                        let idx = nd_index(i, &strides);
                        out_vec.push(self[idx.clone()].$math(rhs[idx]));
                    }
                }
                Tensor::from_vec(out_vec, self.dim.clone()).leaf(false)
            }
        }

        impl<'a, S, A, B, E> $trait<&'a TensorBase<S, A>> for TensorBase<S, B>
        where
            S: Dimension,
            E: DataElement,
            A: DataBuffer<Item = E>,
            B: DataBuffer<Item = E>,
        {
            type Output = Tensor<S, E>;

            #[inline]
            fn $math(self, rhs: &'a TensorBase<S, A>) -> Self::Output {
                assert_eq!(self.len(), rhs.len());
                let s_ptr = self.ptr.as_ptr();
                let o_ptr = rhs.ptr.as_ptr();
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                if (self.is_standard_layout() && rhs.is_standard_layout()) {
                    unsafe {
                        for i in 0..self.len() {
                            out_vec.push(s_ptr.add(i).read().$math(o_ptr.add(i).read()));
                        }
                    }
                } else {
                    for i in 0..self.len() {
                        let idx = nd_index(i, &strides);
                        out_vec.push(self[idx.clone()].$math(rhs[idx]));
                    }
                }
                Tensor::from_vec(out_vec, self.dim.clone()).leaf(false)
            }
        }

        impl<'a, S, A, B, E> $trait<TensorBase<S, A>> for &'a TensorBase<S, B>
        where
            S: Dimension,
            E: DataElement,
            A: DataBuffer<Item = E>,
            B: DataBuffer<Item = E>,
        {
            type Output = Tensor<S, E>;

            #[inline]
            fn $math(self, rhs: TensorBase<S, A>) -> Self::Output {
                assert_eq!(self.len(), rhs.len());
                let s_ptr = self.ptr.as_ptr();
                let o_ptr = rhs.ptr.as_ptr();
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                if (self.is_standard_layout() && rhs.is_standard_layout()) {
                    unsafe {
                        for i in 0..self.len() {
                            out_vec.push(s_ptr.add(i).read().$math(o_ptr.add(i).read()));
                        }
                    }
                } else {
                    for i in 0..self.len() {
                        let idx = nd_index(i, &strides);
                        out_vec.push(self[idx.clone()].$math(rhs[idx]));
                    }
                }
                Tensor::from_vec(out_vec, self.dim.clone()).leaf(false)
            }
        }

        impl<'a, S, A, B, E> $trait<&'a TensorBase<S, A>> for &'a TensorBase<S, B>
        where
            S: Dimension,
            E: DataElement,
            A: DataBuffer<Item = E>,
            B: DataBuffer<Item = E>,
        {
            type Output = Tensor<S, E>;

            #[inline]
            fn $math(self, rhs: &'a TensorBase<S, A>) -> Self::Output {
                assert_eq!(self.len(), rhs.len());
                let s_ptr = self.ptr.as_ptr();
                let o_ptr = rhs.ptr.as_ptr();
                let strides = self.default_strides();
                let mut out_vec = Vec::with_capacity(self.len());
                if (self.is_standard_layout() && rhs.is_standard_layout()) {
                    
                    unsafe {
                        for i in 0..self.len() {
                            out_vec.push(s_ptr.add(i).read().$math(o_ptr.add(i).read()));
                        }
                    }
                } else {
                    for i in 0..self.len() {
                        let idx = nd_index(i, &strides);
                        out_vec.push(self[idx.clone()].$math(rhs[idx]));
                    }
                }
                Tensor::from_vec(out_vec, self.dim.clone()).leaf(false)
            }
        }

        impl<S, A> $trait<f32> for TensorBase<S, A>
        where
            S: Dimension,
            A: DataBuffer<Item = f32>,
        {
            type Output = Tensor<S, f32>;

            #[inline]
            fn $math(self, rhs: f32) -> Self::Output {
                if self.is_standard_layout() {
                    let mut out_vec = Vec::with_capacity(self.len());
                    unsafe {
                        let ptr = self.ptr.as_ptr();
                        for i in 0..self.len() {
                            out_vec.push(ptr.add(i).read().$math(rhs));
                        }
                    }
                    Tensor::from_vec(out_vec, self.dim.clone()).leaf(false)
                } else {
                    let strides = self.default_strides();
                    let mut out_vec = Vec::with_capacity(self.len());
                    for i in 0..self.len() {
                        let idx = nd_index(i, &strides);
                        out_vec.push(self[idx].$math(rhs));
                    }
                    Tensor::from_vec(out_vec, self.dim.clone()).leaf(false)
                }
            }
        }

        impl<S, A> $trait<&TensorBase<S, A>> for f32
        where
            S: Dimension,
            A: DataBuffer<Item = f32>,
        {
            type Output = Tensor<S, f32>;

            #[inline]
            fn $math(self, rhs: &TensorBase<S, A>) -> Self::Output {
                if rhs.is_standard_layout() {
                    let mut out_vec = Vec::with_capacity(rhs.len());
                    unsafe {
                        let ptr = rhs.ptr.as_ptr();
                        for i in 0..rhs.len() {
                            out_vec.push(ptr.add(i).read().$math(self));
                        }
                    }
                    Tensor::from_vec(out_vec, rhs.dim.clone()).leaf(false)
                } else {
                    let strides = rhs.default_strides();
                    let mut out_vec = Vec::with_capacity(rhs.len());
                    for i in 0..rhs.len() {
                        let idx = nd_index(i, &strides);
                        out_vec.push(rhs[idx].$math(self));
                    }
                    Tensor::from_vec(out_vec, rhs.dim.clone()).leaf(false)
                }
            }
        }

        impl<S, A> $trait<&TensorBase<S, A>> for f64
        where
            S: Dimension,
            A: DataBuffer<Item = f64>,
        {
            type Output = Tensor<S, f64>;

            #[inline]
            fn $math(self, rhs: &TensorBase<S, A>) -> Self::Output {
                if rhs.is_standard_layout() {
                    let mut out_vec = Vec::with_capacity(rhs.len());
                    unsafe {
                        let ptr = rhs.ptr.as_ptr();
                        for i in 0..rhs.len() {
                            out_vec.push(ptr.add(i).read().$math(self));
                        }
                    }
                    Tensor::from_vec(out_vec, rhs.dim.clone()).leaf(false)
                } else {
                    let strides = rhs.default_strides();
                    let mut out_vec = Vec::with_capacity(rhs.len());
                    for i in 0..rhs.len() {
                        let idx = nd_index(i, &strides);
                        out_vec.push(rhs[idx].$math(self));
                    }
                    Tensor::from_vec(out_vec, rhs.dim.clone()).leaf(false)
                }
            }
        }

        impl<S, A> $trait<f64> for TensorBase<S, A>
        where
            S: Dimension,
            A: DataBuffer<Item = f64>,
        {
            type Output = Tensor<S, f64>;

            #[inline]
            fn $math(self, rhs: f64) -> Self::Output {
                if self.is_standard_layout() {
                    let mut out_vec = Vec::with_capacity(self.len());
                    unsafe {
                        let ptr = self.ptr.as_ptr();
                        for i in 0..self.len() {
                            out_vec.push(ptr.add(i).read().$math(rhs));
                        }
                    }
                    Tensor::from_vec(out_vec, self.dim.clone()).leaf(false)
                } else {
                    let strides = self.default_strides();
                    let mut out_vec = Vec::with_capacity(self.len());
                    for i in 0..self.len() {
                        let idx = nd_index(i, &strides);
                        out_vec.push(self[idx].$math(rhs));
                    }
                    Tensor::from_vec(out_vec, self.dim.clone()).leaf(false)
                }
            }
        }

        impl<'a, S, A> $trait<f32> for &'a TensorBase<S, A>
        where
            S: Dimension,
            A: DataBuffer<Item = f32>,
        {
            type Output = Tensor<S, f32>;

            #[inline]
            fn $math(self, rhs: f32) -> Self::Output {
                if self.is_standard_layout() {
                    let mut out_vec = Vec::with_capacity(self.len());
                    unsafe {
                        let ptr = self.ptr.as_ptr();
                        for i in 0..self.len() {
                            out_vec.push(ptr.add(i).read().$math(rhs));
                        }
                    }
                    Tensor::from_vec(out_vec, self.dim.clone()).leaf(false)
                } else {
                    let strides = self.default_strides();
                    let mut out_vec = Vec::with_capacity(self.len());
                    for i in 0..self.len() {
                        let idx = nd_index(i, &strides);
                        out_vec.push(self[idx].$math(rhs));
                    }
                    Tensor::from_vec(out_vec, self.dim.clone()).leaf(false)
                }
            }
        }

        impl<'a, S, A> $trait<f64> for &'a TensorBase<S, A>
        where
            S: Dimension,
            A: DataBuffer<Item = f64>,
        {
            type Output = Tensor<S, f64>;

            #[inline]
            fn $math(self, rhs: f64) -> Self::Output {
                if self.is_standard_layout() {
                    let mut out_vec = Vec::with_capacity(self.len());
                    unsafe {
                        let ptr = self.ptr.as_ptr();
                        for i in 0..self.len() {
                            out_vec.push(ptr.add(i).read().$math(rhs));
                        }
                    }
                    Tensor::from_vec(out_vec, self.dim.clone()).leaf(false)
                } else {
                    let strides = self.default_strides();
                    let mut out_vec = Vec::with_capacity(self.len());
                    for i in 0..self.len() {
                        let idx = nd_index(i, &strides);
                        out_vec.push(self[idx].$math(rhs));
                    }
                    Tensor::from_vec(out_vec, self.dim.clone()).leaf(false)
                }
            }
        }
    };
}

impl<S, A, B, E> MulAssign<TensorBase<S, B>> for TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer<Item = E>,
    B: DataBuffer<Item = E>,
    E: DataElement,
{
    #[inline]
    fn mul_assign(&mut self, rhs: TensorBase<S, B>) {
        self.assign_with(&rhs, |a, b| a * b);
    }
}

impl<'a, S, A, B, E> MulAssign<&'a TensorBase<S, B>> for TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer<Item = E>,
    B: DataBuffer<Item = E>,
    E: DataElement,
{
    #[inline]
    fn mul_assign(&mut self, rhs: &'a TensorBase<S, B>) {
        self.assign_with(&rhs, |a, b| a * b);
    }
}


impl<S, A, B, E> AddAssign<TensorBase<S, B>> for TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer<Item = E>,
    B: DataBuffer<Item = E>,
    E: DataElement,
{
    #[inline]
    fn add_assign(&mut self, rhs: TensorBase<S, B>) {
        self.assign_with(&rhs, |a, b| a + b);
    }
}

impl<S, A, B, E> SubAssign<TensorBase<S, B>> for TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer<Item = E>,
    B: DataBuffer<Item = E>,
    E: DataElement,
{
    #[inline]
    fn sub_assign(&mut self, rhs: TensorBase<S, B>) {
        self.assign_with(&rhs, |a, b| a - b); 
    }
}

impl<'a, S, A, B, E> AddAssign<&'a TensorBase<S, B>> for TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer<Item = E>,
    B: DataBuffer<Item = E>,
    E: DataElement,
{
    #[inline]
    fn add_assign(&mut self, rhs: &'a TensorBase<S, B>) {
        self.assign_with(&rhs, |a, b| a + b);
    }
}

impl<'a, S, A, B, E> SubAssign<&'a TensorBase<S, B>> for TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer<Item = E>,
    B: DataBuffer<Item = E>,
    E: DataElement,
{
    #[inline]
    fn sub_assign(&mut self, rhs: &'a TensorBase<S, B>) {
        self.assign_with(&rhs, |a, b| a - b);
    }
}

impl<S, A, E> MulAssign<E> for TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer<Item = E>,
    E: DataElement,
{
    #[inline]
    fn mul_assign(&mut self, rhs: E) {
        let default_strides = self.default_strides();
        if default_strides.shape() == self.strides.shape() {
            let ptr = self.ptr.as_ptr();
            for i in 0..self.len() {
                let assign_at = unsafe { ptr.add(i) };
                unsafe { assign_at.write((*assign_at) * rhs) }
            }
        } else {
            let ptr = self.ptr.as_ptr();
            for i in 0..self.len() {
                let assign_at = unsafe {
                    ptr.offset(vec_ptr_offset(
                        nd_index(i, &default_strides),
                        &self.dim,
                        &self.strides,
                    ))
                };
                unsafe { assign_at.write((*assign_at) * rhs) }
            }
        }
    }
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
                let v1: TensorView<DimMaxOf<L, R>, Dtype>;
                let v2: TensorView<DimMaxOf<L, R>, Dtype>;
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
                    *grad_lhs += g_out;
                    let g_out = grad_out.into_dimensionality::<R>();
                    *grad_rhs += g_out;
                } else {
                    *grad_lhs += reduced_grad(l.1, grad_out);
                    *grad_rhs += reduced_grad(r.1, grad_out);
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
                    *grad_lhs += g_out;
                    let g_out = grad_out.into_dimensionality::<R>();
                    *grad_rhs -= g_out;
                } else {
                    *grad_lhs += reduced_grad(l.1, grad_out);
                    *grad_rhs -= reduced_grad(r.1, grad_out);
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
                    *grad_lhs +=  g_out * rhs_clone;
                    let g_out = grad_out.into_dimensionality::<R>();
                    let lhs_clone = lhs_clone.into_dimensionality::<R>();
                    *grad_rhs += g_out * lhs_clone;
                } else {
                    let rhs_broadcast = rhs_clone.broadcast(grad_out.dim());
                    let lhs_broadcast = lhs_clone.broadcast(grad_out.dim());

                    let lhs_local_grad = rhs_broadcast * grad_out;
                    *grad_lhs += reduced_grad(l.1, &lhs_local_grad);
                    let rhs_local_grad = grad_out * lhs_broadcast;
                    *grad_rhs += reduced_grad(r.1, &rhs_local_grad);
                }
            });
        }

        out.put_backward_ops(backops);
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::impl_constructors::tensor;

    #[test]
    fn div_assign() {
        let mut a = tensor([[5., 5.], [5., 5.]]);
        let b = 5.;

        a *= 1. / b;
        assert_eq!(a, tensor([[1., 1.], [1., 1.]]));
    }
}
