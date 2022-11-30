use std::iter::zip;

use crate::{
    num_taits::{One, Zero},
    prelude::{
        dim::{DimMax, DimMaxOf, Dimension},
        utils::merge_backward_ops,
        TensorBase,
    },
};

impl<'a, L, R, Dtype> std::ops::Add<&'a TensorBase<R, Dtype>> for &'a TensorBase<L, Dtype>
where
    R: Dimension,
    L: DimMax<R> + Dimension,
    Dtype: One + Zero,
    &'a Dtype: std::ops::Add<&'a Dtype, Output = Dtype>,
{
    type Output = TensorBase<DimMaxOf<L, R>, Dtype>;

    fn add(self, rhs: &'a TensorBase<R, Dtype>) -> Self::Output {
        if self.shape() == rhs.shape() {
            let mut a = Vec::<Dtype>::with_capacity(self.len());
            for (l, r) in zip(self.data.iter(), rhs.data.iter()) {
                a.push(l + r);
            }
            let dim = self.dim.into_dimensionality::<DimMaxOf<L, R>>();
            let mut backops = merge_backward_ops(self, rhs);
            if backops.is_some() {
                // let b_ops = backops.as_mut().unwrap().add_backward_op(|grads| {

                // })
            }
            TensorBase::from_vec(a, dim).with_backops(backops)
        } else {
            let v1: TensorBase<DimMaxOf<L, R>, &Dtype>;
            let v2: TensorBase<DimMaxOf<L, R>, &Dtype>;
            let dim: DimMaxOf<L, R>;
            if self.ndim() >= rhs.ndim() {
                dim = self.dim().into_dimensionality::<DimMaxOf<L, R>>();
            } else {
                dim = rhs.dim().into_dimensionality::<DimMaxOf<L, R>>();
            }
            v1 = self.broadcast(dim.clone());
            v2 = rhs.broadcast(v1.dim());

            let mut a = Vec::<Dtype>::with_capacity(self.len());
            for (l, r) in zip(v1.data.iter(), v2.data.iter()) {
                a.push(*l + *r);
            }

            let backops = merge_backward_ops(self, rhs);
            TensorBase::from_vec(a, dim).with_backops(backops)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::TensorBase;

    #[test]
    fn add_tensors() {
        let t1 = TensorBase::from_vec(vec![1, 2, 3], [1, 3]);
        let t2 = TensorBase::from_vec(vec![1, 2, 3], [1, 3]);
        let c = &t1 + &t2;
        println!("{:?}", c);
    }
}
