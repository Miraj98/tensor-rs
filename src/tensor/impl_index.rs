use std::ops::Index;

use super::{dim::Dimension, OwnedData, TensorBase, ViewData, Data};

impl<'a, Dtype> Index<usize> for ViewData<'a, Dtype> {
    type Output = Dtype;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[idx]
    }
}

impl<Dtype: PartialEq> Index<usize> for OwnedData<Dtype> {
    type Output = Dtype;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[idx]
    }
}

impl<S: Dimension, A, Dtype> Index<S> for TensorBase<S, A>
where
    A: Index<usize, Output = Dtype> + Data<Dtype = Dtype>,
{
    type Output = Dtype;
    fn index(&self, index: S) -> &Self::Output {
        let idx = index.get_iter().enumerate().fold(0, |acc, (i, val)| {
            if *val >= self.dim[i] * self.stride_reps[i] {
                panic!("Out of bound index")
            }
            acc + self.strides[i] * (val % self.dim[i])
        });

        &self.data[idx]
    }
}
