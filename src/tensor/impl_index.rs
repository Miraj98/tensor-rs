use std::ops::{Index};

use super::{TensorBase, dim::Dimension};

impl<S: Dimension, Dtype> Index<S> for TensorBase<S, Dtype> {
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