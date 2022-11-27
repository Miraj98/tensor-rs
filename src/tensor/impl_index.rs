use std::ops::{Index};

use super::TensorBase;

impl<const D: usize, Dtype> Index<[usize; D]> for TensorBase<D, Dtype> {
    type Output = Dtype;
    fn index(&self, index: [usize; D]) -> &Self::Output {
        let idx = index.iter().enumerate().fold(0, |acc, (i, val)| {
            if *val >= self.dim[i] * self.stride_reps[i] {
                panic!("Out of bound index")
            }
            acc + self.strides[i] * (val % self.dim[i])
        });

        &self.data[idx]
    }
} 