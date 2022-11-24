use std::{rc::Rc, iter::zip};

use super::TensorBase;

impl<const D: usize> Clone for TensorBase<D> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            data: self.data.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<const D: usize> TensorBase<D> {
    pub fn broadcast<const N: usize>(&self, to_shape: [usize; N]) -> TensorBase<N, &f32> {
        assert!(to_shape.len() >= self.dim.len());

        let mut extended_dim = [1; N];
        for i in ((to_shape.len() - self.dim.len())..to_shape.len()).rev() {
            extended_dim[i] = self.dim[i];
        }

        for (i, (old, new)) in zip(self.dim, to_shape).enumerate().rev() {
            if old != new {
                if !(old == 1 || new == 1) {
                    panic!("Cannot be broadcasted");
                } else {
                    if old == 1 {}
                }
            }
        }

        let mut strides = [1; N];
        for i in 0..to_shape.len() - 1 {
            strides[i] = to_shape[i + 1];
        }

        TensorBase {
            id: self.id,
            data: Rc::new(self.data.iter().map(|a| a).collect()),
            dim: to_shape,
            strides,
        }
    }
}
