use std::rc::Rc;

use super::TensorBase;
use crate::unique_id::unique_id;

impl<const D: usize> TensorBase<D> {
    fn from_elem(shape: [usize; D], elem: f32) -> Self {
        let total_size = shape.iter().fold(1, |acc, x| acc * x);
        let mut strides = [1; D];

        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        Self {
            id: unique_id(),
            data: Rc::new(vec![elem; total_size]),
            dim: shape,
            strides,
        }
    }

    pub fn zeros(shape: [usize; D]) -> Self {
        Self::from_elem(shape, 0.)
    }

    pub fn ones(shape: [usize; D]) -> Self {
        Self::from_elem(shape, 1.)
    }

    pub fn randn(_shape: [usize; D]) -> Self {
        todo!()
    }
}
