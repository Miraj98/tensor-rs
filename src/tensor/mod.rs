pub mod impl_tensor;
pub mod impl_constructors;
pub mod ops;

use std::rc::Rc;
use crate::unique_id::UniqueId;

#[derive(Debug)]
pub struct TensorBase<const D: usize, Dtype = f32> {
    id: UniqueId,
    data: Rc<Vec<Dtype>>,
    pub dim: [usize; D],
    pub strides: [usize; D],
}
