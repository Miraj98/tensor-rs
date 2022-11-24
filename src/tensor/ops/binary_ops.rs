use crate::prelude::{TensorBase, Function};

pub struct Add<const D: usize>(pub TensorBase<D>, pub TensorBase<D>);

impl<const D: usize> Function for Add<D> {
    type Output = TensorBase<D>;

    fn forward(&self) -> Self::Output {
        todo!()
    }

    fn backward(&self) {
        todo!()
    }
}