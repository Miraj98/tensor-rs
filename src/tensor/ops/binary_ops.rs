use crate::prelude::{TensorBase, Function};

pub struct Add<const D: usize, A>(TensorBase<D, A>, TensorBase<D, A>);
impl<const D: usize, A> Function<D> for Add<D, A> {
    fn backward(&self, grads: &mut crate::prelude::GradientMap<D, Self::Dtype>) {
        
    }
}