use super::{dim::Dimension, Data, TensorBase};

impl<S, A> PartialEq for TensorBase<S, A>
where
    S: Dimension,
    A: Data,
{
    fn eq(&self, other: &Self) -> bool {
        println!("Comparing two tensors");
        self.dim.shape() == other.dim.shape()
            && self.strides.shape() == other.dim.shape()
            && self.data == other.data
    }
}
