use crate::{DataBuffer, TensorBase, dim::Dimension};

pub struct TensorIterator<'a, S, A>
where
    S: Dimension,
    A: DataBuffer
{
    t: &'a TensorBase<S, A>,
    dim: S,
    current_dim: usize,
    nd_index: S,
}

impl<'a, S, A> IntoIterator for &'a TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer,
{
    type Item = A::Item;
    type IntoIter = TensorIterator<'a, S, A>;

    fn into_iter(self) -> Self::IntoIter {
        TensorIterator { t: self, dim: self.dim(), current_dim: self.dim.ndim() - 1, nd_index: S::zeros() } 
    }
}

impl<'a, S, A> Iterator for TensorIterator<'a, S, A>
where
    S: Dimension,
    A: DataBuffer
{
    type Item = A::Item;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.nd_index[self.current_dim] < self.dim[self.current_dim] {
                let val = Some(self.t[self.nd_index.clone()]);
                self.nd_index[self.current_dim] += 1;
                return val;
            } else if self.current_dim > 0 {
                self.current_dim -= 1;
            } else {
                return None;
            }
        }
    }
}
