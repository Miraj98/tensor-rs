use crate::{dim::Dimension, DataBuffer, TensorBase};

pub struct TensorIterator<'a, S, A>
where
    S: Dimension,
    A: DataBuffer,
{
    t: &'a TensorBase<S, A>,
    ndim: usize,
    nd_index: Option<S>,
    max_nd_index: S,
}

impl<'a, S, A> IntoIterator for &'a TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer,
{
    type Item = A::Item;
    type IntoIter = TensorIterator<'a, S, A>;

    fn into_iter(self) -> Self::IntoIter {
        let mut max_nd_index = S::zeros();
        for (i, val) in self.dim.get_iter().enumerate() {
            max_nd_index[i] = *val - 1;
        }
        TensorIterator {
            t: self,
            ndim: self.dim.ndim(),
            nd_index: Some(S::zeros()),
            max_nd_index,
        }
    }
}

impl<'a, S, A> Iterator for TensorIterator<'a, S, A>
where
    S: Dimension,
    A: DataBuffer,
{
    type Item = A::Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.nd_index.is_none() {
            return None;
        }
        let nd_index = self.nd_index.take().unwrap();
        let val = self.t[nd_index.clone()];

        // Go to the index that must be updated
        let mut axis = self.ndim - 1;
        let mut next_nd_idx = nd_index.clone();
        loop {
            if nd_index[axis] == self.max_nd_index[axis] {
                if axis == 0 {
                    return Some(val); // Due to the `take()` above, the `nd_index` will automatically be pointing to `None` in the next `next()` call.
                }
                next_nd_idx[axis] = 0;
                axis -=1 ;
            } else {
                next_nd_idx[axis] += 1;
                break;
            }
        }

        self.nd_index = Some(next_nd_idx);
        return Some(val);
    }
}

#[cfg(test)]
mod tests {
    use crate::impl_constructors::tensor;

    #[test]
    fn iter_tensor() {
        let b = tensor([
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]],
            [[13., 14., 15.], [16., 17., 18.], [19., 20., 21.], [22., 23., 24.]],
        ]);

        for (i, val) in b.into_iter().enumerate() {
            assert_eq!(val, (i + 1) as f32);
        }
    }
}
