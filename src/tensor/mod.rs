use std::cmp::{max, min};
use std::marker::PhantomData;
use num_integer::Integer;
use crate::prelude::{unique_id, UniqueId};

pub struct TensorBase<const D: usize, Dtype = f32> {
    id: UniqueId,
    data: Vec<Dtype>,
    dim: [usize; D],
    strides: [usize; D],
    stride_reps: [usize; D],
    marker: PhantomData<Dtype>,
}

impl<const D: usize, Dtype> TensorBase<D, Dtype> {
    pub fn from_vec(a: Vec<Dtype>, dim: [usize; D]) -> TensorBase<D, Dtype> {
        let mut strides = [1; D];
        for i in (0..dim.len() - 1).rev() {
            strides[i] = dim[i + 1] * strides[i + 1];
        }
        TensorBase {
            id: unique_id(),
            data: a,
            dim,
            strides,
            stride_reps: [1; D],
            marker: PhantomData,
        }
    }

    fn traverse(&self) where Dtype: std::fmt::Display {
        for i in 0..self.data.len() {
            let mut idx = [0; D];
            let mut r = i;
            for (is, s) in self.strides.iter().enumerate() {
                let (q, m) = r.div_rem(s);
                idx[is] = q;
                if m == 0 {
                    break;
                }
                r = m;
            }
            println!("The idx are {:?}", idx);
        }
    }

    pub fn get(&self, index: [usize; D]) -> &Dtype {
        let idx = index
            .iter()
            .enumerate()
            .fold(0, |acc, (i, val)| {
                if *val >= self.dim[i] * self.stride_reps[i] {
                    panic!("Out of bound index")
                }
                acc + self.strides[i] * (val % self.dim[i])});
        &self.data[idx]
    }

    pub fn update_stride_reps(&mut self, a: [usize; D]) {
        self.stride_reps = a;
    }

    pub fn view(&self) -> TensorBase<D, &Dtype> {
        let a = self.data.iter().collect();

        TensorBase {
            id: self.id.clone(),
            data: a,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            stride_reps: [1; D],
            marker: PhantomData,
        }
    }

    pub fn view_mut(&mut self) -> TensorBase<D, &mut Dtype> {
        let a = self.data.iter_mut().collect();

        TensorBase {
            id: self.id.clone(),
            data: a,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            stride_reps: [1; D],
            marker: PhantomData,
        }
    }

    pub fn broadcast<const N: usize>(&self, to_dim: [usize; N]) -> TensorBase<N, &Dtype> {
        assert!(self.dim.len() < N);

        let mut extended_dims = [1; N];
        for i in 0..self.dim.len() {
            if self.dim[self.dim.len() - 1 - i] != to_dim[to_dim.len() - 1 - i] {
                if min(
                    to_dim[to_dim.len() - 1 - i],
                    self.dim[self.dim.len() - 1 - i],
                ) == 1
                {
                    extended_dims[to_dim.len() - 1 - i] = max(
                        to_dim[to_dim.len() - 1 - i],
                        self.dim[self.dim.len() - 1 - i],
                    );
                } else {
                    panic!("Incompatible for broadcasting");
                }
            } else {
                extended_dims[to_dim.len() - 1 - i] = self.dim[self.dim.len() - 1 - i];
            }
        }

        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::TensorBase;

    #[test]
    fn test() {
        let a = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let t = TensorBase::from_vec(a, [2, 2, 2]);
        t.traverse();
    }

    #[test]
    fn test2() {
        let a = vec![3, 4];
        let mut t = TensorBase::from_vec(a, [2, 1]);
        t.update_stride_reps([1, 2]);
        println!("{}", t.get([1, 2]));
        t.traverse();
    }
}