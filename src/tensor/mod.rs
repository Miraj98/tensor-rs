pub mod utils;

use crate::prelude::{unique_id, UniqueId};
use num_integer::Integer;
use std::cmp::{max, min};
use std::marker::PhantomData;
use utils::generate_strides;

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
        let total_len = dim.iter().fold(1, |acc, val| acc**val);
        assert_eq!(total_len, a.len());
        let strides = generate_strides(&dim);
        TensorBase {
            id: unique_id(),
            data: a,
            dim,
            strides,
            stride_reps: [1; D],
            marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.dim.iter().fold(1, |acc, val| acc**val)
    }

    fn traverse(&self)
    where
        Dtype: std::fmt::Display,
    {
        for i in 0..self.len() {
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
            println!("{:?} -> {}", idx, self.unbounded_get(idx));
        }
    }

    pub fn unbounded_get(&self, index: [usize; D]) -> &Dtype {
        let idx = index.iter().enumerate().fold(0, |acc, (i, val)| {
            acc + self.strides[i] * (val % self.dim[i])
        });
        &self.data[idx]
    }

    pub fn get(&self, index: [usize; D]) -> &Dtype {
        let idx = index.iter().enumerate().fold(0, |acc, (i, val)| {
            if *val >= self.dim[i] * self.stride_reps[i] {
                panic!("Out of bound index")
            }
            acc + self.strides[i] * (val % self.dim[i])
        });
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

    pub fn broadcast<const N: usize>(&self, to_dim: [usize; N]) -> TensorBase<N, &Dtype> where Dtype: std::fmt::Display {
        assert!(D < N);

        let mut extended_dims = [1; N];
        self.dim
            .iter()
            .enumerate()
            .for_each(|(i, val)| extended_dims[N - D + i] = *val);
        let padded_dims = extended_dims.clone();
        let padded_strides = generate_strides(&padded_dims);

        let mut stride_reps = [1; N];
        self.stride_reps
            .iter()
            .enumerate()
            .for_each(|(i, val)| stride_reps[N - D + i] = *val);

        for i in 0..N {
            if extended_dims[N - 1 - i] != to_dim[N - 1 - i] {
                if min(to_dim[N - 1 - i], extended_dims[N - 1 - i]) == 1 {
                    extended_dims[N - 1 - i] = max(to_dim[N - 1 - i], extended_dims[N - 1 - i]);
                    if extended_dims[N - 1 - i] == 1 {
                        stride_reps[N - 1 - i] *= extended_dims[N - 1 - i]
                    }
                } else {
                    panic!("Incompatible for broadcasting");
                }
            } else {
                extended_dims[N - 1 - i] = extended_dims[N - 1 - i];
            }
        }

        println!("{:?} broadcasted to -> {:?}", self.dim, extended_dims);
        println!("[DIMENSION] {:?} padded to -> {:?}", self.dim, padded_dims);
        println!("[STRIDES] {:?} padded to -> {:?}", self.strides, padded_strides);

        // Traverse all data points to generate the broadcasted view
        let new_len = extended_dims.iter().fold(1, |acc, val| acc**val);
        let new_strides = generate_strides(&extended_dims);
        println!("New data size: {}", new_len);
        for i in 0..new_len {
            // For every i create the tuple repr of the idx
            let mut idx = [0; N];
            let mut r = i;
            for (is, s) in new_strides.iter().enumerate() {
                let (q, m) = r.div_rem(s);
                idx[is] = q;
                if m == 0 {
                    break;
                }
                r = m;
            }
            println!("Curr idx {:?}", idx);

            let id = idx.iter().enumerate().fold(0, |acc, (i, val)| {
                acc + padded_strides[i] * (val % padded_dims[i])
            });
            let val = &self.data[id];
        }

        TensorBase {
            id: self.id.clone(),
            data: self.data.iter().collect(),
            dim: extended_dims,
            strides: generate_strides(&extended_dims),
            marker: PhantomData,
            stride_reps,
        }
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

    #[test]
    fn broadcast_test() {
        let a = vec![3, 4];
        let t = TensorBase::from_vec(a, [2, 1]);
        let a = t.broadcast([3, 1, 5]);
        println!("strides {:?}", a.strides);
        println!("dim {:?}", a.dim);
        println!("data {:?}", a.data);
        // a.traverse();
    }
}
