pub mod impl_index;
pub mod utils;

use crate::unique_id::{unique_id, UniqueId};
use std::cmp::{max, min};
use std::marker::PhantomData;
use std::rc::Rc;
use utils::{generate_strides, tnsr_idx, vec_id};

pub struct TensorBase<const D: usize, Dtype = f32> {
    id: UniqueId,
    data: Rc<Vec<Dtype>>,
    dim: [usize; D],
    strides: [usize; D],
    stride_reps: [usize; D],
    marker: PhantomData<Dtype>,
}

impl<const D: usize, Dtype> TensorBase<D, Dtype> {
    pub fn from_vec(a: Vec<Dtype>, dim: [usize; D]) -> TensorBase<D, Dtype> {
        let total_len = dim.iter().fold(1, |acc, val| acc * *val);
        assert_eq!(total_len, a.len());
        let strides = generate_strides(&dim);
        TensorBase {
            id: unique_id(),
            data: Rc::new(a),
            dim,
            strides,
            stride_reps: [1; D],
            marker: PhantomData,
        }
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
            data: Rc::new(a),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            stride_reps: [1; D],
            marker: PhantomData,
        }
    }

    pub fn broadcast<const N: usize>(&self, to_dim: [usize; N]) -> TensorBase<N, &Dtype>
    where
        Dtype: std::fmt::Display,
    {
        assert!(D < N);

        // New dimensions
        let mut extended_dims = [1; N];
        self.dim
            .iter()
            .enumerate()
            .for_each(|(i, val)| extended_dims[N - D + i] = *val);
        // Old dimensions and strides but padding the extra dims as 1
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

        // Traverse all data points to generate the broadcasted view
        let new_len = extended_dims.iter().fold(1, |acc, val| acc * *val);
        let new_strides = generate_strides(&extended_dims);
        let mut broadcasted_data = Vec::<&Dtype>::with_capacity(new_len);
        for i in 0..new_len {
            let id = vec_id(tnsr_idx(i, &new_strides), &padded_dims, &padded_strides);
            broadcasted_data.push(&self.data[id]);
        }

        TensorBase {
            id: self.id.clone(),
            data: Rc::new(broadcasted_data),
            dim: extended_dims,
            strides: generate_strides(&extended_dims),
            marker: PhantomData,
            stride_reps,
        }
    }
}

impl<const D: usize, Dtype> Clone for TensorBase<D, Dtype> {
    fn clone(&self) -> Self {
        TensorBase {
            id: self.id.clone(),
            data: Rc::clone(&self.data),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            stride_reps: self.stride_reps.clone(),
            marker: self.marker,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TensorBase;

    #[test]
    fn test2() {
        let a = vec![3, 4];
        let mut t = TensorBase::from_vec(a, [2, 1]);
        t.update_stride_reps([1, 2]);
        println!("{}", t.get([1, 2]));
    }

    #[test]
    fn broadcast_test() {
        let a = vec![3, 4];
        let t = TensorBase::from_vec(a, [2, 1]);
        let a = t.broadcast([3, 1, 5]);
        println!("strides {:?}", a.strides);
        println!("dim {:?}", a.dim);
        println!("data {:?}", a.data);
    }
}
