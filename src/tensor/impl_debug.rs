use super::TensorBase;
use std::fmt::{Debug, Formatter};

impl Debug for TensorBase<1> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.data.iter())
            .entry(&format!("shape={:?}", self.dim))
            .entry(&format!("strides={:?}", self.strides))
            .finish()
    }
}

impl Debug for TensorBase<2> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut vec = Vec::<Vec<f32>>::new();

        for (idx, dim) in self.dim.iter().enumerate() {
            let mut inner = Vec::<f32>::new();
            for j in 0..*dim {
                inner.push(self.data[j * self.strides[idx]])
            }
            vec.push(inner);
        }

        f.debug_list()
            .entries(vec)
            .entry(&format!("shape={:?}", self.dim))
            .entry(&format!("strides={:?}", self.strides))
            .finish()
    }
}

impl Debug for TensorBase<3> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut vec = Vec::<Vec<Vec<f32>>>::new();

        for i in 0..self.dim[0] {
            let mut vec2d = Vec::<Vec<f32>>::new();
            for j in 0..self.dim[1] {
                let mut vec1d = Vec::<f32>::new();
                for k in 0..self.dim[2] {
                    vec1d.push(
                        self.data[i * self.strides[0] + j * self.strides[1] + k * self.strides[2]],
                    );
                }
                vec2d.push(vec1d);
            }
            vec.push(vec2d);
        }

        f.debug_list()
            .entries(vec)
            .entry(&format!("shape={:?}", self.dim))
            .entry(&format!("strides={:?}", self.strides))
            .finish()
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::TensorBase;

    #[test]
    fn print1d() {
        let a = TensorBase::ones([10]);
        println!("{:#?}", a);
    }

    #[test]
    fn print2d() {
        let a = TensorBase::ones([2, 2]);
        println!("{:#?}", a);
    }

    #[test]
    fn print3d() {
        let a = TensorBase::ones([2, 2, 2]);
        println!("{:?}", a);
    }
}
