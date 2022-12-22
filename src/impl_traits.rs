use crate::{
    dim::Dimension, utils::nd_index, DataBuffer, DataElement, Tensor, TensorBase, TensorView,
};
use std::{cell::RefCell, ops::Index, iter::zip};

impl<S, A> Index<S> for TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer,
{
    type Output = A::Item;

    fn index(&self, index: S) -> &Self::Output {
        assert_eq!(self.strides.slice().len(), index.slice().len());
        for (i, v) in index.slice().iter().enumerate() {
            assert!(*v < self.dim.slice()[i]);
        }
        let s = self.strides.slice().iter();
        let m = index.slice().iter();
        let mut offset: isize = 0;

        for (i, j) in zip(s, m) {
            offset += (*i) as isize * (*j) as isize;
        }

        unsafe { self.ptr.as_ptr().offset(offset).as_ref().unwrap() }
    }
}

impl<S, A> Clone for TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer,
{
    fn clone(&self) -> Self {
        TensorBase {
            id: self.id,
            ptr: self.ptr,
            data: self.data.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            is_leaf: self.is_leaf,
            requires_grad: false,
            backward_ops: RefCell::new(None),
        }
    }
}

impl<S, S2, A> PartialEq<Tensor<S2, A>> for Tensor<S, A>
where
    S: Dimension,
    S2: Dimension,
    A: DataElement,
{
    fn eq(&self, other: &Tensor<S2, A>) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        if let Some(self_s) = self.as_slice() {
            if let Some(rhs_s) = other.as_slice() {
                return self_s == rhs_s;
            }
        }

        let lhs_default_strides = self.default_strides();
        let rhs_default_strides = other.default_strides();
        for i in 0..self.len() {
            let l = nd_index(i, &lhs_default_strides);
            let r = nd_index(i, &rhs_default_strides);
            if self[l] != other[r] {
                return false;
            }
        }

        true
    }
}

impl<'a, S, S2, A> PartialEq<&'a Tensor<S2, A>> for Tensor<S, A>
where
    S: Dimension,
    S2: Dimension,
    A: DataElement,
{
    fn eq(&self, other: &&'a Tensor<S2, A>) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        if let Some(self_s) = self.as_slice() {
            if let Some(rhs_s) = other.as_slice() {
                return self_s == rhs_s;
            }
        }

        let lhs_default_strides = self.default_strides();
        let rhs_default_strides = other.default_strides();
        for i in 0..self.len() {
            let l = nd_index(i, &lhs_default_strides);
            let r = nd_index(i, &rhs_default_strides);
            if self[l] != other[r] {
                return false;
            }
        }

        true
    }
}

impl<S, S2, A> PartialEq<TensorView<S2, A>> for Tensor<S, A>
where
    S: Dimension,
    S2: Dimension,
    A: DataElement,
{
    fn eq(&self, other: &TensorView<S2, A>) -> bool {
        println!("eq being called");
        if self.shape() != other.shape() {
            println!(
                "Shapes are not equal {:?}, {:?}",
                self.shape(),
                other.shape()
            );
            return false;
        }

        if let Some(self_s) = self.as_slice() {
            if let Some(rhs_s) = other.as_slice() {
                return self_s == rhs_s;
            }
        }

        let lhs_default_strides = self.default_strides();
        let rhs_default_strides = other.default_strides();
        for i in 0..self.len() {
            let l = nd_index(i, &lhs_default_strides);
            let r = nd_index(i, &rhs_default_strides);
            if self[l] != other[r] {
                return false;
            }
        }

        true
    }
}

impl<S, S2, A> PartialEq<TensorView<S2, A>> for TensorView<S, A>
where
    S: Dimension,
    S2: Dimension,
    A: DataElement,
{
    fn eq(&self, other: &TensorView<S2, A>) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        if let Some(self_s) = self.as_slice() {
            if let Some(rhs_s) = other.as_slice() {
                return self_s == rhs_s;
            }
        }

        let lhs_default_strides = self.default_strides();
        let rhs_default_strides = other.default_strides();
        for i in 0..self.len() {
            let l = nd_index(i, &lhs_default_strides);
            let r = nd_index(i, &rhs_default_strides);
            if self[l] != other[r] {
                return false;
            }
        }

        true
    }
}

impl<S, S2, A> PartialEq<Tensor<S2, A>> for TensorView<S, A>
where
    S: Dimension,
    S2: Dimension,
    A: DataElement,
{
    fn eq(&self, other: &Tensor<S2, A>) -> bool {
        if self.shape() != other.shape() {
            return false;
        }

        if let Some(self_s) = self.as_slice() {
            if let Some(rhs_s) = other.as_slice() {
                return self_s == rhs_s;
            }
        }

        let lhs_default_strides = self.default_strides();
        let rhs_default_strides = other.default_strides();
        for i in 0..self.len() {
            let l = nd_index(i, &lhs_default_strides);
            let r = nd_index(i, &rhs_default_strides);
            if self[l] != other[r] {
                return false;
            }
        }

        true
    }
}
