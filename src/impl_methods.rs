use std::{cell::RefCell, ptr::NonNull, ops::Range};

use crate::{
    dim::Dimension,
    gradient::{BackwardOps, GradientMap},
    impl_constructors::TensorConstructors,
    unique_id::unique_id,
    utils::{generate_strides, nd_index, unlimited_transmute},
    DataBuffer, DataElement, OwnedData, Tensor, TensorBase, TensorView, TensorViewMut, UniqueId,
    ViewData,
};

impl<S, Dtype> Tensor<S, Dtype>
where
    S: Dimension,
    Dtype: DataElement,
{
    pub fn from_vec(mut a: Vec<Dtype>, dim: S) -> Tensor<S, Dtype> {
        let total_len = dim.count();
        assert_eq!(total_len, a.len());
        let v = unsafe { NonNull::new_unchecked(a.as_mut_ptr()) };
        let strides = generate_strides(&dim);
        TensorBase {
            id: unique_id(),
            data: OwnedData::new(a),
            ptr: v,
            dim,
            strides,
            is_leaf: true,
            requires_grad: false,
            backward_ops: RefCell::new(None),
        }
    }

    pub fn requires_grad(mut self, b: bool) -> Self {
        self.requires_grad = b;
        if b && self.is_leaf && self.backward_ops.borrow().is_none() {
            *self.backward_ops.borrow_mut() = Some(BackwardOps(Vec::new()));
        }

        self
    }

    pub fn leaf(mut self, is_leaf: bool) -> Self {
        self.is_leaf = is_leaf;
        self
    }
}

impl<S, A> TensorBase<S, A>
where
    S: Dimension,
    A: DataBuffer,
{
    pub(crate) fn id(&self) -> &UniqueId {
        &self.id
    }

    pub fn len(&self) -> usize {
        self.dim.count()
    }

    pub fn dim(&self) -> S {
        self.dim.clone()
    }

    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }

    pub fn strides(&self) -> S {
        self.strides.clone()
    }

    pub fn as_slice(&self) -> Option<&[A::Item]> {
        if self.is_standard_layout() {
            unsafe {
                Some(std::slice::from_raw_parts(
                    self.ptr.as_ptr(),
                    self.dim.count(),
                ))
            }
        } else {
            None
        }
    }

    pub fn map(&self, mut f: impl FnMut(&A::Item) -> A::Item) -> Tensor<S, A::Item> {
        if let Some(slc) = self.as_slice() {
            let new_data = slc.iter().map(f).collect();
            Tensor::from_vec(new_data, self.dim.clone())
        } else {
            let default_strides = self.default_strides();
            let mut out_vec = Vec::with_capacity(self.len());
            for i in 0..self.len() {
                let idx = nd_index(i, &default_strides);
                out_vec[i] = f(&self[idx]);
            }
            Tensor::from_vec(out_vec, self.dim())
        }
    }

    pub fn slice_2d(&self, dx: Range<usize>, dy: Range<usize>) -> TensorView<'_, S, A::Item> {
        assert!(self.ndim() >= 2);
        let n = self.ndim();
        let mut out_dim = self.dim();
        out_dim[n - 1] = dx.end - dx.start;
        out_dim[n - 2] = dy.end - dy.start;

        let (rs, cs) = (self.strides[n - 2], self.strides[n - 1]);
        let (ox, oy) = (dx.start, dy.start);

        let ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(ox * cs + oy * rs)) };
        self.view_from_data_ptr_and_dim(ptr, out_dim)
    }

    pub fn slice_mut_2d(&mut self, dx: Range<usize>, dy: Range<usize>) -> TensorViewMut<'_, S, A::Item> {
        assert!(self.ndim() >= 2);
        let n = self.ndim();
        let mut out_dim = self.dim();
        out_dim[n - 1] = dx.end - dx.start;
        out_dim[n - 2] = dy.end - dy.start;

        let (rs, cs) = (self.strides[n - 2], self.strides[n - 1]);
        let (ox, oy) = (dx.start, dy.start);

        let ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(ox * cs + oy * rs)) };
        self.view_mut_from_data_ptr_and_dim(ptr, out_dim)
    }

    pub fn assign<S2, B>(&mut self, other: &TensorBase<S2, B>)
    where
        S2: Dimension,
        B: DataBuffer<Item = A::Item>,
    {
        assert!(self.len() >= other.len());
        todo!()
    }

    pub fn into_dimensionality<D2: Dimension>(&self) -> &TensorBase<D2, A> {
        unsafe { unlimited_transmute(self) }
    }

    fn view_from_data_ptr_and_dim(&self, ptr: NonNull<A::Item>, dim: S) -> TensorView<'_, S, A::Item>  {
        TensorBase {
            id: self.id,
            data: ViewData {
                marker: std::marker::PhantomData::<&A::Item>,
            },
            ptr,
            dim,
            strides: self.strides.clone(),
            is_leaf: self.is_leaf,
            requires_grad: false,
            backward_ops: RefCell::new(None),
        }
    }

    fn view_mut_from_data_ptr_and_dim(&mut self, ptr: NonNull<A::Item>, dim: S) -> TensorViewMut<'_, S, A::Item>  {
        TensorBase {
            id: self.id,
            data: ViewData {
                marker: std::marker::PhantomData::<&mut A::Item>,
            },
            ptr,
            dim,
            strides: self.strides.clone(),
            is_leaf: self.is_leaf,
            requires_grad: false,
            backward_ops: RefCell::new(None),
        }
    }

    pub fn view(&self) -> TensorView<'_, S, A::Item> {
        TensorBase {
            id: self.id,
            data: ViewData {
                marker: std::marker::PhantomData::<&A::Item>,
            },
            ptr: self.ptr,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            is_leaf: self.is_leaf,
            requires_grad: false,
            backward_ops: RefCell::new(None),
        }
    }

    pub fn view_mut(&self) -> TensorViewMut<'_, S, A::Item> {
        TensorBase {
            id: self.id,
            data: ViewData {
                marker: std::marker::PhantomData::<&mut A::Item>,
            },
            ptr: self.ptr,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            is_leaf: self.is_leaf,
            requires_grad: false,
            backward_ops: RefCell::new(None),
        }
    }

    pub fn is_standard_layout(&self) -> bool {
        self.strides.slice() == generate_strides(&self.dim).slice()
    }

    pub fn default_strides(&self) -> S {
        generate_strides(&self.dim)
    }

    pub fn t(&self) -> TensorView<'_, S, A::Item> {
        let mut self_view = self.view();
        let strides = self.strides.rev();
        let dim = self.dim.rev();
        self_view.dim = dim;
        self_view.strides = strides;

        self_view
    }

    pub fn reshape(&self, dim: S) -> TensorView<'_, S, A::Item> {
        assert_eq!(self.dim.count(), dim.count());
        let mut self_view = self.view();
        let strides = generate_strides(&dim);
        self_view.dim = dim;
        self_view.strides = strides;
        self_view
    }

    pub fn shape(&self) -> &[usize] {
        self.dim.slice()
    }

    pub fn broadcast<K>(&self, dim: K) -> TensorView<'_, K, A::Item>
    where
        K: Dimension,
    {
        assert!(self.dim.ndim() <= dim.ndim());
        let mut new_strides = dim.clone();
        let mut new_strides_iter = new_strides.slice_mut().iter_mut().rev();

        for ((er, es), tr) in self
            .dim
            .slice()
            .iter()
            .rev()
            .zip(self.strides.slice().iter().rev())
            .zip(new_strides_iter.by_ref())
        {
            if *er == *tr {
                *tr = *es;
            } else {
                assert_eq!(*er, 1);
                *tr = 0;
            }
        }

        for tr in new_strides_iter {
            *tr = 0;
        }

        TensorBase {
            id: self.id,
            data: ViewData {
                marker: std::marker::PhantomData,
            },
            ptr: self.ptr,
            dim,
            strides: new_strides,
            is_leaf: self.is_leaf,
            requires_grad: self.requires_grad,
            backward_ops: RefCell::new(None),
        }
    }

    pub(crate) fn detach_backward_ops(&self) -> Option<BackwardOps> {
        self.backward_ops.borrow_mut().take()
    }

    pub(crate) fn put_backward_ops(&self, backops: Option<BackwardOps>) {
        *self.backward_ops.borrow_mut() = backops;
    }
}

impl<A> TensorBase<[usize; 0], A>
where
    A: DataBuffer,
{
    pub fn backward(&self) -> GradientMap
    where
        A: 'static,
    {
        if self.backward_ops.borrow().is_none() {
            panic!("Use requires_grad(true) to enable gradient computation");
        }

        let mut backops = self.detach_backward_ops().unwrap();
        let id = self.id;
        let dim = self.dim();
        backops.add_backward_op(move |grad| {
            let mut_ref: &mut Tensor<[usize; 0], A::Item> = grad.mut_grad_by_id(id, dim.clone());
            *mut_ref = Tensor::ones(dim);
        });

        let grads = backops.execute();
        grads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_test() {
        let avec: Vec<f32> = vec![3., 4.];
        let a = TensorBase::from_vec(avec, [2, 1]);
        let broadcasted = a.broadcast([3, 2, 5]);
        let similar = TensorBase::from_vec(
            vec![
                3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0,
                4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0,
            ],
            [3, 2, 5],
        );
        assert_eq!(broadcasted, similar);
    }
}
