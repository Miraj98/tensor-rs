use std::{cell::RefCell, ops::Range, ptr::NonNull};
use crate::{
    dim::{Dimension, Ix0},
    gradient::{BackwardOps, GradientMap},
    impl_constructors::TensorConstructors,
    unique_id::unique_id,
    utils::{generate_strides, nd_index, unlimited_transmute, vec_ptr_offset},
    DataBuffer, DataElement, OwnedData, Tensor, TensorBase, TensorView, TensorViewMut, UniqueId,
    ViewData, ViewMutData
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
        } else if !b && self.is_leaf && self.backward_ops.borrow().is_some() {
           let _ = self.detach_backward_ops(); 
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

    pub fn invert_axis(&mut self, i: usize) {
        let m = self.dim[i];
        let s = self.strides[i] as isize;
        let ptr = self.ptr.as_ptr();
        unsafe { self.ptr = NonNull::new(ptr.offset((m - 1) as isize * s)).unwrap() }
        self.strides[i] = (-s) as usize;
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

    pub fn map_inplace(&mut self, mut f: impl FnMut(&A::Item) -> A::Item) {
        let ptr = self.ptr.as_ptr();
        unsafe {
            for i in 0..self.len() {
                let p = ptr.add(i);
                let x = p.read();
                p.write(f(&x));
            }
        }
    }

    pub fn outer_dim(&self, i: usize) -> TensorView<S::Smaller, A::Item> {
        assert!(i < self.dim[0]);
        let mut ptr = self.ptr.as_ptr();
        unsafe { ptr = ptr.offset(i as isize * self.strides[0] as isize) };
        let mut dim = S::Smaller::ones();
        let mut strides = S::Smaller::ones();
        for i in 1..self.dim.ndim() {
            dim[i - 1] = self.dim[i];
            strides[i - 1] = self.strides[i];
        }

        TensorView {
            id: self.id,
            dim,
            strides,
            is_leaf: self.is_leaf,
            ptr: NonNull::new(ptr).unwrap(),
            data: ViewData { _view: self.data.data() },
            backward_ops: RefCell::new(None),
            requires_grad: self.requires_grad,
        }
    }

    pub fn slice_2d(&self, dx: Range<usize>, dy: Range<usize>) -> TensorView<S, A::Item> {
        assert!(self.ndim() >= 2);
        let n = self.ndim();
        let mut out_dim = self.dim();
        out_dim[n - 1] = dx.end - dx.start;
        out_dim[n - 2] = dy.end - dy.start;

        let (rs, cs) = (self.strides[n - 2] as isize, self.strides[n - 1] as isize);
        let (ox, oy) = (dx.start, dy.start);

        let ptr = unsafe {
            NonNull::new_unchecked(
                self.ptr
                    .as_ptr()
                    .offset(ox as isize * cs + oy as isize * rs),
            )
        };
        self.view_from_data_ptr_and_dim(ptr, out_dim)
    }

    pub fn slice_mut_2d(
        &mut self,
        dx: Range<usize>,
        dy: Range<usize>,
    ) -> TensorViewMut<S, A::Item> {
        assert!(self.ndim() >= 2);
        let n = self.ndim();
        let mut out_dim = self.dim();
        out_dim[n - 1] = dx.end - dx.start;
        out_dim[n - 2] = dy.end - dy.start;

        let (rs, cs) = (self.strides[n - 2] as isize, self.strides[n - 1] as isize);
        let (ox, oy) = (dx.start, dy.start);

        let ptr = unsafe {
            NonNull::new_unchecked(
                self.ptr
                    .as_ptr()
                    .offset(ox as isize * cs + oy as isize * rs),
            )
        };
        self.view_mut_from_data_ptr_and_dim(ptr, out_dim)
    }

    pub fn assign<B>(&mut self, other: &TensorBase<S, B>)
    where
        B: DataBuffer<Item = A::Item>,
    {
        self.assign_with(other, |_, y| y);
    }

    pub fn assign_with<B>(
        &mut self,
        other: &TensorBase<S, B>,
        f: impl Fn(A::Item, B::Item) -> A::Item,
    ) where
        B: DataBuffer<Item = A::Item>,
    {
        assert_eq!(self.shape(), other.shape());
        if self.is_standard_layout() && other.is_standard_layout() {
            let self_ptr = self.ptr.as_ptr();
            let rhs_ptr = other.ptr.as_ptr();
            for i in 0..self.len() {
                let assign_at = unsafe { self_ptr.add(i) };
                let rhs_at = unsafe { rhs_ptr.add(i) };
                unsafe { assign_at.write(f(*assign_at, *rhs_at)) }
            }
        } else {
            let default_strides = self.default_strides();
            let ptr = self.ptr.as_ptr();
            let o_ptr = other.ptr.as_ptr();
            for i in 0..other.len() {
                let assign_at = unsafe {
                    ptr.offset(vec_ptr_offset(
                        nd_index(i, &default_strides),
                        &self.dim,
                        &self.strides,
                    ))
                };
                let assign = unsafe {
                    o_ptr.offset(vec_ptr_offset(
                        nd_index(i, &default_strides),
                        &other.dim,
                        &other.strides,
                    ))
                };
                unsafe { assign_at.write(f(*assign_at, *assign)) }
            }
        }
    }

    pub fn into_dimensionality<D2: Dimension>(&self) -> &TensorBase<D2, A> {
        unsafe { unlimited_transmute(self) }
    }

    fn view_from_data_ptr_and_dim(
        &self,
        ptr: NonNull<A::Item>,
        dim: S,
    ) -> TensorView<S, A::Item> {
        TensorBase {
            id: self.id,
            data: ViewData {
                _view: self.data.data()
            },
            ptr,
            dim,
            strides: self.strides.clone(),
            is_leaf: self.is_leaf,
            requires_grad: false,
            backward_ops: RefCell::new(None),
        }
    }

    fn view_mut_from_data_ptr_and_dim(
        &mut self,
        ptr: NonNull<A::Item>,
        dim: S,
    ) -> TensorViewMut<S, A::Item> {
        TensorBase {
            id: self.id,
            data: ViewMutData { _view: self.data.data() },
            ptr,
            dim,
            strides: self.strides.clone(),
            is_leaf: self.is_leaf,
            requires_grad: false,
            backward_ops: RefCell::new(None),
        }
    }

    pub fn view(&self) -> TensorView<S, A::Item> {
        TensorBase {
            id: self.id,
            data: ViewData {
                _view: self.data.data()
            },
            ptr: self.ptr,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            is_leaf: self.is_leaf,
            requires_grad: false,
            backward_ops: RefCell::new(None),
        }
    }

    pub fn view_mut(&self) -> TensorViewMut<S, A::Item> {
        TensorBase {
            id: self.id,
            data: ViewMutData {
                _view: self.data.data(),
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

    pub fn t(&self) -> TensorView<S, A::Item> {
        let mut self_view = self.view();
        let strides = self.strides.rev();
        let dim = self.dim.rev();
        self_view.dim = dim;
        self_view.strides = strides;

        self_view
    }

    pub fn reshape<S2>(&self, dim: S2) -> TensorView<S2, A::Item>
    where
        S2: Dimension,
    {
        assert_eq!(self.dim.count(), dim.count());
        let strides = generate_strides(&dim);
        let self_view = TensorBase {
            id: self.id,
            data: ViewData {
                _view: self.data.data(),
            },
            ptr: self.ptr,
            dim,
            strides,
            is_leaf: self.is_leaf,
            requires_grad: false,
            backward_ops: RefCell::new(None),
        };
        self_view
    }

    pub fn shape(&self) -> &[usize] {
        self.dim.slice()
    }

    pub fn broadcast<K>(&self, dim: K) -> TensorView<K, A::Item>
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
                _view: self.data.data(),
            },
            ptr: self.ptr,
            dim,
            strides: new_strides,
            is_leaf: self.is_leaf,
            requires_grad: self.requires_grad,
            backward_ops: RefCell::new(None),
        }
    }

    pub fn detach_backward_ops(&self) -> Option<BackwardOps> {
        let ret = self.backward_ops.borrow_mut().take();
        ret
    }

    pub fn put_backward_ops(&self, backops: Option<BackwardOps>) {
        *self.backward_ops.borrow_mut() = backops;
    }

    pub fn max(&self) -> (usize, A::Item) {
        todo!()
    }
}

impl<A> TensorBase<Ix0, A>
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
    use crate::impl_constructors::tensor;

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

    #[test]
    fn assign_test() {
        let mut a = Tensor::ones([3, 3]);
        let b = Tensor::zeros([4, 4]);
        let mut aview = a.slice_mut_2d(0..2, 0..2);
        let bview = b.slice_2d(0..2, 0..2);
        aview.assign(&bview);
        assert_eq!(a, tensor([[0., 0., 1.], [0., 0., 1.], [1., 1., 1.]]));
    }

    #[test]
    fn invert_axis() {
        let mut a = tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        a.invert_axis(0);
        assert_eq!(a, tensor([[7., 8., 9.], [4., 5., 6.], [1., 2., 3.]]));
        a.invert_axis(1);
        assert_eq!(a, tensor([[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]));
        a.invert_axis(0);
        a.invert_axis(1);
        assert_eq!(a, tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]));
    }

    #[test]
    fn slice_2d() {
        let ones: Tensor<_, f32> = Tensor::ones([2, 2, 2]);
        let px = 1;
        let py = 1;
        let mut x_padded: Tensor<_, f32> = Tensor::zeros([2, 4, 4]);
        x_padded.slice_mut_2d(px..px+2, py..py+2).assign(&ones);
        assert_eq!(x_padded, tensor([[
            [0., 0., 0., 0.],
            [0., 1., 1., 0.],
            [0., 1., 1., 0.],
            [0., 0., 0., 0.],
        ], [
            [0., 0., 0., 0.],
            [0., 1., 1., 0.],
            [0., 1., 1., 0.],
            [0., 0., 0., 0.],
        ]]))
    }
}
