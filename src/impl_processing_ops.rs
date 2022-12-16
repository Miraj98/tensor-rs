use crate::{
    dim::{Ix2, Ix3, Ix4},
    impl_constructors::TensorConstructors,
    utils::{generate_strides, merge_backward_ops},
    DataBuffer, Tensor, TensorBase,
};
use matrixmultiply::sgemm;

impl<A> Matmul<TensorBase<Ix2, A>> for TensorBase<Ix2, A>
where
    A: DataBuffer<Item = f32> + 'static,
{
    type Output = Tensor<Ix2, f32>;

    fn matmul(&self, rhs: &TensorBase<Ix2, A>) -> Self::Output {
        let mut backops = merge_backward_ops(self, rhs);
        let out = self.dot(rhs);
        let out_id = out.id;
        let lhs_clone = self.clone();
        let rhs_clone = rhs.clone();
        if backops.is_some() {
            backops.as_mut().unwrap().add_backward_op(move |grad| {
                let (grad_lhs, grad_rhs, grad_out): (
                    &mut Tensor<_, f32>,
                    &mut Tensor<_, f32>,
                    &Tensor<Ix2, f32>,
                ) = grad.mmr_grad(
                    (lhs_clone.id, lhs_clone.dim()),
                    (rhs_clone.id, rhs_clone.dim()),
                    out_id,
                );
                *grad_lhs += grad_out.dot(&rhs_clone.t());
                *grad_rhs += &lhs_clone.t().dot(grad_out);
            });
            out.put_backward_ops(backops);
        }
        out
    }
}

impl<A, B> Conv2d<TensorBase<Ix4, B>> for TensorBase<Ix3, A>
where
    A: DataBuffer<Item = f32> + 'static,
    B: DataBuffer<Item = f32> + 'static,
    // E: DataElement + 'static,
{
    type Output = Tensor<Ix3, f32>;

    fn conv2d(&self, kernels: &TensorBase<Ix4, B>, strides: (usize, usize)) -> Self::Output {
        let mut backops = merge_backward_ops(self, kernels);
        let (cin, cout, x, xs, k, ks, out, sx, sy, h, w, kh, kw, out_t) =
            conv2d_params(self, kernels, strides);
        conv2d(cin, cout, x, xs, k, ks, out, sx, sy, h, w, kh, kw);

        if backops.is_some() {
            let x_clone = self.clone();
            let kernels_clone = kernels.clone();
            let out_clone = out_t.clone();
            backops.as_mut().unwrap().add_backward_op(move |grad| {
                let (x_grad, k_grad, out_grad): (_, _, &Tensor<Ix3, f32>) = grad.mmr_grad(
                    (x_clone.id, x_clone.dim()),
                    (kernels_clone.id, kernels_clone.dim()),
                    out_clone.id,
                );
                let out_grad_view = out_grad.reshape([1, out_grad.dim[0], out_grad.dim[1], out_grad.dim[2]]);
                *k_grad += x_clone.conv2d(&out_grad_view, (1,1));
            })
        }

        out_t
    }
}
pub trait Matmul<Rhs> {
    type Output;
    fn matmul(&self, rhs: &Rhs) -> Self::Output;
}

pub trait Conv2d<Rhs> {
    type Output;
    fn conv2d(&self, rhs: &Rhs, strides: (usize, usize)) -> Self::Output;
}

impl<A> TensorBase<Ix2, A>
where
    A: DataBuffer<Item = f32>,
{
    #[inline]
    pub fn dot<B>(&self, rhs: &TensorBase<Ix2, B>) -> Tensor<Ix2, f32>
    where
        B: DataBuffer<Item = f32>,
    {
        assert!(self.shape()[1] == rhs.shape()[0]);
        let out_dim = [self.shape()[0], rhs.shape()[1]];
        let out_strides = generate_strides(&out_dim);
        let mut o = vec![0.; out_dim[0] * out_dim[1]];

        unsafe {
            sgemm(
                self.shape()[0],
                self.shape()[1],
                rhs.shape()[1],
                1.,
                self.ptr.as_ptr(),
                self.strides()[0] as isize,
                self.strides()[1] as isize,
                rhs.ptr.as_ptr(),
                rhs.strides()[0] as isize,
                rhs.strides()[1] as isize,
                0.,
                o.as_mut_ptr(),
                out_strides[0] as isize,
                out_strides[1] as isize,
            )
        }

        Tensor::from_vec(o, out_dim)
    }
}

pub fn conv2d_params<A, B>(
    lhs: &TensorBase<Ix3, A>,
    kernels: &TensorBase<Ix4, B>,
    strides: (usize, usize),
) -> (
    usize,
    usize,
    *const f32,
    (isize, isize, isize),
    *const f32,
    (isize, isize, isize, isize),
    *mut f32,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    Tensor<Ix3, f32>,
)
where
    A: DataBuffer<Item = f32>,
    B: DataBuffer<Item = f32>,
{
    assert!(lhs.ndim() >= 2);
    assert_eq!(lhs.dim[0], kernels.dim[1]);
    assert!(lhs.dim[1] > kernels.dim[2]);
    assert!(lhs.dim[2] > kernels.dim[3]);

    let cin = lhs.dim[0];
    let cout = kernels.dim[0];
    let x = lhs.ptr.as_ptr();
    let xs = (
        lhs.strides[0] as isize,
        lhs.strides[1] as isize,
        lhs.strides[2] as isize,
    );
    let k = kernels.ptr.as_ptr();
    let ks = (
        kernels.strides[0] as isize,
        kernels.strides[1] as isize,
        kernels.strides[2] as isize,
        kernels.strides[3] as isize,
    );
    let sx = strides.0;
    let sy = strides.1;
    let h = lhs.dim[1];
    let w = lhs.dim[2];
    let kh = kernels.dim[2];
    let kw = kernels.dim[3];

    let h_out = (h - kh) / sy + 1;
    let w_out = (w - kw) / sx + 1;
    let mut out_t: Tensor<_, f32> = Tensor::zeros([cout, h_out, w_out]);
    let out = unsafe { out_t.ptr.as_mut() };
    (cin, cout, x, xs, k, ks, out, sx, sy, h, w, kh, kw, out_t)
}

pub fn conv2d(
    cin: usize,
    cout: usize,
    x: *const f32,
    xs: (isize, isize, isize),
    k: *const f32,
    ks: (isize, isize, isize, isize),
    out: *mut f32,
    sx: usize,
    sy: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
) {
    let h_out = (h - kh) / sy + 1;
    let w_out = (w - kw) / sx + 1;
    for _c in 0..cout {
        for _h in 0..h_out {
            for _w in 0..w_out {
                let mut acc: f32 = 0.;
                for _cin in 0..cin {
                    for _kh in 0..kh {
                        for _kw in 0..kw {
                            let k_idx = _c as isize * ks.0
                                + _cin as isize * ks.1
                                + _kh as isize * ks.2
                                + _kw as isize * ks.3;
                            let x_idx = _cin as isize * xs.0
                                + (_h * sy + _kh) as isize * xs.1
                                + (_w * sx + _kw) as isize * xs.2;
                            let k_elem = unsafe { *k.offset(k_idx) };
                            let x_elem = unsafe { *x.offset(x_idx) };
                            acc += k_elem * x_elem;
                        }
                    }
                }
                let out_idx = (_c * h_out * w_out) + (_h * w_out) + _w;
                unsafe { out.add(out_idx).write(acc) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Conv2d, Matmul};
    use crate::{
        impl_constructors::{tensor, TensorConstructors},
        Tensor,
    };

    #[test]
    fn matmul_backward_ops_test() {
        let a = Tensor::ones([2, 2]);
        let b = Tensor::ones([2, 2]);

        assert!(a.backward_ops.borrow().is_none());
        assert!(b.backward_ops.borrow().is_none());

        let c = a.matmul(&b);

        assert_eq!(c, tensor([[2., 2.], [2., 2.]]));
        assert!(c.backward_ops.borrow().is_none());
    }

    #[test]
    fn conv() {
        let a = tensor([
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
            [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
        ]);
        let b = tensor([[
            [[1., 2.], [3., 4.]],
            [[1., 2.], [3., 4.]],
            [[1., 2.], [3., 4.]],
        ]]);
        let c = a.conv2d(&b, (1, 1));

        assert_eq!(c, tensor([[[37., 47.], [67., 77.]]]) * 3. as f32);
    }
}
