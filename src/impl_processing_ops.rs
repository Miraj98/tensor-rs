use crate::{
    dim::{Ix2, Ix3, Ix4, ShapePattern},
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
{
    type Output = Tensor<Ix3, f32>;

    fn conv2d(&self, kernels: &TensorBase<Ix4, B>, strides: (usize, usize)) -> Self::Output {
        let mut backops = merge_backward_ops(self, kernels);
        let (cin, cout, x, xs, k, ks, out, sx, sy, h, w, kh, kw, out_t) =
            conv2d_params(self, kernels, strides);
        unsafe {
            conv2d(cin, cout, x, xs, k, ks, out, sx, sy, h, w, kh, kw);
        }

        if backops.is_some() {
            let x_clone = self.clone();
            let kernels_clone = kernels.clone();
            let out_clone = out_t.clone();
            backops.as_mut().unwrap().add_backward_op(move |grad| {
                let (x_g, k_g, out_g): (_, _, &Tensor<Ix3, f32>) = grad.mmr_grad(
                    (x_clone.id, x_clone.dim()),
                    (kernels_clone.id, kernels_clone.dim()),
                    out_clone.id,
                );

                let kg_local: Tensor<[usize; 4], f32> = Tensor::zeros(kernels_clone.dim());
                let xg_local: Tensor<[usize; 3], f32> = Tensor::zeros(x_clone.dim());
                let x_ptr = x_clone.ptr.as_ptr();
                let out_g_ptr = out_g.ptr.as_ptr();
                let xs = x_clone.strides.ipattern();
                let ks = out_g.strides.ipattern();

                // Calculate kg_local(kernel gradients)
                unsafe { 
                    let h = x_clone.dim[1];
                    let w = x_clone.dim[2];
                    let kh = out_g.dim[1];
                    let kw = out_g.dim[2];

                    for _cout in 0..out_g.dim[0] as isize {
                        let k = out_g_ptr.offset(_cout * ks.0);
                        let out = kg_local.ptr.as_ptr().offset(_cout * kg_local.strides[0] as isize);
                        for _cin in 0..x_clone.dim[0] as isize {
                            let x = x_ptr.offset(_cin * xs.0);
                            conv2d(1, 1, x, xs, k, (0, ks.0, ks.1, ks.2), out.offset(_cin * kg_local.strides[1] as isize), sx, sy, h, w, kh, kw);
                        }
                    }
                }

                // Calculate xg_local(input gradients)
                unsafe {
                    let kh = out_g.dim[out_g.dim.len() - 2];
                    let kw = out_g.dim[out_g.dim.len() - 1];
                    let h = kernels_clone.dim[kernels_clone.dim.len() - 2] + 2*(kh - 1);
                    let w = kernels_clone.dim[kernels_clone.dim.len() - 1] + 2*(kw - 1);
                    let mut x_padded: Tensor<_, f32> = Tensor::zeros([kernels_clone.dim[0], kernels_clone.dim[1], h, w]);
                    x_padded
                        .slice_mut_2d(kw - 1..kw - 1 + kernels_clone.dim[kernels_clone.dim.len() - 1], kh - 1..kh - 1 + kernels_clone.dim[kernels_clone.dim.len() - 2])
                        .assign(&kernels_clone);

                    for _cout in 0..kernels_clone.dim[0] {
                        let kernel_view_3d = kernels_clone.outer_dim(_cout);
                        let out_g_view_2d = out_g.outer_dim(_cout);

                        for _cin in 0..xg_local.dim[0] {
                            let kernel_view_2d = kernel_view_3d.outer_dim(_cin);
                            let x = kernel_view_2d.ptr.as_ptr();
                            let xs = kernel_view_3d.strides.ipattern();
                            let k = out_g_view_2d.ptr.as_ptr();
                            let ks = out_g.strides.ipattern();
                            let out = xg_local.ptr.as_ptr().add(_cin * xg_local.strides[0]);
                            conv2d(1, 1, x, xs, k, (0, ks.0, ks.1, ks.2), out, sx, sy, h, w, kh, kw)
                        }
                    }
                }

                *k_g += kg_local;
                *x_g += xg_local;
            })
        }
        out_t.put_backward_ops(backops);
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
    assert!(lhs.dim[1] >= kernels.dim[2]);
    assert!(lhs.dim[2] >= kernels.dim[3]);

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

pub unsafe fn conv2d(
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
    for _c in 0..cout as isize {
        for _h in 0..h_out as isize {
            for _w in 0..w_out as isize {
                let mut acc: f32 = 0.;
                for _cin in 0..cin as isize {
                    for _kh in 0..kh as isize {
                        for _kw in 0..kw as isize {
                            let k_idx = _c * ks.0 + _cin * ks.1 + _kh * ks.2 + _kw * ks.3;
                            let x_idx = _cin * xs.0 + (_h * (sy as isize) + _kh) * xs.1 + (_w * (sx as isize) + _kw) * xs.2;
                            let k_elem = unsafe { *k.offset(k_idx) };
                            let x_elem = unsafe { *x.offset(x_idx) };
                            acc += k_elem * x_elem;
                        }
                    }
                }
                let out_idx = (_c * (h_out as isize) * (w_out as isize)) + (_h * (w_out as isize)) + _w;
                unsafe { out.offset(out_idx).write(out.offset(out_idx).read() + acc) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Conv2d, Matmul};
    use crate::{
        impl_constructors::{tensor, TensorConstructors},
        impl_processing_ops::{conv2d, conv2d_params},
        impl_reduce_ops::ReduceOps,
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
    fn conv_normal() {
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

    #[test]
    fn conv_partial() {
        let x_t = tensor([
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]],
            [[9., 10.], [11., 12.]],
        ]);
        let k_t: Tensor<_, f32> = Tensor::ones(x_t.dim());
        let k_t4 = k_t.reshape([1, k_t.dim[0], k_t.dim[1], k_t.dim[2]]);

        let (_, _, x, xs, k, ks, out, sx, sy, h, w, kh, kw, out_t) =
            conv2d_params(&x_t, &k_t4, (1, 1));
        unsafe {
            conv2d(1, 1, x, xs, k, ks, out, sx, sy, h, w, kh, kw);
        }

        assert_eq!(out_t, tensor([[[10.]]]));
    }

    #[test]
    fn conv_backward() {
        let a = tensor([
            [
                [1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.],
            ],[
                [2., 2., 2.],
                [2., 2., 2.],
                [2., 2., 2.],
            ],[
                [3., 3., 3.],
                [3., 3., 3.],
                [3., 3., 3.],
            ],
        ]).requires_grad(true);
        let b = tensor([[
            [
                [4., 4.],
                [4., 4.]
            ],
            [
                [5., 5.],
                [5., 5.]
            ],
            [
                [6., 6.],
                [6., 6.]
            ],
        ], [
            [
                [6., 6.],
                [6., 6.]
            ],
            [
                [7., 7.],
                [7., 7.]
            ],
            [
                [8., 8.],
                [8., 8.]
            ],
        ]]).requires_grad(true);
        let c = a.conv2d(&b, (1, 1));
        let c_sum = c.sum();
        let gradients = c_sum.backward();
        let b_g = gradients.grad(&b);
        let a_g = gradients.grad(&a);
        println!("{a_g:?}");
        assert_eq!(b_g, &tensor([[
            [
                [4., 4.],
                [4., 4.]
            ],
            [
                [8., 8.],
                [8., 8.]
            ],
            [
                [12., 12.],
                [12., 12.]
            ],
        ], [
            [
                [4., 4.],
                [4., 4.]
            ],
            [
                [8., 8.],
                [8., 8.]
            ],
            [
                [12., 12.],
                [12., 12.]
            ],
        ]]));
    }
}
