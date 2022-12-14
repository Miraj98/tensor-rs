use crate::{
    dim::{Ix2, Ix3, Ix4},
    impl_constructors::TensorConstructors,
    impl_reduce_ops::ReduceOps,
    utils::{generate_strides, merge_backward_ops},
    DataBuffer, DataElement, Tensor, TensorBase,
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
        assert!(self.ndim() >= 2);
        assert_eq!(self.dim[0], kernels.dim[1]);
        assert!(self.dim[1] > kernels.dim[2]);
        assert!(self.dim[2] > kernels.dim[3]);

        let cin = self.dim[0];
        let cout = kernels.dim[0];
        let x = self.ptr.as_ptr();
        let xs = (self.strides[0] as isize, self.strides[1] as isize, self.strides[2] as isize);
        let k = kernels.ptr.as_ptr();
        let ks =  (kernels.strides[0] as isize, kernels.strides[1] as isize, kernels.strides[2] as isize, kernels.strides[3] as isize);
        let sx = strides.0;
        let sy = strides.1;
        let h = self.dim[1];
        let w = self.dim[2];
        let kh = kernels.dim[2];
        let kw = kernels.dim[3];

        let h_out = (h - kh) / sy + 1;
        let w_out = (w - kw) / sx + 1;
        let mut out_t: Tensor<_, f32> = Tensor::zeros([cout, h_out, w_out]);
        let out = unsafe { out_t.ptr.as_mut() };

        conv2d(cin, cout, x, xs, k, ks, out, sx, sy, h, w, kh, kw);

        // let (sx, sy) = strides;
        // let mut out_dim = [kernels.dim[0], 0, 0];
        // out_dim[1] = (self.dim[1] - kernels.dim[2]) / sy + 1;
        // out_dim[2] = (self.dim[2] - kernels.dim[3]) / sx + 1;
        // let h = out_dim[1];
        // let w = out_dim[2];
        // let out: Tensor<_, E> = Tensor::zeros(out_dim);
        // let out_ptr = out.ptr.as_ptr();

        // for i in 0..kernels.dim[0] {
        //     for x in 0..w {
        //         for y in 0..h {
        //             let slc = self.slice_2d(
        //                 (x * sx)..(kernels.dim[3] + x * sx),
        //                 (y * sy)..(kernels.dim[2] + y * sy),
        //             );
        //             let o = (slc * kernels.outer_dim(i)).sum();
        //             unsafe { *out_ptr.add(i * x * y + y * w + x) = o.ptr.as_ptr().read() };
        //         }
        //     }
        // }

        // let mut backops = merge_backward_ops(self, kernels);
        // if backops.is_some() {
        //     let self_clone = self.clone();
        //     let mut kernels_clone = kernels.clone();
        //     let out_clone = out.clone();
        //     // TODO: Optimize the backward pass
        //     backops.as_mut().unwrap().add_backward_op(move |grad| {
        //         let (lhs_grad, rhs_grad, out_grad): (
        //             &mut TensorBase<_, _>,
        //             &mut TensorBase<_, _>,
        //             &Tensor<[usize; 3], _>,
        //         ) = grad.mmr_grad(
        //             (self_clone.id, self_clone.dim()),
        //             (kernels_clone.id, kernels_clone.dim()),
        //             out_clone.id,
        //         );
        //         let px = kernels_clone.dim[3] - 1;
        //         let py = kernels_clone.dim[2] - 1;
        //         let mut padded = Tensor::<_, E>::zeros([
        //             out_grad.dim[0],
        //             2 * py + self_clone.dim[1],
        //             2 * px + self_clone.dim[2],
        //         ]);
        //         padded
        //             .slice_mut_2d(px..px + out_grad.dim[1], py..py + out_grad.dim[2])
        //             .assign(out_grad);
        //         kernels_clone.invert_axis(2);
        //         kernels_clone.invert_axis(3);
        //         *lhs_grad += padded.conv2d(&kernels_clone, (1, 1));
        //         // *rhs_grad +=
        //     })
        // }

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

    // out_dim = [cout, h_out, w_out]

    for _c in 0..cout {
        for _h in 0..h_out {
            for _w in 0..w_out {
                let mut acc: f32 = 0.;
                for _cin in 0..cin {
                    for _kh in 0..kh {
                        for _kw in 0..kw {
                            let k_idx = _c as isize * ks.0 + _cin as isize * ks.1 + _kh as isize * ks.2 + _kw as isize * ks.3;
                            let x_idx = _cin as isize * xs.0 + (_h * sy + _kh) as isize * xs.1 + (_w * sx + _kw) as isize *xs.2;
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

    // for c in 0..cout {
    //     for i in 0..cin {
    //         for j in 0..h_out {
    //             for k in 0..w_out {
    //                 let mut sum = 0.;
    //                 for p in 0..kh {
    //                     for q in 0..kw {
    //                         let x_idx = k * xs.0 * xs.1 + (i * sy + p) * xs.1 + j * sx + q;
    //                         let k_idx = c * ks.0 * ks.1 * ks.2 + k * ks.1 * ks.2 + p * ks.2 + q;
    //                         sum += unsafe { *x.add(x_idx) } * unsafe { *k.add(k_idx) };
    //                     }
    //                 }
    //             }
    //             let out_idx = c * h * w + i * w + j;
    //             unsafe { *out.add(out_idx) = sum };
    //         }
    //     }
    // }
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
