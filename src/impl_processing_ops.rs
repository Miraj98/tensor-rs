use crate::{
    dim::Dimension,
    impl_constructors::TensorConstructors,
    impl_reduce_ops::ReduceOps,
    utils::{generate_strides, merge_backward_ops},
    DataBuffer, DataElement, Tensor, TensorBase,
};
use matrixmultiply::sgemm;

impl<A> Matmul<TensorBase<[usize; 2], A>> for TensorBase<[usize; 2], A>
where
    A: DataBuffer<Item = f32> + 'static,
{
    type Output = Tensor<[usize; 2], f32>;

    fn matmul(&self, rhs: &TensorBase<[usize; 2], A>) -> Self::Output {
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
                    &Tensor<[usize; 2], f32>,
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

impl<A, B, E> Conv2d<TensorBase<[usize; 4], B>> for TensorBase<[usize; 3], A>
where
    A: DataBuffer<Item = E>,
    B: DataBuffer<Item = E>,
    E: DataElement + 'static,
{
    type Output = Tensor<[usize; 3], E>;

    fn conv2d(&self, rhs: &TensorBase<[usize; 4], B>, strides: (usize, usize)) -> Self::Output {
        assert!(self.ndim() >= 2);
        let (sx, sy) = strides;
        let mut out_dim = [rhs.dim[0], 0, 0];
        out_dim[1] = (self.dim[1] - rhs.dim[2]) / sy + 1;
        out_dim[2] = (self.dim[2] - rhs.dim[3]) / sx + 1;
        let h = out_dim[1];
        let w = out_dim[2];
        let out: Tensor<_, E> = Tensor::zeros(out_dim);
        let out_ptr = out.ptr.as_ptr();

        for i in 0..rhs.dim[0] {
            for x in 0..w {
                for y in 0..h {
                    let slc = self.slice_2d(
                        (x * sx)..(rhs.dim[3] + x * sx),
                        (y * sy)..(rhs.dim[2] + y * sy),
                    );
                    let o = (slc * rhs.outer_dim(i)).sum();
                    unsafe { *out_ptr.add(i * x * y + y * w + x) = o.ptr.as_ptr().read() };
                }
            }
        }

        out
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

impl<A> TensorBase<[usize; 2], A>
where
    A: DataBuffer<Item = f32>,
{
    #[inline]
    pub fn dot<B>(&self, rhs: &TensorBase<[usize; 2], B>) -> Tensor<[usize; 2], f32>
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

        assert_eq!(c, tensor([[[37., 47.], [67., 77.]]]) * 3.);
    }
}
