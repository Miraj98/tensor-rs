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
            *grad_lhs = grad_lhs.clone() + grad_out.dot(&rhs_clone.t());
            *grad_rhs = grad_rhs.clone() + &lhs_clone.t().dot(grad_out);
        });

        out.put_backward_ops(backops);
        out
    }
}

impl<S, A, B, E> Conv2d<TensorBase<S, B>> for TensorBase<S, A>
where
    S: Dimension + 'static,
    A: DataBuffer<Item = E>,
    B: DataBuffer<Item = E>,
    E: DataElement + 'static,
{
    type Output = Tensor<S, E>;

    fn conv2d(&self, rhs: &TensorBase<S, B>, strides: (usize, usize)) -> Self::Output {
        assert!(self.ndim() >= 2);
        let (sx, sy) = strides;
        let c = self.ndim();
        let mut out_dim = self.dim();
        out_dim[c - 2] = (self.dim[c - 2] - rhs.dim[c - 2]) / sy + 1;
        out_dim[c - 1] = (self.dim[c - 1] - rhs.dim[c - 1]) / sx + 1;
        let h = out_dim[c - 2];
        let w = out_dim[c - 1];
        let out: Tensor<_, E> = Tensor::zeros(out_dim);
        let out_ptr = out.ptr.as_ptr();

        for x in 0..w {
            for y in 0..h {
                let slc = self.slice_2d(
                    (x * sx)..(rhs.dim[c - 1] + x * sx),
                    (y * sy)..(rhs.dim[c - 2] + y * sy),
                );
                let o = (slc * rhs).sum();
                unsafe { *out_ptr.add(y * w + x) = o.ptr.as_ptr().read() };
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
    pub fn dot<B>(&self, rhs: &TensorBase<[usize; 2], B>) -> Tensor<[usize; 2], f32>
    where
        B: DataBuffer<Item = f32>,
    {
        let a = self.shape()[1];
        let b = self.shape()[1];
        println!("a: {}, b: {}", a, b);
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
    use crate::impl_constructors::tensor;

    use super::Conv2d;

    #[test]
    fn conv() {
        let a = tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let b = tensor([[1., 2.], [3., 4.]]);
        let c = a.conv2d(&b, (1, 1));

        assert_eq!(c, tensor([[37., 47.], [67., 77.]]));
    }
}
