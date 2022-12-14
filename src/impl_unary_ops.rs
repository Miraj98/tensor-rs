use crate::{dim::Dimension, DataElement, OwnedData, Tensor, TensorBase};

pub trait TensorUnaryOps {
    fn sigmoid(&self) -> Self;
    fn tanh(&self) -> Self;
    fn relu(&self) -> Self;
    fn square(&self) -> Self;
    fn ln(&self) -> Self;
}

impl<S> TensorUnaryOps for TensorBase<S, OwnedData<f32>>
where
    S: Dimension + 'static,
{
    fn sigmoid(&self) -> Self {
        let o = self.map(f32::sigmoid);

        if self.backward_ops.borrow().is_some() {
            let out_clone = o.clone();
            let lhs_id = self.id;
            let mut backward_ops = self.detach_backward_ops();
            backward_ops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, f32>, &Tensor<_, f32>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);
                *input += out_clone.map(|x| x * (1.0 - x)) * out;
            });
            *o.backward_ops.borrow_mut() = backward_ops;
        }
        o
    }

    fn square(&self) -> Self {
        let o = self.map(|x| *x * *x);

        if self.backward_ops.borrow().is_some() {
            let out_clone = o.clone();
            let self_clone = self.clone();
            let lhs_id = self.id;
            let mut backward_ops = self.detach_backward_ops();
            backward_ops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, f32>, &Tensor<_, f32>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);
                let mut local = self_clone.map(|x| 2. * x);
                local *= out;
                *input += local;
            });
            *o.backward_ops.borrow_mut() = backward_ops;
        }
        o
    }

    fn ln(&self) -> Self {
       let o = self.map(|x| x.ln());

        if self.backward_ops.borrow().is_some() {
            let out_clone = o.clone();
            let self_clone = self.clone();
            let lhs_id = self.id;
            let mut backward_ops = self.detach_backward_ops();
            backward_ops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, f32>, &Tensor<_, f32>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);
                let mut local = self_clone.map(|x| 1. / x);
                local *= out;
                *input += local;
            });
            *o.backward_ops.borrow_mut() = backward_ops;
        }
        o
    }

    fn tanh(&self) -> Self {
        let o = self.map(|x| x.tanh());
        if self.backward_ops.borrow().is_some() {
            let out_clone = o.clone();
            let lhs_id = self.id;
            let mut backward_ops = self.detach_backward_ops();
            backward_ops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, f32>, &Tensor<_, f32>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);
                *input += out_clone.map(|x| (1.0 - x) * (1.0 - x)) * out;
            });
            *o.backward_ops.borrow_mut() = backward_ops;
        }

        o
    }

    fn relu(&self) -> Self {
        let o = self.map(f32::relu);
        if self.backward_ops.borrow().is_some() {
            let mut backward_ops = self.detach_backward_ops();
            let out_clone = o.clone();
            let lhs_id = self.id;
            backward_ops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, f32>, &Tensor<_, f32>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);
                *input += out_clone.map(|x| if *x > 0. { 1.0 } else { 0. }) * out;
            });
            *o.backward_ops.borrow_mut() = backward_ops;
        }

        o
    }
}

impl<S> TensorUnaryOps for TensorBase<S, OwnedData<f64>>
where
    S: Dimension + 'static,
{
    fn sigmoid(&self) -> Self {
        let o = self.map(f64::sigmoid);

        if self.backward_ops.borrow().is_some() {
            let mut out_clone = o.clone();
            let lhs_id = self.id;
            let mut backward_ops = self.detach_backward_ops();
            backward_ops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, f64>, &Tensor<_, f64>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);
                out_clone.map_inplace(|x| x * (1.0 - x));
                out_clone *= out;
                *input += out_clone;
            });
            *o.backward_ops.borrow_mut() = backward_ops;
        }
        o
    }

    fn square(&self) -> Self {
        let o = self.map(|x| *x * *x);

        if self.backward_ops.borrow().is_some() {
            let out_clone = o.clone();
            let self_clone = self.clone();
            let lhs_id = self.id;
            let mut backward_ops = self.detach_backward_ops();
            backward_ops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, f64>, &Tensor<_, f64>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);
                let mut local = self_clone.map(|x| 2. * x);
                local *= out;
                *input += local;
            });
            *o.backward_ops.borrow_mut() = backward_ops;
        }
        o
    }

    fn ln(&self) -> Self {
        let o = self.map(|x| x.ln());
        if self.backward_ops.borrow().is_some() {
            let out_clone = o.clone();
            let self_clone = self.clone();
            let lhs_id = self.id;
            let mut backward_ops = self.detach_backward_ops();
            backward_ops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, f64>, &Tensor<_, f64>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);
                let mut local = self_clone.map(|x| 1. / x);
                local *= out;
                *input += local;
            });
            *o.backward_ops.borrow_mut() = backward_ops;
        }
        o
    }

    fn tanh(&self) -> Self {
        let o = self.map(|x| x.tanh());
        if self.backward_ops.borrow().is_some() {
            let mut out_clone = o.clone();
            let lhs_id = self.id;
            let mut backward_ops = self.detach_backward_ops();
            backward_ops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, f64>, &Tensor<_, f64>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);

                out_clone.map_inplace(|x| (1.0 - x) * (1.0 - x));
                out_clone *= out;
                *input += out_clone;
            });
            *o.backward_ops.borrow_mut() = backward_ops;
        }

        o
    }

    fn relu(&self) -> Self {
        let o = self.map(f64::relu);
        if self.backward_ops.borrow().is_some() {
            let mut backward_ops = self.detach_backward_ops();
            let mut out_clone = o.clone();
            let lhs_id = self.id;
            backward_ops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, f64>, &Tensor<_, f64>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);
                out_clone.map_inplace(|x| if *x > 0. { 1.0 } else { 0. });
                out_clone *= out;
                *input += out_clone;
            });
            *o.backward_ops.borrow_mut() = backward_ops;
        }

        o
    }
}

#[cfg(test)]
mod tests {
    use crate::{impl_constructors::tensor, impl_reduce_ops::ReduceOps, Tensor};

    use super::TensorUnaryOps;

    #[test]
    fn square() {
        let a: Tensor<_, f32> = tensor([[2., 3.], [3. , 4.]]).requires_grad(true);
        let b = a.square();
        let c = b.sum();
        let g = c.backward();

        assert_eq!(tensor([[4., 6.], [6., 8.]]), g.grad(&a), "square grad");
    }
}
