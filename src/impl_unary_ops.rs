use crate::{TensorBase, OwnedData, prelude::dim::Dimension, Tensor, DataElement};

pub trait TensorUnaryOps {
    fn sigmoid(&self) -> Self;
    fn tanh(&self) -> Self;
    fn relu(&self) -> Self;
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
                *input = input.clone() + (out_clone.map(|x| x * (1.0 - x)) * out);
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
                *input = input.clone() + out_clone.map(|x| (1.0 - x) * (1.0 - x)) * out;
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
                *input = input.clone() + out_clone.map(|x| if *x > 0. { 1.0 } else { 0. }) * out;
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
            let out_clone = o.clone();
            let lhs_id = self.id;
            let mut backward_ops = self.detach_backward_ops();
            backward_ops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, f64>, &Tensor<_, f64>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);
                *input = input.clone() + (out_clone.map(|x| x * (1.0 - x)) * out);
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
                let (input, out): (&mut Tensor<_, f64>, &Tensor<_, f64>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);
                *input = input.clone() + out_clone.map(|x| (1.0 - x) * (1.0 - x)) * out;
            });
            *o.backward_ops.borrow_mut() = backward_ops;
        }

        o
    }

    fn relu(&self) -> Self {
        let o = self.map(f64::relu);
        if self.backward_ops.borrow().is_some() {
            let mut backward_ops = self.detach_backward_ops();
            let out_clone = o.clone();
            let lhs_id = self.id;
            backward_ops.as_mut().unwrap().add_backward_op(move |grad| {
                let (input, out): (&mut Tensor<_, f64>, &Tensor<_, f64>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);
                *input = input.clone() + out_clone.map(|x| if *x > 0. { 1.0 } else { 0. }) * out;
            });
            *o.backward_ops.borrow_mut() = backward_ops;
        }

        o
    }
}