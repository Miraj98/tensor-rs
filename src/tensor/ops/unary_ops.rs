use crate::prelude::{dim::Dimension, Data, OwnedData, Tensor, TensorBase};

pub trait TensorUnaryOps {
    fn sigmoid(&self) -> Self;
    fn tanh(&self) -> Self;
    fn relu(&self) -> Self;
}

impl<S> TensorUnaryOps for TensorBase<S, OwnedData<f32>>
where
    S: Dimension,
{
    fn sigmoid(&self) -> Self {
        let o = self.map(|x| 1.0 / (1.0 + (-x).exp()));
        if self.backward_ops.borrow().is_some() {
            let out_clone = o.clone();
            let lhs_id = self.id;
            let backward_ops = self.detach_backward_ops();
            backward_ops.as_mut().unwrap().add_backward_op(move |grad| {
                let (mut input, out): (&mut Tensor<_, f32>, &Tensor<_, f32>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);
                *input = input + out * out_clone.map(|x| x * (1.0 - x));
            });
            o.backward_ops.replace(Some(backward_ops));
        }
    }

    fn tanh(&self) -> Self {
        let o = self.map(|x| x.tanh());
        if self.backward_ops.borrow().is_some() {
            let out_clone = o.clone();
            let lhs_id = self.id;
            let backward_ops = self.detach_backward_ops();
            backward_ops.as_mut().unwrap().add_backward_op(move |grad| {
                let (mut input, out): (&mut Tensor<_, f32>, &Tensor<_, f32>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);
                *input = input + out * out_clone.map(|x| (1.0 - x) * (1.0 - x));
            });
            o.backward_ops.replace(Some(backward_ops));
        }
    }

    fn relu(&self) -> Self {
        let o = self.map(|x| x.max(0.0));
        if self.backward_ops.borrow().is_some() {
            let backward_ops = self.detach_backward_ops();
            let out_clone = o.clone();
            let lhs_id = self.id;
            backward_ops.as_mut().unwrap().add_backward_op(move |grad| {
                let (mut input, out): (&mut Tensor<_, f32>, &Tensor<_, f32>) =
                    grad.mr_grad((lhs_id, out_clone.dim()), out_clone.id);
                *input = input + out * out_clone.map(|x| if x > 0. {1.0} else {0.});
            });
            o.backward_ops.replace(Some(backward_ops));
        }
    }
}
