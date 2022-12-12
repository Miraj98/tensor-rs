use crate::{
    impl_binary_ops::TensorBinaryOps, impl_constructors::TensorConstructors,
    impl_processing_ops::Matmul, DataElement, Tensor,
};

pub trait Layer<Input> {
    type Output;

    fn forward(&self, input: Input) -> Self::Output;
}

pub struct Linear<A>
where
    A: DataElement,
{
    w: Tensor<[usize; 2], A>,
    b: Tensor<[usize; 2], A>,
}

impl<A> Linear<A>
where
    A: DataElement,
{
    pub fn new(dim: [usize; 2]) -> Self {
        Self {
            w: Tensor::randn(dim).requires_grad(true),
            b: Tensor::randn([dim[0], 1]).requires_grad(true),
        }
    }

    pub fn bias(&self) -> &Tensor<[usize; 2], A> {
        &self.b
    }

    pub fn weight(&self) -> &Tensor<[usize; 2], A> {
        &self.w
    }
}

impl Layer<&Tensor<[usize; 2]>> for Linear<f32> {
    type Output = Tensor<[usize; 2]>;

    fn forward(&self, input: &Tensor<[usize; 2]>) -> Self::Output {
        self.w.matmul(input).add(&self.b)
    }
}

pub struct Conv2d {
    w: Tensor<[usize; 4]>,
    b: Tensor<[usize; 4]>,
    strides: (usize, usize),
}

impl Conv2d {
    pub fn new(
        input_channels: usize,
        output_channels: usize,
        kernel_size: (usize, usize),
        strides: (usize, usize),
    ) -> Self {
        Self {
            w: Tensor::randn([output_channels, input_channels, kernel_size.0, kernel_size.1]).requires_grad(true),
            b: Tensor::randn([output_channels, 1, 1, 1]).requires_grad(true),
            strides,
        }
    }

    pub fn bias(&self) -> &Tensor<[usize; 4]> {
        &self.b
    }

    pub fn weight(&self) -> &Tensor<[usize; 4]> {
        &self.w
    }

    pub fn strides(&self) -> (usize, usize) {
        self.strides
    }
}

#[cfg(test)]
mod tests {
    use crate::{impl_reduce_ops::ReduceOps, Tensor};

    use super::{Layer, Linear};

    #[test]
    fn linear() {
        let m = Linear::new([30, 2]);
        let a = Tensor::from_vec(vec![1., 2.], [2, 1]);
        let o = m.forward(&a);
        let loss = o.sum();
        let g = loss.backward();
        let w_grad = g.grad(m.weight());
        let b_grad = g.grad(m.bias());
        println!("{:?}", w_grad);
        println!("{:?}", b_grad);
    }
}
