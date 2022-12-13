use crate::{
    TensorBase,
    impl_binary_ops::TensorBinaryOps, impl_constructors::TensorConstructors,
    impl_processing_ops::{Matmul, Conv2d as Convolution2d}, DataBuffer, DataElement, Tensor,
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

pub struct Conv2d<E> 
where
    E: DataElement
{
    w: Tensor<[usize; 4], E>,
    b: Tensor<[usize; 4], E>,
    strides: (usize, usize),
}

impl<E> Conv2d<E>
where
    E: DataElement + 'static
{
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

    pub fn bias(&self) -> &Tensor<[usize; 4], E> {
        &self.b
    }

    pub fn weight(&self) -> &Tensor<[usize; 4], E> {
        &self.w
    }

    pub fn strides(&self) -> (usize, usize) {
        self.strides
    }
}

/*
impl<B> Layer<TensorBase<[usize; 3], B>> for Conv2d<f32>
where
    B: DataBuffer<Item = f32> + 'static,
{
    type Output = Option<i32>;
    fn forward(&self, input: TensorBase<[usize; 3], B>) -> Self::Output {
        for _ in 0..input.dim[0] {
            let _w = self.w.outer_dim(0);
            let out = input.conv2d(&_w, (1, 1));
            println!("{:?}", out);
        }
        None
    }
}
*/

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
