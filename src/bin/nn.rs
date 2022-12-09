use std::time::Instant;

use tensor_rs::{
    impl_binary_ops::TensorBinaryOps, impl_constructors::TensorConstructors,
    impl_processing_ops::Matmul, impl_unary_ops::TensorUnaryOps, Tensor,
};

struct NN {
    w1: Tensor<[usize; 2], f32>,
    b1: Tensor<[usize; 2], f32>,
    w2: Tensor<[usize; 2], f32>,
    b2: Tensor<[usize; 2], f32>,
}

impl NN {
    pub fn new() -> Self {
        Self {
            w1: Tensor::randn([30, 784]).requires_grad(true),
            b1: Tensor::randn([30, 1]).requires_grad(true),
            w2: Tensor::randn([10, 30]).requires_grad(true),
            b2: Tensor::randn([10, 1]).requires_grad(true),
        }
    }

    pub fn forward(&self, input: &Tensor<[usize; 2], f32>) -> Tensor<[usize; 2], f32> {
        let mut o = self.w1.matmul(input).add(&self.b1).sigmoid();
        o = self.w2.matmul(&o).add(&self.b2).sigmoid();
        o
    }
}

fn main() {
    let nn = NN::new();
    let input = Tensor::randn([784, 1]);

    let mut o: Tensor<[usize; 2], f32> = Tensor::ones([10 ,1]);
    let start = Instant::now();
    for _ in 0..30 {
        o = nn.forward(&input);
    }
    println!("{:?}", o);
    println!("Time taken {:?} secs", start.elapsed().as_secs_f32());
}
