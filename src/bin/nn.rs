use tensor_rs::{
    impl_binary_ops::TensorBinaryOps, impl_constructors::TensorConstructors,
    impl_processing_ops::Matmul, impl_reduce_ops::ReduceOps, impl_unary_ops::TensorUnaryOps,
    prelude::GradientMap, Tensor,
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
        let mut o = (self.w1.dot(input) + (&self.b1)).sigmoid();
        o = (self.w2.dot(&o) + (&self.b2)).sigmoid();
        o
    }

    pub fn backprop(
        &self,
        input: &Tensor<[usize; 2], f32>,
        output: &Tensor<[usize; 2], f32>,
    ) -> (Tensor<[usize; 0], f32>, GradientMap) {
        let mut o = (self.w1.matmul(input).add(&self.b1)).sigmoid();
        o = (self.w2.matmul(&o).add(&self.b2)).sigmoid();
        o = o.sub(output);
        let loss = o.mul(&o).sum();
        let grad = loss.backward();
        (loss, grad)
    }

    pub fn mini_batch(&self, batch: Vec<(Tensor<[usize; 2], f32>, Tensor<[usize; 2], f32>)>) {
        let mut w1_grad: Tensor<[usize; 2], f32> = Tensor::zeros([30, 784]);
        let mut b1_grad: Tensor<[usize; 2], f32> = Tensor::zeros([30, 1]);
        let mut w2_grad: Tensor<[usize; 2], f32> = Tensor::zeros([10, 30]);
        let mut b2_grad: Tensor<[usize; 2], f32> = Tensor::zeros([10, 1]);

        for (i, o) in batch.iter() {
            let (loss, grad) = self.backprop(i, o);
            w1_grad += grad.grad(&self.w1);
            w2_grad += grad.grad(&self.w2);
            b1_grad += grad.grad(&self.b1);
            b2_grad += grad.grad(&self.b2);
        }
    }
}

fn main() {
    let nn = NN::new();
    let input = Tensor::randn([784, 1]);
    let output = Tensor::zeros([10, 1]);

    let (loss, _) = nn.backprop(&input, &output);

    println!("\nLoss\n{:?}", loss);

    // let mut o: Tensor<[usize; 2], f32> = Tensor::ones([10 ,1]);
    // let start = Instant::now();
    // for _ in 0..60_000 {
    //     o = nn.forward(&input);
    // }
    // println!("Time taken {:?} secs", start.elapsed().as_secs_f32());
    // println!("{:?}", o);
}
