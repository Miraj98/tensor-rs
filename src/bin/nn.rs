use std::time::Instant;

use tensor_rs::{
    impl_binary_ops::TensorBinaryOps, impl_constructors::TensorConstructors,
    impl_processing_ops::Matmul, impl_reduce_ops::ReduceOps, impl_unary_ops::TensorUnaryOps,
    prelude::{GradientMap, BackwardOps}, Tensor, mnist::{mnist::MnistData, mnist, Dataloader},
};


pub fn flush_denormals_to_zero() {
    #[cfg(all(target_arch = "x86", target_feature = "sse"))]
    {
        use std::arch::x86::{_MM_FLUSH_ZERO_ON, _MM_SET_FLUSH_ZERO_MODE};
        unsafe { _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON) }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    {
        use std::arch::x86_64::{_MM_FLUSH_ZERO_ON, _MM_SET_FLUSH_ZERO_MODE};
        unsafe { _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON) }
    }
}

struct NN {
    w1: Tensor<[usize; 2], f32>,
    b1: Tensor<[usize; 2], f32>,
    w2: Tensor<[usize; 2], f32>,
    b2: Tensor<[usize; 2], f32>,
    mnsit: MnistData,
}

impl NN {
    pub fn new() -> Self {
        Self {
            w1: Tensor::randn([30, 784]).requires_grad(true),
            b1: Tensor::randn([30, 1]).requires_grad(true),
            w2: Tensor::randn([10, 30]).requires_grad(true),
            b2: Tensor::randn([10, 1]).requires_grad(true),
            mnsit: mnist::load_data(),
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

    fn mini_batch(&mut self, batch: Vec<(Tensor<[usize; 2], f32>, Tensor<[usize; 2], f32>)>) {
        let start = Instant::now();
        let mut total_backops_time = 0.;
        let lr = 2.0;
        let mut w1_grad: Tensor<[usize; 2], f32> = Tensor::zeros([30, 784]);
        let mut b1_grad: Tensor<[usize; 2], f32> = Tensor::zeros([30, 1]);
        let mut w2_grad: Tensor<[usize; 2], f32> = Tensor::zeros([10, 30]);
        let mut b2_grad: Tensor<[usize; 2], f32> = Tensor::zeros([10, 1]);
        let mut loss: Tensor<[usize; 0], f32> = Tensor::zeros([]);
        let alpha = lr / batch.len() as f32;

        for (i, o) in batch.iter() {
            let backprop_start = Instant::now();
            let (l, grad) = self.backprop(i, o);
            w1_grad += grad.grad(&self.w1);
            w2_grad += grad.grad(&self.w2);
            b1_grad += grad.grad(&self.b1);
            b2_grad += grad.grad(&self.b2);
            loss += l;
            *self.w1.backward_ops.borrow_mut() = Some(BackwardOps::new());
        }

        w1_grad *= alpha;
        w2_grad *= alpha;
        b1_grad *= alpha;
        b2_grad *= alpha;

        self.w1 -= w1_grad;
        self.w2 -= w2_grad;
        self.b1 -= b1_grad;
        self.b2 -= b2_grad;
        // println!("Backward ops time: {:?} secs", total_backops_time);
        // println!("Mini-batch completion time: {:?} secs", start.elapsed().as_secs_f64());
    }

    pub fn train(&mut self, batch_size: usize, epochs: usize) {
        let total_batches = self.mnsit.dataset_size as usize / batch_size;

        println!("Total batches: {total_batches}");

        for i in 0..epochs {
            let start = Instant::now();
            for j in 0..total_batches {
                let batch = self.mnsit.get_batch(batch_size, j);
                self.mini_batch(batch);
            }
            println!("Epoch: {} Time taken: {:?} ", i, start.elapsed().as_secs_f32());
        }

    }
}

fn main() {
    flush_denormals_to_zero();
    let mut nn = NN::new();
    nn.train(10, 30);
}
