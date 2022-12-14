use crate::{Tensor, dim::Ix2};

pub trait Dataloader {
    fn get_by_idx(&self, idx: usize) -> (Tensor<Ix2, f32>, Tensor<Ix2, f32>);
    fn get_batch(
        &self,
        batch_size: usize,
        batch_idx: usize,
    ) -> Vec<(Tensor<Ix2, f32>, Tensor<Ix2, f32>)>;
    fn size(&self) -> u16;
}

pub const PX_SIZE: usize = 28;

pub mod mnist {
    use std::{fs, time::Instant};
    use crate::{impl_constructors::TensorConstructors, Tensor, dim::Ix2};
    use super::{Dataloader, PX_SIZE};

    pub struct MnistData {
        raw_data: Vec<u8>,
        pub raw_labels_data: Vec<u8>,
        pub dataset_size: u16,
    }

    pub fn load_data() -> MnistData {
        MnistData {
            raw_data: fs::read("./src/mnist/mnist_data/train-images-idx3-ubyte").unwrap(),
            raw_labels_data: fs::read("./src/mnist/mnist_data/train-labels-idx1-ubyte").unwrap(),
            dataset_size: 60_000,
        }
    }

    impl Dataloader for MnistData {
        fn get_by_idx(&self, idx: usize) -> (Tensor<Ix2, f32>, Tensor<Ix2, f32>) {
            return (
                self.get_image_nn_input(idx),
                self.get_image_label_vector(idx),
            );
        }

        fn get_batch(
            &self,
            batch_size: usize,
            batch_idx: usize,
        ) -> Vec<(Tensor<Ix2, f32>, Tensor<Ix2, f32>)> {
            let _start = Instant::now();
            let mut b = Vec::<(Tensor<Ix2, f32>, Tensor<Ix2, f32>)>::new();

            if self.size() % batch_size as u16 != 0 {
                panic!("Batch size must be a whole factor of the total dataset size")
            }

            for i in 0..batch_size {
                let idx = batch_idx * batch_size + i;
                b.push(self.get_by_idx(idx))
            }

            b
        }

        fn size(&self) -> u16 {
            self.dataset_size
        }
    }

    impl MnistData {
        pub fn get_img_buffer(&self, idx: usize) -> &[u8] {
            &self.raw_data
                [(PX_SIZE * PX_SIZE * idx + 16)..(16 + idx * PX_SIZE * PX_SIZE + PX_SIZE * PX_SIZE)]
        }

        pub fn get_image_label(&self, idx: usize) -> u8 {
            self.raw_labels_data[idx + 8]
        }

        pub fn get_image_label_vector(&self, idx: usize) -> Tensor<Ix2, f32> {
            let out = Tensor::<_, f32>::zeros([10, 1]);
            let ptr = out.ptr.as_ptr();
            let offset = self.get_image_label(idx) as usize;
            unsafe {
                ptr.add(offset).write(1.);
            }
            return out;
        }

        pub fn get_image_nn_input(&self, idx: usize) -> Tensor<Ix2, f32> {
            let buf = self
                .get_img_buffer(idx)
                .to_vec()
                .iter()
                .map(|v| *v as f32 / 256.)
                .collect();
            Tensor::from_vec(buf, [PX_SIZE * PX_SIZE, 1])
        }
    }
}
