use crate::{unique_id::UniqueId, prelude::{TensorBase}};
use std::{collections::HashMap, any::Any, fmt::Debug};

pub struct BackwardOps(pub(crate) Vec<Box<dyn FnOnce(&mut GradientMap)>>);

impl Debug for BackwardOps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackwardOps").field("num_operations", &self.0.len()).finish()
    }
}

impl BackwardOps {
    pub(crate) fn add_backward_op<F: 'static + FnOnce(&mut GradientMap)>(&mut self, operation: F) {
        self.0.push(Box::new(operation));
    }

    pub fn append(&mut self, other: &mut Self) {
        self.0.append(&mut other.0);
    }

    pub fn execute(mut self) -> GradientMap {
        let mut gradients: GradientMap = GradientMap(HashMap::<UniqueId, Box<dyn Any>>::new());
        for operation in self.0.drain(..).rev() {
            (operation)(&mut gradients);
        }
        gradients
    }
}

pub struct GradientMap(pub(crate) HashMap<UniqueId, Box<dyn Any>>);

impl GradientMap {
    pub fn grad<const D: usize, Dtype: 'static>(&self, t: &TensorBase<[usize; D], Dtype>) -> &TensorBase<[usize; D], Dtype> {
        self.0
        .get(t.id())
        .unwrap()
        .as_ref()
        .downcast_ref()
        .unwrap()
    }
}

