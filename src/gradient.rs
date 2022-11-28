use crate::{unique_id::UniqueId};
use std::{collections::HashMap, any::Any};

pub struct BackwardOps<const D: usize>(Vec<Box<dyn FnOnce(&mut GradientMap)>>);

impl<const D: usize> BackwardOps<D> {
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
    pub fn grad(&self) {
        todo!()
    }
}

