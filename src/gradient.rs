use crate::{prelude::{TensorBase, dim::Dimension, HasUniqueId, impl_constructors::TensorConstructors}, unique_id::UniqueId, num_taits::{One, Zero}};
use std::{any::Any, collections::HashMap, fmt::Debug};

pub struct BackwardOps(pub(crate) Vec<Box<dyn FnOnce(&mut GradientMap)>>);

pub trait Merge {
    fn merge(self, other: Self) -> Self;
}

impl Debug for BackwardOps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackwardOps")
            .field("num_operations", &self.0.len())
            .finish()
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

impl Merge for Option<BackwardOps> {
    fn merge(mut self, mut other: Self) -> Self {
        if self.is_none() && other.is_none() {
            Some(BackwardOps(Vec::new()))
        } else if self.is_some() && other.is_none() {
            self
        } else if self.is_none() && other.is_some() {
            other
        } else {
            self.as_mut().unwrap().append(other.as_mut().unwrap());
            self
        }
    }
}

pub struct GradientMap(pub(crate) HashMap<UniqueId, Box<dyn Any>>);

impl GradientMap {
    pub fn grad<T>(
        &self,
        t: &T
    ) -> &T where T: HasUniqueId + TensorConstructors + 'static {
        self.0.get(t.id()).unwrap().as_ref().downcast_ref().unwrap()
    }

    pub fn mut_grad<T>(&mut self, t: &T) -> &T where T: HasUniqueId + 'static {
        todo!()
        // self.0.entry(*t.id()).or_insert_with(|| Box::new(T::on))
    }
}
