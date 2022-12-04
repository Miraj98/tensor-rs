use crate::{
    num_taits::{One, Zero},
    prelude::{dim::Dimension, impl_constructors::TensorConstructors, Tensor},
    unique_id::UniqueId,
};
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
    pub fn grad<S, Dtype>(&self, t: &Tensor<S, Dtype>) -> &Tensor<S, Dtype>
    where
        S: Dimension + 'static,
        Dtype: Zero + One + PartialEq + 'static,
    {
        self.0.get(t.id()).unwrap().as_ref().downcast_ref().unwrap()
    }

    pub fn grad_by_id<S, Dtype>(&self, t: UniqueId) -> &Tensor<S, Dtype>
    where
        S: Dimension + 'static,
        Dtype: Zero + One + PartialEq + 'static,
    {
        self.0.get(&t).unwrap().as_ref().downcast_ref().unwrap()
    }

    pub fn mut_grad<S, Dtype>(&mut self, t: &Tensor<S, Dtype>) -> &mut Tensor<S, Dtype>
    where
        S: Dimension + 'static,
        Dtype: Zero + One + PartialEq + 'static,
    {
        self.0
            .entry(*t.id())
            .or_insert_with(|| Box::new(Tensor::<_, Dtype>::zeros(t.dim())))
            .as_mut()
            .downcast_mut()
            .unwrap()
    }

    pub fn mut_grad_by_id_with_ones<S, Dtype>(&mut self, t: UniqueId, dim: S) -> &mut Tensor<S, Dtype>
    where
        S: Dimension + 'static,
        Dtype: Zero + One + PartialEq + 'static,
    {
        self.0
            .entry(t)
            .or_insert_with(|| Box::new(Tensor::<_, Dtype>::ones(dim)))
            .as_mut()
            .downcast_mut()
            .unwrap()
    }

    pub fn mut_grad_by_id<S, Dtype>(&mut self, t: UniqueId, dim: S) -> &mut Tensor<S, Dtype>
    where
        S: Dimension + 'static,
        Dtype: Zero + One + PartialEq + 'static,
    {
        self.0
            .entry(t)
            .or_insert_with(|| Box::new(Tensor::<_, Dtype>::zeros(dim)))
            .as_mut()
            .downcast_mut()
            .unwrap()
    }

    pub fn mr_grad<S, Dtype>(
        &mut self,
        l1: (UniqueId, S),
        l2: UniqueId,
    ) -> (&mut Tensor<S, Dtype>, &Tensor<S, Dtype>)
    where
        S: Dimension + 'static,
        Dtype: Zero + One + PartialEq + 'static,
    {
        let t1 = self.mut_grad_by_id(l1.0, l1.1) as *mut Tensor<S, Dtype>;
        let t2 = self.grad_by_id(l2) as *const Tensor<S, Dtype>;
        unsafe { (&mut *t1, &*t2) }
    }

    pub fn mmr_grad<L1, L2, L3, Dtype>(
        &mut self,
        l1: (UniqueId, L1),
        l2: (UniqueId, L2),
        l3: UniqueId,
    ) -> (
        &mut Tensor<L1, Dtype>,
        &mut Tensor<L2, Dtype>,
        &Tensor<L3, Dtype>,
    )
    where
        L1: Dimension + 'static,
        L2: Dimension + 'static,
        L3: Dimension + 'static,
        Dtype: Zero + One + PartialEq + 'static,
    {
        let t1 = self.mut_grad_by_id(l1.0, l1.1) as *mut Tensor<L1, Dtype>;
        let t2 = self.mut_grad_by_id(l2.0, l2.1) as *mut Tensor<L2, Dtype>;
        let t3 = self.grad_by_id(l3) as *const Tensor<L3, Dtype>;
        unsafe { (&mut *t1, &mut *t2, &*t3) }
    }
}
