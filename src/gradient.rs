use std::{collections::HashMap, any::Any};

use crate::unique_id::UniqueId;

pub trait Function {
    type Output;

    fn forward(&self) -> Self::Output;
    fn backward(&self);
}

pub struct GradientMap(HashMap<UniqueId, Box<dyn Any>>);

impl GradientMap {
    pub fn grad(&self) {
        todo!()
    }
}