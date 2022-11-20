use ndarray::{Dimension, Array};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

pub struct TensorInner<A, D> where A: Clone, D: Dimension {
    data: RefCell<Array<A, D>>,
    grad: RefCell<Option<Array<A, D>>>,
    requires_grad: Cell<bool>,
}

pub struct TensorOuter<A, D1, D2> where A: Clone, D1: Dimension, D2: Dimension {
    inner: Rc<TensorInner<A, D1>>,
    saved_tensors: Option<Rc<TensorInner<A, D2>>>,
}

impl<A, D> TensorOuter<A, D, D> where A: Clone, D: Dimension {
    pub fn new(a: Array<A, D>, requires_grad: bool) -> TensorOuter<A, D, D> {
        return TensorOuter {
            inner: Rc::new(TensorInner {
                data: RefCell::new(a),
                grad: RefCell::new(None), 
                requires_grad: Cell::new(requires_grad)
            }),
            saved_tensors: None,
        }
    }
}

pub type TensorBase<A, D> = Rc<TensorInner<A, D>>;

pub trait Add<A: Clone, D: Dimension> {
    fn add(&self, a: TensorBase<A, D>) -> TensorBase<A, D>;
}

#[cfg(test)]
mod tests {
}
