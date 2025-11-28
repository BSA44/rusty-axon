use crate::engine::value::Node;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_state(&mut self);
}