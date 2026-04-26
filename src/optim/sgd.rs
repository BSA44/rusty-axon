//! Stochastic gradient descent optimizer placeholder.

use std::collections::HashSet;

use crate::engine::value::Node;

use crate::optim::optimizer::Optimizer;
/// Classic stochastic gradient descent optimizer.
pub struct Sgd {
    learning_rate: f32,
    parameters: Vec<Node>,
}

impl Sgd {
    /// Create a new SGD optimizer.
    pub fn new(learning_rate: f32, parameters: Vec<Node>) -> Self {
        Self {
            learning_rate,
            parameters,
        }
    }
}

impl Optimizer for Sgd {
    /// Apply one optimization step over the provided parameters.
    fn step(&mut self) {
        for param in self.parameters.iter_mut() {
            param.set_value(param.get_value() - self.learning_rate * param.get_gradient());
        }
    }

    /// Reset gradients in preparation for a new backward pass.
    ///
    /// `Param`-view Nodes route their gradient into a `MatMulTape`'s flat
    /// buffers; resetting them one Node at a time would call `reset_grads()`
    /// `in_dim*out_dim + out_dim` times per layer.  Dedup by tape pointer so
    /// each tape is reset exactly once; `Owned` Nodes are zeroed directly.
    fn zero_state(&mut self) {
        let mut seen_tapes: HashSet<usize> = HashSet::new();
        for param in self.parameters.iter() {
            match param.param_tape_ptr() {
                Some(ptr) => {
                    if seen_tapes.insert(ptr as usize) {
                        param.reset_param_tape();
                    }
                }
                None => param.zero_gradient(),
            }
        }
    }
}
