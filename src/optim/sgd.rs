//! Stochastic gradient descent optimizer placeholder.

use crate::engine::value::Node;

use crate::optim::optimizer::Optimizer;
/// Classic stochastic gradient descent optimizer.
pub struct Sgd {
    learning_rate: f64,
    parameters: Vec<Node>,
}

impl Sgd {
    /// Create a new SGD optimizer.
    pub fn new(learning_rate: f64, parameters: Vec<Node>) -> Self {
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

    /// Reset optimizer state (e.g., momentum buffers).
    fn zero_state(&mut self) {
        for param in self.parameters.iter_mut() {
            param.zero_gradient();
        }
    }
}
