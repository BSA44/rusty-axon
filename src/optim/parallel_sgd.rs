//! Parallel Stochastic Gradient Descent optimizer.
//!
//! Uses parallel iteration for parameter updates, which can provide
//! speedups for networks with large parameter counts (1000+).

use rayon::prelude::*;
use crate::engine::value::Node;
use crate::optim::optimizer::Optimizer;

/// Parallel SGD optimizer that updates parameters in parallel.
///
/// Best for networks with many parameters (1000+). For smaller networks,
/// the overhead of parallel execution may not be worth it.
///
/// # Example
/// ```ignore
/// use rusty_axon::optim::parallel_sgd::ParallelSgd;
/// use rusty_axon::nn::mlp::Mlp;
/// use rusty_axon::nn::activations::Activations;
///
/// let mlp = Mlp::new(&[8, 64, 32, 1], &[Activations::ReLU, Activations::ReLU, Activations::Sigmoid]);
/// let mut optimizer = ParallelSgd::new(0.01, mlp.parameters());
///
/// // Training loop
/// optimizer.zero_state();
/// // ... forward and backward pass ...
/// optimizer.step();
/// ```
pub struct ParallelSgd {
    learning_rate: f64,
    parameters: Vec<Node>,
}

impl ParallelSgd {
    /// Create a new parallel SGD optimizer.
    pub fn new(learning_rate: f64, parameters: Vec<Node>) -> Self {
        Self {
            learning_rate,
            parameters,
        }
    }

    /// Set the learning rate.
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    /// Get the current learning rate.
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Get the number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }
}

impl Optimizer for ParallelSgd {
    /// Apply one optimization step in parallel.
    ///
    /// For each parameter: `param = param - learning_rate * gradient`
    fn step(&mut self) {
        let lr = self.learning_rate;
        
        // Extract values and gradients (sequential, but fast)
        let values_and_grads: Vec<(f64, f64)> = self.parameters
            .iter()
            .map(|p| (p.get_value(), p.get_gradient()))
            .collect();
        
        // Parallel: compute new weight values
        let updates: Vec<f64> = values_and_grads
            .par_iter()
            .map(|(val, grad)| val - lr * grad)
            .collect();
        
        // Apply updates (sequential, but very fast)
        for (param, new_val) in self.parameters.iter_mut().zip(updates.into_iter()) {
            param.set_value(new_val);
        }
    }

    /// Reset all gradients to zero.
    fn zero_state(&mut self) {
        for param in self.parameters.iter() {
            param.zero_gradient();
        }
    }
}

