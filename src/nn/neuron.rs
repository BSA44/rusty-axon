//! Single neuron abstraction composed of weighted inputs and an activation.

use crate::engine::value::Value;

/// Basic neuron that consumes a vector of inputs and produces a scalar output.
pub struct Neuron {
    // TODO: store learnable weights, bias, and activation metadata.
}

impl Neuron {
    /// Create a new neuron with the requested number of inputs.
    pub fn new(_inputs: usize) -> Self {
        todo!("Initialize weights, bias, and activation");
    }

    /// Execute the forward pass for this neuron.
    pub fn forward(&self, _inputs: &[Value]) -> Value {
        todo!("Compute weighted sum and apply activation");
    }
}
