//! Scalar value node that participates in automatic differentiation.

/// Scalar value tracked by the autograd engine.
pub struct Value {
    // TODO: store data, gradient, parents, and operation metadata.
}

impl Value {
    /// Construct a value node from raw data.
    pub fn new() -> Self {
        todo!("Store the provided scalar data and initialize autograd metadata");
    }

    /// Trigger backpropagation starting from this value.
    pub fn backward(&mut self) {
        todo!("Kick off reverse-mode automatic differentiation");
    }

    /// Reset the gradient accumulated on this node.
    pub fn zero_grad(&mut self) {
        todo!("Clear the gradient associated with this value");
    }
}
