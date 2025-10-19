//! Definitions of differentiable operations that can be applied to values.

/// Every differentiable operation should describe how to perform the forward
/// computation and how to propagate gradients backward.
pub trait Operation {
    /// Perform the forward evaluation for this operation and return the result
    /// as a newly created value node.
    fn forward(&self) {
        todo!("Apply the primitive's forward computation");
    }

    /// Propagate gradients to the operation's inputs during the backward pass.
    fn backward(&self) {
        todo!("Distribute the upstream gradient to the operation's arguments");
    }
}
