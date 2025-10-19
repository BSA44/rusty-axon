//! Abstractions for managing the dynamic computation graph.

/// Placeholder type that will eventually hold the nodes created during forward
/// passes and enable reverse-mode automatic differentiation.
pub struct ComputationGraph {
    // TODO: store references to nodes/values created during a forward pass.
}

impl ComputationGraph {
    /// Create an empty computation graph.
    pub fn new() -> Self {
        todo!("Initialize graph bookkeeping once node structure is defined");
    }

    /// Reset any gradient-related state without discarding the graph topology.
    pub fn zero_grad(&mut self) {
        todo!("Iterate over tracked nodes and clear their gradients");
    }

    /// Register a newly created value with the graph so it participates in
    /// backpropagation.
    pub fn track_value(&mut self) {
        todo!("Store metadata for the provided value node");
    }
}
