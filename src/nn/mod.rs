//! Neural network building blocks built on top of the autograd engine.

pub mod layer;
pub mod mlp;
pub mod neuron;
pub mod activations;
pub mod visualization;
pub mod parallel;

// Re-export commonly used types
pub use mlp::Mlp;
pub use layer::Layer;
pub use neuron::Neuron;
pub use activations::Activations;
pub use parallel::ParallelTrainer;

#[cfg(test)]
mod tests;