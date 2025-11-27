//! Neural network building blocks built on top of the autograd engine.

pub mod layer;
pub mod mlp;
pub mod neuron;
pub mod activations;
pub mod visualization;

// Re-export commonly used types
pub use mlp::Mlp;
pub use layer::Layer;
pub use neuron::Neuron;
pub use activations::Activations;

#[cfg(test)]
mod tests;