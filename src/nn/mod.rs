//! Neural network building blocks built on top of the autograd engine.

pub mod activations;
pub mod layer;
pub mod mlp;
pub mod neuron;
pub mod visualization;

// Re-export commonly used types
pub use activations::Activations;
pub use layer::Layer;
pub use mlp::Mlp;
pub use neuron::Neuron;

#[cfg(test)]
mod tests;
