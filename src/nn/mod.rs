//! Neural network building blocks built on top of the autograd engine.

pub mod activations;
pub mod layer;
pub mod linear;
pub mod mlp;
pub mod neuron;
pub mod param_view;
pub mod visualization;

// Re-export commonly used types
pub use activations::Activations;
pub use layer::Layer;
pub use linear::Linear;
pub use mlp::Mlp;
pub use neuron::Neuron;
pub use param_view::{ParamKind, ParamView};

#[cfg(test)]
mod tests;
