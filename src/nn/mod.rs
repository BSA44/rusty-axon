//! Neural network building blocks.
//!
//! Phase 6: this module is always-on, but the `Node`-based pieces — the
//! legacy scalar `Neuron`/`Layer` baseline, the visualization helpers, and
//! the `ParamView` re-export — are gated on `train`.  `activations`,
//! `linear`, `matmul`, and `mlp` are always available; their `Node`-using
//! methods are gated internally.

pub mod activations;
#[cfg(feature = "train")]
pub mod layer;
pub mod linear;
pub mod matmul;
pub mod mlp;
#[cfg(feature = "train")]
pub mod neuron;
#[cfg(feature = "train")]
pub mod param_view;
#[cfg(feature = "train")]
pub mod visualization;

// Re-export commonly used types
pub use activations::Activations;
#[cfg(feature = "train")]
pub use layer::Layer;
pub use linear::Linear;
pub use mlp::Mlp;
#[cfg(feature = "train")]
pub use neuron::Neuron;
#[cfg(feature = "train")]
pub use param_view::{ParamKind, ParamView};

#[cfg(test)]
#[cfg(feature = "train")]
mod tests;
