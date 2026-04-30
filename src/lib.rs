//! Library entry point for the `rusty-axon` training-capable edge framework.
//!
//! Phase 6 splits the crate into:
//! - `engine`: scalar autograd (`Value`, `Node`, `Operation`) — `train` only.
//! - `nn`: neural-network building blocks.  Always-on, but the `Node`-based
//!   train surface (`Linear::forward`, `Mlp::forward`, `Activations::apply`,
//!   visualization, the legacy `Neuron`/`Layer` baseline) is gated on
//!   `train`; the pure-`&[f32]` inference surface (`Linear::infer_into_f32`,
//!   `Mlp::infer`, `Mlp::infer_into`, `Activations::apply_f32_inplace`)
//!   is always available.
//! - `format`: `.axn` model file I/O.  Always-on so inference builds can
//!   load pretrained weights.
//! - `optim`: parameter update routines (SGD, MeProp) — `train` only.
//! - `loss`: loss functions (MSE, RMSE, CrossEntropy) — `train` only.

#[cfg(feature = "train")]
pub mod engine;
pub mod format;
#[cfg(feature = "train")]
pub mod loss;
pub mod nn;
#[cfg(feature = "train")]
pub mod optim;

// Re-export the most commonly used types.  The train-only re-exports stay
// gated; the always-on ones are surfaced for both feature combos.
#[cfg(feature = "train")]
pub use engine::value::Value;
pub use nn::activations::Activations;
pub use nn::arena::InferArena;
#[cfg(feature = "train")]
pub use nn::layer::Layer;
pub use nn::linear::Linear;
pub use nn::mlp::Mlp;
#[cfg(feature = "train")]
pub use nn::neuron::Neuron;
#[cfg(feature = "train")]
pub use nn::visualization::check_graphviz;
#[cfg(feature = "train")]
pub use nn::visualization::render_network_to;
#[cfg(feature = "train")]
pub use nn::visualization::save_network_graph;
#[cfg(feature = "train")]
pub use nn::visualization::NetworkVisualizationConfig;
