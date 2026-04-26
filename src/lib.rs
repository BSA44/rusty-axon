//! Library entry point for the `rusty-axon` training-capable edge framework.
//!
//! The crate is divided into the following high-level areas:
//! - `engine`: scalar autograd (`Value`, `Node`, `Operation`) — `train` only.
//! - `nn`: neural-network building blocks (neurons, layers, MLPs, activations,
//!   visualization). Currently gated on `train`; Phase 6 will split this into
//!   a train path (`Node`-based) and an always-on pure-`&[f32]` inference path.
//! - `optim`: parameter update routines (SGD, MeProp) — `train` only.
//! - `loss`: loss functions (MSE, RMSE, CrossEntropy) — `train` only.

// All of the existing v0.2 module tree depends on the scalar autograd engine,
// so every public module is gated on `train` in Phase 0. Phase 6 splits `nn`
// to expose a pure-`&[f32]` inference surface when only `inference` is on.
#[cfg(feature = "train")]
pub mod engine;
#[cfg(feature = "train")]
pub mod loss;
#[cfg(feature = "train")]
pub mod nn;
#[cfg(feature = "train")]
pub mod optim;

// Re-export the most commonly used types so downstream crates can simply
// `use rusty_axon::Value;`.
#[cfg(feature = "train")]
pub use engine::value::Value;
#[cfg(feature = "train")]
pub use nn::activations::Activations;
#[cfg(feature = "train")]
pub use nn::layer::Layer;
#[cfg(feature = "train")]
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
