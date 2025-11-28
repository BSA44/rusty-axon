//! Library entry point for the `rusty-axon` micrograd implementation.
//!
//! The crate is divided into three high-level areas:
//! - `engine`: core autograd data structures and differentiation logic.
//! - `nn`: basic neural network building blocks constructed on top of the engine.
//! - `optim`: parameter update routines (optimizers, schedulers, etc.).

pub mod engine;
pub mod nn;
pub mod optim;
pub mod loss;

// Re-export the most commonly used types so downstream crates can simply
// `use rusty_axon::Value;`.
pub use engine::value::Value;
pub use nn::visualization::render_network_to;
pub use nn::activations::Activations;
pub use nn::mlp::Mlp;
pub use nn::layer::Layer;
pub use nn::neuron::Neuron;
pub use nn::visualization::NetworkVisualizationConfig;
pub use nn::visualization::save_network_graph;
pub use nn::visualization::check_graphviz;

