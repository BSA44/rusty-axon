//! Library entry point for the `rusty-axon` micrograd implementation.
//!
//! The crate is divided into three high-level areas:
//! - `engine`: core autograd data structures and differentiation logic.
//! - `nn`: basic neural network building blocks constructed on top of the engine.
//! - `optim`: parameter update routines (optimizers, schedulers, etc.).

pub mod engine;
pub mod nn;
pub mod optim;

// Re-export the most commonly used types so downstream crates can simply
// `use rusty_axon::Value;`.
pub use engine::value::Value;
