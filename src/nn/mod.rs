//! Neural network building blocks built on top of the autograd engine.

pub mod layer;
pub mod mlp;
pub mod neuron;
pub mod activations;

#[cfg(test)]
mod tests;