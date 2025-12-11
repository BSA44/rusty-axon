//! Optimizers for updating parameters during training.

pub mod sgd;
pub mod optimizer;
pub mod meprop;
pub mod parallel_sgd;

#[cfg(test)]
mod tests;