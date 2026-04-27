//! Core autograd engine pieces.
//!
//! Phase 6 relocated `MatMulTape` to [`crate::nn::matmul`] so the inference
//! build (`--no-default-features --features inference`) can keep the
//! parameter buffers without pulling in the train-only `Node` graph.  The
//! engine module itself is now `train`-only.

pub mod ops;
pub mod value;

#[cfg(test)]
mod tests;

pub use crate::nn::matmul::MatMulTape;
pub use value::{Node, Value};
