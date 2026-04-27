//! `ParamView` re-exports.
//!
//! The view type itself lives next to `MatMulTape` in [`crate::nn::matmul`]
//! because [`crate::engine::value::Node`] needs to reference it directly.
//! This module provides a stable `nn`-side import path matching the Phase 2
//! plan layout.

pub use crate::nn::matmul::{ParamKind, ParamView};
