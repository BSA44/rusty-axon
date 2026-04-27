//! On-disk binary formats produced by `rusty-axon`.
//!
//! Currently exposes a single format, [`axn`], the v0.1 wire format defined
//! in `docs/AXN_FORMAT.md`.  Phase 7 reuses the same format for INT8
//! weights-only quantized tensors via the `Dtype::I8` discriminant and the
//! per-tensor `scale` field.

#[cfg(target_endian = "big")]
compile_error!(".axn is little-endian; big-endian targets are unsupported");

pub mod axn;

#[cfg(test)]
mod axn_tests;
