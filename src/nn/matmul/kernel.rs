//! Compile-time kernel selector.  Re-exports `sgemm_rm` from either
//! [`super::kernel_mm`] (matrixmultiply, default) or [`super::kernel_naive`]
//! (forced naive, used by the `naive-matmul` feature flag and by builds that
//! drop the `matrixmultiply` dep).
//!
//! The two kernels are functionally identical to within rounding.  Phase 4's
//! kernel-agreement test verifies this when both implementations are compiled
//! — i.e. when the `matrixmultiply` feature is on.

#[cfg(all(feature = "matrixmultiply", not(feature = "naive-matmul")))]
pub use super::kernel_mm::sgemm_rm;

#[cfg(any(not(feature = "matrixmultiply"), feature = "naive-matmul"))]
pub use super::kernel_naive::sgemm_rm;
