//! `matrixmultiply`-backed `sgemm_rm`.
//!
//! Wraps `matrixmultiply::sgemm` for row-major matrices.  On `aarch64` the
//! crate auto-dispatches to a NEON microkernel (verified via `nm` on the
//! cross-compiled binary in Phase 9), which is the entire reason the fused
//! [`super::MatMulTape`] exists.
//!
//! Selected at compile time by [`super::kernel`] when the `matrixmultiply`
//! feature is on and `naive-matmul` is off — i.e. the default build.

/// Row-major `sgemm` via `matrixmultiply::sgemm`.  Same contract as
/// [`super::kernel_naive::sgemm_rm`].
#[allow(dead_code)] // unused when the `naive-matmul` flag forces the naive kernel
pub fn sgemm_rm(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    debug_assert!(lda >= k, "lda ({}) must be >= k ({})", lda, k);
    debug_assert!(ldb >= n, "ldb ({}) must be >= n ({})", ldb, n);
    debug_assert!(ldc >= n, "ldc ({}) must be >= n ({})", ldc, n);
    if m == 0 || n == 0 {
        return;
    }
    debug_assert!(a.len() >= (m - 1) * lda + k, "A slice too small");
    debug_assert!(b.len() >= k.saturating_sub(1) * ldb + n, "B slice too small");
    debug_assert!(c.len() >= (m - 1) * ldc + n, "C slice too small");

    // Safety:
    //   - The debug_assert!s above check the slice-length invariant required
    //     by matrixmultiply: every (i, j) ∈ [0, m) × [0, n) satisfies
    //     i*lda + (k-1) < a.len(), (k-1)*ldb + j < b.len(), and
    //     i*ldc + j < c.len(); same for k samples on A/B.
    //   - Row-major strides: rs = leading-dim, cs = 1.
    //   - `a` and `b` are immutable borrows; `c` is the only mutable borrow,
    //     which Rust's borrow checker enforces does not alias `a` or `b`.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            alpha,
            a.as_ptr(),
            lda as isize,
            1,
            b.as_ptr(),
            ldb as isize,
            1,
            beta,
            c.as_mut_ptr(),
            ldc as isize,
            1,
        );
    }
}
