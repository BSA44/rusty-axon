//! Naive scalar `sgemm_rm` fallback used when the `matrixmultiply` crate is
//! disabled or when the `naive-matmul` feature flag is set to force it.
//!
//! Selected at compile time by [`super::kernel`] based on the active feature
//! combination.  The signature mirrors a row-major BLAS `sgemm`:
//! `C := beta * C + alpha * A @ B` where `A` is `m × k`, `B` is `k × n`, and
//! `C` is `m × n`.  Leading dimensions (`lda`, `ldb`, `ldc`) are the
//! stride-between-rows in elements; for tightly packed row-major storage
//! `lda = k`, `ldb = n`, `ldc = n`.

/// Row-major naive `sgemm`.  Used as the Phase 8 baseline for the
/// matrixmultiply speedup table when `naive-matmul` is on.
///
/// # Panics (debug builds only)
///
/// Panics on a debug build if any of `lda < k`, `ldb < n`, `ldc < n`, or any
/// of the slice lengths is too small to address `[m, k] · [k, n] -> [m, n]`.
#[allow(dead_code)] // unused outside tests under the default (matrixmultiply) build
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

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[i * lda + kk] * b[kk * ldb + j];
            }
            let cij = i * ldc + j;
            c[cij] = beta * c[cij] + alpha * acc;
        }
    }
}
