//! Pure `sgemm_rm` micro-benchmark — the kernel-only speedup ratio that
//! anchors the matrixmultiply-vs-naive table in the paper.
//!
//! Three shapes:
//! - **64x64x64** — square small.
//! - **256x256x256** — square medium; matrixmultiply's headline case.
//! - **784x640x1**  — the matvec shape `Linear::forward` issues for the
//!   first layer of the bench MLP (`784 -> 640`).  This is the shape that
//!   actually dominates per-call inference latency.
//!
//! Runs under any feature combo; the kernel is selected at compile time via
//! the `matrixmultiply` / `naive-matmul` features (see
//! [`rusty_axon::nn::matmul::sgemm_rm`]).

mod common;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rusty_axon::nn::matmul::sgemm_rm;

use crate::common::seeded_random_vec;

fn bench_sgemm_64x64x64(c: &mut Criterion) {
    let m = 64;
    let k = 64;
    let n = 64;
    let a = seeded_random_vec(m * k, 1);
    let b = seeded_random_vec(k * n, 2);
    let mut out = vec![0.0_f32; m * n];

    c.bench_function("matmul_kernel/sgemm_rm_64x64x64", |bench| {
        bench.iter(|| {
            sgemm_rm(
                m,
                k,
                n,
                1.0,
                black_box(&a),
                k,
                black_box(&b),
                n,
                0.0,
                black_box(&mut out),
                n,
            );
        });
    });
}

fn bench_sgemm_256x256x256(c: &mut Criterion) {
    let m = 256;
    let k = 256;
    let n = 256;
    let a = seeded_random_vec(m * k, 3);
    let b = seeded_random_vec(k * n, 4);
    let mut out = vec![0.0_f32; m * n];

    c.bench_function("matmul_kernel/sgemm_rm_256x256x256", |bench| {
        bench.iter(|| {
            sgemm_rm(
                m,
                k,
                n,
                1.0,
                black_box(&a),
                k,
                black_box(&b),
                n,
                0.0,
                black_box(&mut out),
                n,
            );
        });
    });
}

fn bench_sgemm_matvec_784x640(c: &mut Criterion) {
    // m=out_dim=640, k=in_dim=784, n=1 — the forward shape `Linear::forward`
    // issues for the first hidden layer of the bench MLP.
    let m = 640;
    let k = 784;
    let n = 1;
    let a = seeded_random_vec(m * k, 5);
    let b = seeded_random_vec(k * n, 6);
    let mut out = vec![0.0_f32; m * n];

    c.bench_function("matmul_kernel/sgemm_rm_matvec_784x640", |bench| {
        bench.iter(|| {
            sgemm_rm(
                m,
                k,
                n,
                1.0,
                black_box(&a),
                k,
                black_box(&b),
                n,
                0.0,
                black_box(&mut out),
                n,
            );
        });
    });
}

criterion_group!(
    benches,
    bench_sgemm_64x64x64,
    bench_sgemm_256x256x256,
    bench_sgemm_matvec_784x640,
);
criterion_main!(benches);
