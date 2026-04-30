//! Pure-`f32` inference throughput for the bench MLP via the zero-alloc
//! [`Mlp::infer_into_arena`] path — the headline edge-inference number for
//! the paper's latency table.
//!
//! Always-on: the inference forward path is available under both `train` and
//! `inference` feature sets.

mod common;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rusty_axon::nn::activations::Activations;
use rusty_axon::nn::arena::InferArena;
use rusty_axon::nn::mlp::Mlp;

use crate::common::{seeded_random_vec, ARCH, INPUT_DIM, OUTPUT_DIM};

fn bench_infer_into_arena(c: &mut Criterion) {
    let activations = [
        Activations::ReLU,
        Activations::ReLU,
        Activations::ReLU,
        Activations::None,
    ];
    let mlp = Mlp::new(ARCH, &activations);
    let mut arena = InferArena::for_mlp(&mlp);
    // Deterministic input so successive runs measure the same workload;
    // accuracy is irrelevant for a latency bench.
    let input = seeded_random_vec(INPUT_DIM, 0xCAFE_BABE);
    let mut output = vec![0.0_f32; OUTPUT_DIM];

    c.bench_function("forward_infer/f32_arena_784_640_320_100_10", |bench| {
        bench.iter(|| {
            mlp.infer_into_arena(
                black_box(&input),
                black_box(&mut output),
                black_box(&mut arena),
            );
        });
    });
}

criterion_group!(benches, bench_infer_into_arena);
criterion_main!(benches);
