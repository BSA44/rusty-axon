//! INT8-quantized inference throughput for the bench MLP.
//!
//! Builds a random-init f32 model, quantizes every layer to per-tensor
//! symmetric INT8 (Phase 7 PTQ), then benches [`Mlp::infer_into_arena`] on
//! the quantized weights.  Phase 7 made INT8 inference-only — these benches
//! only exist under `--features quant-i8`.

#[cfg(feature = "quant-i8")]
mod common;

#[cfg(feature = "quant-i8")]
use criterion::{black_box, criterion_group, criterion_main, Criterion};
#[cfg(feature = "quant-i8")]
use rusty_axon::nn::activations::Activations;
#[cfg(feature = "quant-i8")]
use rusty_axon::nn::arena::InferArena;
#[cfg(feature = "quant-i8")]
use rusty_axon::nn::mlp::Mlp;

#[cfg(feature = "quant-i8")]
use crate::common::{seeded_random_vec, ARCH, INPUT_DIM, OUTPUT_DIM};

#[cfg(feature = "quant-i8")]
fn bench_infer_into_arena_i8(c: &mut Criterion) {
    let activations = [
        Activations::ReLU,
        Activations::ReLU,
        Activations::ReLU,
        Activations::None,
    ];
    let mut mlp = Mlp::new(ARCH, &activations);
    mlp.quantize_to_i8();
    let mut arena = InferArena::for_mlp(&mlp);
    let input = seeded_random_vec(INPUT_DIM, 0xCAFE_BABE);
    let mut output = vec![0.0_f32; OUTPUT_DIM];

    c.bench_function("forward_infer/i8_arena_784_640_320_100_10", |bench| {
        bench.iter(|| {
            mlp.infer_into_arena(
                black_box(&input),
                black_box(&mut output),
                black_box(&mut arena),
            );
        });
    });
}

#[cfg(feature = "quant-i8")]
criterion_group!(benches, bench_infer_into_arena_i8);
#[cfg(feature = "quant-i8")]
criterion_main!(benches);

#[cfg(not(feature = "quant-i8"))]
fn main() {
    eprintln!("forward_infer_i8: skipped (requires --features quant-i8)");
}
