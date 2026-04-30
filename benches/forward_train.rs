//! Train-path forward bench: `Mlp::forward(&[Node])` building the full
//! `Node` graph for the bench MLP.  Establishes the per-call cost of the
//! fused-MatMul forward, which the speedup-vs-`Layer<Neuron>` table compares
//! against [`forward_train_legacy`].
//!
//! Train-only — `Mlp::forward` is gated on `cfg(feature = "train")`.

#[cfg(feature = "train")]
mod common;

#[cfg(feature = "train")]
use criterion::{black_box, criterion_group, criterion_main, Criterion};
#[cfg(feature = "train")]
use rusty_axon::nn::activations::Activations;
#[cfg(feature = "train")]
use rusty_axon::nn::mlp::Mlp;

#[cfg(feature = "train")]
use crate::common::train_helpers::image_to_nodes;
#[cfg(feature = "train")]
use crate::common::{seeded_random_vec, ARCH, INPUT_DIM};

#[cfg(feature = "train")]
fn bench_forward_fused(c: &mut Criterion) {
    let activations = [
        Activations::ReLU,
        Activations::ReLU,
        Activations::ReLU,
        Activations::None,
    ];
    let mlp = Mlp::new(ARCH, &activations);
    let input = seeded_random_vec(INPUT_DIM, 0xCAFE_BABE);
    let inputs = image_to_nodes(&input);

    c.bench_function("forward_train/fused_784_640_320_100_10", |bench| {
        bench.iter(|| {
            // Each call rebuilds the Node graph (allocations included) so the
            // bench reflects realistic per-sample training overhead.
            let _ = mlp.forward(black_box(&inputs));
        });
    });
}

#[cfg(feature = "train")]
criterion_group!(benches, bench_forward_fused);
#[cfg(feature = "train")]
criterion_main!(benches);

#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("forward_train: skipped (requires --features train)");
}
