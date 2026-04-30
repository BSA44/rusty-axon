//! Legacy scalar-`Neuron` forward bench: same architecture as
//! [`forward_train`] but every layer is a `Layer<Neuron>` doing scalar dot
//! products through `Rc<RefCell<Value>>`.  This is the **baseline** for the
//! fused-MatMul speedup ratio in the paper.
//!
//! `Layer<Neuron>` at `784 -> 640 -> 320 -> 100 -> 10` runs **orders of
//! magnitude slower** than the fused path; `sample_size(10)` and
//! `measurement_time(300 s)` give criterion enough budget to collect its
//! sample-floor.  Train-only.

#[cfg(feature = "train")]
mod common;

#[cfg(feature = "train")]
use std::time::Duration;

#[cfg(feature = "train")]
use criterion::{black_box, criterion_group, criterion_main, Criterion};
#[cfg(feature = "train")]
use rusty_axon::engine::value::Node;
#[cfg(feature = "train")]
use rusty_axon::nn::activations::Activations;
#[cfg(feature = "train")]
use rusty_axon::nn::layer::Layer;

#[cfg(feature = "train")]
use crate::common::train_helpers::image_to_nodes;
#[cfg(feature = "train")]
use crate::common::{seeded_random_vec, ARCH, INPUT_DIM};

#[cfg(feature = "train")]
fn forward_legacy(layers: &[Layer], inputs: &[Node]) -> Vec<Node> {
    let mut current = inputs.to_vec();
    for layer in layers {
        current = layer.forward(&current);
    }
    current
}

#[cfg(feature = "train")]
fn bench_forward_legacy(c: &mut Criterion) {
    // Build the same architecture as the fused path, but using the legacy
    // scalar `Layer<Neuron>` baseline.  `Layer::new` takes `&Activations`.
    let layers: Vec<Layer> = ARCH
        .windows(2)
        .enumerate()
        .map(|(i, w)| {
            let activation = if i == ARCH.len() - 2 {
                Activations::None
            } else {
                Activations::ReLU
            };
            Layer::new(w[0], w[1], &activation)
        })
        .collect();
    let input = seeded_random_vec(INPUT_DIM, 0xCAFE_BABE);
    let inputs = image_to_nodes(&input);

    let mut group = c.benchmark_group("forward_train_legacy");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(300));
    group.warm_up_time(Duration::from_secs(5));
    group.bench_function("legacy_neuron_784_640_320_100_10", |bench| {
        bench.iter(|| {
            let _ = forward_legacy(black_box(&layers), black_box(&inputs));
        });
    });
    group.finish();
}

#[cfg(feature = "train")]
criterion_group!(benches, bench_forward_legacy);
#[cfg(feature = "train")]
criterion_main!(benches);

#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("forward_train_legacy: skipped (requires --features train)");
}
