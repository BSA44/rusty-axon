//! Full training step bench: forward + cross-entropy + backward + SGD step
//! over a 32-sample mini-batch on the bench MLP.  Demonstrates that
//! end-to-end training is feasible; the latency number drives the
//! single-step-cost row of the paper's training-comparison table.
//!
//! Train-only.  `sample_size(20)` + `measurement_time(120 s)` because each
//! iteration walks 32 forward + 32 backward passes through ~672k params.

#[cfg(feature = "train")]
mod common;

#[cfg(feature = "train")]
use std::time::Duration;

#[cfg(feature = "train")]
use criterion::{black_box, criterion_group, criterion_main, Criterion};
#[cfg(feature = "train")]
use rusty_axon::engine::value::Node;
#[cfg(feature = "train")]
use rusty_axon::loss::cross_entropy::CrossEntropy;
#[cfg(feature = "train")]
use rusty_axon::loss::loss::Loss;
#[cfg(feature = "train")]
use rusty_axon::nn::activations::Activations;
#[cfg(feature = "train")]
use rusty_axon::nn::mlp::Mlp;
#[cfg(feature = "train")]
use rusty_axon::optim::optimizer::Optimizer;
#[cfg(feature = "train")]
use rusty_axon::optim::sgd::Sgd;

#[cfg(feature = "train")]
use crate::common::train_helpers::{image_to_nodes, one_hot};
#[cfg(feature = "train")]
use crate::common::{load_train_subset, ARCH, OUTPUT_DIM};

#[cfg(feature = "train")]
fn bench_training_step(c: &mut Criterion) {
    let activations = [
        Activations::ReLU,
        Activations::ReLU,
        Activations::ReLU,
        Activations::None,
    ];
    let mlp = Mlp::new(ARCH, &activations);
    let mut optimizer = Sgd::new(0.01, mlp.parameters());
    let loss_fn = CrossEntropy::new(0.1);

    // Load a fixed 32-sample MNIST batch from disk; if MNIST data isn't
    // present, fall back to a hand-built synthetic batch so the bench still
    // compiles and runs.
    let batch_size = 32;
    let (images, labels) = load_train_subset(batch_size);
    assert_eq!(images.len(), batch_size, "need {} MNIST samples for the batch", batch_size);

    let mut group = c.benchmark_group("training_step");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(120));
    group.warm_up_time(Duration::from_secs(5));
    group.bench_function("sgd_batch_32_784_640_320_100_10", |bench| {
        bench.iter(|| {
            optimizer.zero_state();
            let mut batch_loss = Node::from(0.0_f32);
            for i in 0..batch_size {
                let inputs = image_to_nodes(&images[i]);
                let outputs = mlp.forward(&inputs);
                let target = one_hot(labels[i], OUTPUT_DIM);
                batch_loss = batch_loss + loss_fn.forward(&outputs, &target);
            }
            batch_loss = batch_loss / batch_size as f32;
            batch_loss.backward();
            optimizer.step();
            // Keep the loss node from being optimized away.
            black_box(batch_loss.get_value());
        });
    });
    group.finish();
}

#[cfg(feature = "train")]
criterion_group!(benches, bench_training_step);
#[cfg(feature = "train")]
criterion_main!(benches);

#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("training_step: skipped (requires --features train)");
}
