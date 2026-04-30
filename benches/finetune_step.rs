//! Last-layer-only fine-tune bench: forward through every layer + backward
//! into and SGD-step over only the **final** `Linear`'s parameters.  Models
//! the on-device MNIST personalisation flow that Phase 11 ships as a paper
//! demo.
//!
//! Same architecture as [`training_step`] but with a tiny batch (4) and the
//! optimizer scoped via `Mlp::parameters_for_layers((n - 1)..n)`.  The
//! forward pass still walks all four `Linear`s; only the gradient updates
//! are restricted.  Train-only.

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
fn bench_finetune_step(c: &mut Criterion) {
    let activations = [
        Activations::ReLU,
        Activations::ReLU,
        Activations::ReLU,
        Activations::None,
    ];
    let mlp = Mlp::new(ARCH, &activations);

    // Optimize only the final `Linear` (`100 -> 10`), matching the Phase 11
    // MNIST personalisation flow.
    let n = mlp.num_linear_layers();
    let mut optimizer = Sgd::new(0.01, mlp.parameters_for_layers((n - 1)..n));
    let loss_fn = CrossEntropy::new(0.1);

    let batch_size = 4;
    let (images, labels) = load_train_subset(batch_size);
    assert_eq!(images.len(), batch_size);

    let mut group = c.benchmark_group("finetune_step");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(60));
    group.warm_up_time(Duration::from_secs(3));
    group.bench_function("last_layer_only_batch_4_784_640_320_100_10", |bench| {
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
            black_box(batch_loss.get_value());
        });
    });
    group.finish();
}

#[cfg(feature = "train")]
criterion_group!(benches, bench_finetune_step);
#[cfg(feature = "train")]
criterion_main!(benches);

#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("finetune_step: skipped (requires --features train)");
}
