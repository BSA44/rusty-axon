//! Burn analogue of `rusty-axon/benches/training_step.rs`.
//!
//! Single SGD step (forward + cross-entropy + backward + parameter update)
//! at batch size 32, matching rusty-axon's `bench_training_step_batch32`.
//! This populates the Burn column of Table 3 ("Single training step,
//! batch=32").

use compare_burn::{fixed_input, fixed_labels, train_step_batch32, AutoBackend, Mlp, TRAIN_BATCH};
use criterion::{criterion_group, criterion_main, Criterion};

fn bench(c: &mut Criterion) {
    let device = Default::default();
    let input = fixed_input::<AutoBackend>(TRAIN_BATCH, &device);
    let targets = fixed_labels::<AutoBackend>(TRAIN_BATCH, &device);
    let lr = 0.01_f32;

    c.bench_function("burn_train_step_batch32_784_640_320_100_10", |b| {
        // Each iteration starts from a fresh model so the test measures the
        // cost of one step from a known state — same convention as rusty-axon's
        // `bench_training_step_batch32`, which calls `optimizer.zero_state`
        // before every iter.
        b.iter_with_setup(
            || Mlp::<AutoBackend>::new(&device),
            |model| {
                let _ = train_step_batch32(model, input.clone(), targets.clone(), lr);
            },
        );
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
