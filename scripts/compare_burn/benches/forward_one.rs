//! Burn analogue of `rusty-axon/benches/forward_train.rs`.
//!
//! Single-sample forward pass with the autograd backend live. Pair this
//! with `forward_train_fused_784_640_320_100_10` on the rusty-axon side
//! to populate Table 1 ("Forward latency, single sample") in the paper.

use compare_burn::{fixed_input, forward_one, AutoBackend, Mlp};
use criterion::{criterion_group, criterion_main, Criterion};

fn bench(c: &mut Criterion) {
    let device = Default::default();
    let model: Mlp<AutoBackend> = Mlp::new(&device);
    let input = fixed_input::<AutoBackend>(1, &device);

    c.bench_function("burn_forward_one_784_640_320_100_10", |b| {
        b.iter(|| {
            let _ = forward_one(&model, &input);
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
