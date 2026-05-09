//! Burn analogue of `rusty-axon/benches/forward_infer_f32.rs`.
//!
//! Pure-inference forward pass (no autograd). The headline edge-inference
//! number for the Burn column of Tables 1 and 2.

use compare_burn::{fixed_input, infer_into_buf, Mlp, NdArray, OUTPUT_DIM};
use criterion::{criterion_group, criterion_main, Criterion};

fn bench(c: &mut Criterion) {
    let device = Default::default();
    let model: Mlp<NdArray<f32>> = Mlp::new(&device);
    let input = fixed_input::<NdArray<f32>>(1, &device);
    let mut out = vec![0.0_f32; OUTPUT_DIM];

    c.bench_function("burn_infer_into_buf_784_640_320_100_10", |b| {
        b.iter(|| {
            infer_into_buf(&model, &input, &mut out);
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
