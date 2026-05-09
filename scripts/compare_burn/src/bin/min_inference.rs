//! Minimal inference binary used for the binary-size comparison cell in
//! `docs/COMPARISON.md`. Mirrors `examples/min_inference.rs` in rusty-axon:
//! constructs the MLP, runs one inference, prints the argmax. The whole
//! point is the **stripped binary size**, not the latency.
//!
//! Build under the same `release-edge` profile rusty-axon uses:
//!
//!   cargo build --manifest-path scripts/compare_burn/Cargo.toml \
//!     --profile release-edge --bin min_inference

use compare_burn::{fixed_input, Mlp, NdArray};

fn main() {
    let device = Default::default();
    let model: Mlp<NdArray<f32>> = Mlp::new(&device);
    let input = fixed_input::<NdArray<f32>>(1, &device);
    let logits = model.forward(input);
    let data = logits.into_data();
    let slice = data.as_slice::<f32>().expect("f32 logits");
    let (argmax, _) = slice
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!("argmax = {}", argmax);
}
