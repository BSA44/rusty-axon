//! Shared helpers for the Phase 8 criterion benches.
//!
//! All benches in this directory use the same MLP architecture
//! (`784 -> 640 -> 320 -> 100 -> 10` with `[ReLU, ReLU, ReLU, None]`) and the
//! same MNIST data loader, so the helpers live here rather than being
//! duplicated across seven bench files.
//!
//! Cargo treats every `.rs` directly under `benches/` as a bench target;
//! placing this module under `benches/common/mod.rs` keeps it out of that
//! discovery and lets each bench `mod common;` it back in.

#![allow(dead_code)] // each bench uses a different subset of the helpers

use std::fs::File;
use std::io::{BufRead, BufReader};

/// The MLP shape used by every Phase 8 bench.  Matches
/// `examples/bench_mnist_matmul.rs` so the matmul kernels are exercised at a
/// realistic edge-workload size rather than the toy `784 -> 64 -> 32 -> 10`.
pub const ARCH: &[usize] = &[784, 640, 320, 100, 10];

/// MNIST input dimension (matches `ARCH[0]`).
pub const INPUT_DIM: usize = 784;

/// MNIST output dimension (matches `ARCH[ARCH.len() - 1]`).
pub const OUTPUT_DIM: usize = 10;

/// Default fixed RNG seed for deterministic bench inputs.  Using a constant
/// across runs makes successive `cargo bench` invocations measure the same
/// workload (and lets criterion's outlier detection settle quickly).
pub const SEED: u64 = 0xA5A5_A5A5_A5A5_A5A5;

/// Path to the MNIST training CSV produced by `python-tests/prepare_mnist.py`.
pub const MNIST_TRAIN_CSV: &str = "python-tests/mnist/mnist_train.csv";

/// Path to the MNIST test CSV.
pub const MNIST_TEST_CSV: &str = "python-tests/mnist/mnist_test.csv";

/// Tiny seeded LCG yielding `f32` in `[-1, 1)`.  Used for deterministic
/// random inputs (and for benches that don't care about MNIST data,
/// e.g. `matmul_kernel.rs`).
pub fn seeded_random_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = ((state >> 33) as u32) & 0x007F_FFFF;
        let f = (bits as f32) / ((1u32 << 23) as f32);
        out.push(f * 2.0 - 1.0);
    }
    out
}

/// Load the first `n` MNIST images + labels from a CSV produced by
/// `python-tests/prepare_mnist.py`.  Returns `(images, labels)` where each
/// image is a `Vec<f32>` of length 784 with pixel values normalised to
/// `[0, 1]` (the prepare script already applies that normalisation).
///
/// # Panics
/// Panics if the CSV cannot be opened or a row has the wrong column count.
pub fn load_mnist_csv(path: &str, n: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    let file = File::open(path)
        .unwrap_or_else(|e| panic!("failed to open {} ({}): run `python python-tests/prepare_mnist.py`", path, e));
    let reader = BufReader::new(file);
    let mut images = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    for (i, line) in reader.lines().enumerate() {
        let line = line.expect("read line");
        if i == 0 && line.starts_with("label") {
            continue;
        }
        if images.len() >= n {
            break;
        }
        let values: Vec<f32> = line
            .split(',')
            .map(|s| s.trim().parse::<f32>().unwrap_or(0.0))
            .collect();
        assert_eq!(values.len(), 1 + INPUT_DIM, "expected 785 columns per row");
        labels.push(values[0] as usize);
        images.push(values[1..].to_vec());
    }
    (images, labels)
}

/// Convenience: load just the first `n` training images + labels.
pub fn load_train_subset(n: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    load_mnist_csv(MNIST_TRAIN_CSV, n)
}

/// Argmax over `outputs`.  Returns 0 on ties or empty input.
pub fn predict_f32(outputs: &[f32]) -> usize {
    outputs
        .iter()
        .enumerate()
        .fold(
            (0usize, f32::NEG_INFINITY),
            |(best_i, best_v), (i, &v)| if v > best_v { (i, v) } else { (best_i, best_v) },
        )
        .0
}

// ---- train-only helpers (Node-based) -----------------------------------

#[cfg(feature = "train")]
pub mod train_helpers {
    use rusty_axon::engine::value::Node;

    /// Build a `Vec<Node>` of length `INPUT_DIM` from a single MNIST image.
    pub fn image_to_nodes(image: &[f32]) -> Vec<Node> {
        image.iter().map(|&x| Node::from(x)).collect()
    }

    /// One-hot target as `Vec<Node>` for cross-entropy.
    pub fn one_hot(label: usize, num_classes: usize) -> Vec<Node> {
        (0..num_classes)
            .map(|i| Node::from(if i == label { 1.0_f32 } else { 0.0_f32 }))
            .collect()
    }

    /// Argmax over a node slice (used by the train-side legacy bench).
    pub fn predict_nodes(outputs: &[Node]) -> usize {
        outputs
            .iter()
            .enumerate()
            .fold(
                (0usize, f32::NEG_INFINITY),
                |(best_i, best_v), (i, n)| {
                    let v = n.get_value();
                    if v > best_v {
                        (i, v)
                    } else {
                        (best_i, best_v)
                    }
                },
            )
            .0
    }
}
