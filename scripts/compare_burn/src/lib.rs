//! Burn baseline harness for the rusty-axon paper.
//!
//! Defines the same MLP architecture rusty-axon's Phase 8 criterion
//! suite uses (`784 -> 640 -> 320 -> 100 -> 10` with
//! `[ReLU, ReLU, ReLU, None]`, matching `benches/common/mod.rs`) and
//! exposes three driver functions consumed by the benches in `benches/`:
//!
//! * `forward_one`        — single-sample forward pass, training graph live.
//! * `infer_into_buf`     — single-sample forward pass without autograd.
//! * `train_step_batch32` — full forward + backward + SGD step on batch 32.
//!
//! The intent is **fair head-to-head comparison**, not idiomatic Burn:
//! the architecture, dtype (`f32`), batch size, and learning rate are
//! pinned to match the rusty-axon benches under
//! `rusty-axon/benches/*.rs`. The NdArray backend is used because it is
//! the only Burn backend that is pure-CPU Rust with no BLAS/CUDA — exactly
//! the constraint rusty-axon operates under.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorData};

pub use burn_ndarray::NdArray;
pub type AutoBackend = burn::backend::Autodiff<NdArray<f32>>;

pub const ARCH: &[usize] = &[784, 640, 320, 100, 10];
pub const INPUT_DIM: usize = 784;
pub const OUTPUT_DIM: usize = 10;
pub const TRAIN_BATCH: usize = 32;

/// MLP matching `benches/common/mod.rs` (and `examples/mnist_classifier.rs`)
/// in rusty-axon: `784 -> 640 -> 320 -> 100 -> 10` with three ReLU hidden
/// activations and a linear logits head.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    fc4: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Mlp<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(ARCH[0], ARCH[1]).init(device),
            fc2: LinearConfig::new(ARCH[1], ARCH[2]).init(device),
            fc3: LinearConfig::new(ARCH[2], ARCH[3]).init(device),
            fc4: LinearConfig::new(ARCH[3], ARCH[4]).init(device),
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.fc1.forward(input));
        let x = self.activation.forward(self.fc2.forward(x));
        let x = self.activation.forward(self.fc3.forward(x));
        self.fc4.forward(x)
    }
}

/// Construct a deterministic `[batch, 784]` tensor for benches. Mirrors the
/// fixed-seed input rusty-axon's benches use (`benches/common/mod.rs`)
/// so both frameworks chew on the same numerical workload.
pub fn fixed_input<B: Backend>(batch: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut data = Vec::with_capacity(batch * INPUT_DIM);
    let mut state: u32 = 0x9E37_79B9;
    for _ in 0..(batch * INPUT_DIM) {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // Map to [0, 1) — same range as MNIST pixels / 255.0.
        data.push((state >> 8) as f32 / (1u32 << 24) as f32);
    }
    Tensor::<B, 1>::from_data(TensorData::new(data, [batch * INPUT_DIM]), device)
        .reshape([batch, INPUT_DIM])
}

/// Deterministic integer labels in `[0, 10)` for cross-entropy training.
pub fn fixed_labels<B: Backend>(batch: usize, device: &B::Device) -> Tensor<B, 1, burn::tensor::Int> {
    let mut data = Vec::with_capacity(batch);
    let mut state: u32 = 0xDEAD_BEEF;
    for _ in 0..batch {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        data.push(((state >> 24) % OUTPUT_DIM as u32) as i64);
    }
    Tensor::<B, 1, burn::tensor::Int>::from_data(TensorData::new(data, [batch]), device)
}

// ---------- Driver functions consumed by the benches ----------

/// `forward_one` — single-sample forward pass with autograd live (training
/// graph built but not differentiated). The Burn analogue of rusty-axon's
/// `bench_forward_train_fused`.
pub fn forward_one<B: AutodiffBackend>(model: &Mlp<B>, input: &Tensor<B, 2>) -> Tensor<B, 2> {
    model.forward(input.clone())
}

/// `infer_into_buf` — pure inference. Drops the autograd backend entirely
/// to mirror rusty-axon's `--features inference` build path.
pub fn infer_into_buf<B: Backend>(model: &Mlp<B>, input: &Tensor<B, 2>, out: &mut [f32]) {
    let logits = model.forward(input.clone());
    let data = logits.into_data();
    let slice: &[f32] = data.as_slice().expect("NdArray backend always yields f32");
    out.copy_from_slice(slice);
}

/// `train_step_batch32` — single SGD step (forward + cross-entropy +
/// backward + manual parameter update). Returns the scalar loss so the
/// caller can sanity-check that the optimization is actually running.
pub fn train_step_batch32<B: AutodiffBackend>(
    model: Mlp<B>,
    input: Tensor<B, 2>,
    targets: Tensor<B, 1, burn::tensor::Int>,
    lr: f32,
) -> (Mlp<B>, f32) {
    use burn::nn::loss::CrossEntropyLossConfig;
    use burn::optim::GradientsParams;

    let device = input.device();
    let logits = model.forward(input);
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let loss = loss_fn.forward(logits, targets);
    let loss_value: f32 = loss
        .clone()
        .into_data()
        .as_slice::<f32>()
        .expect("scalar loss")[0];

    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);

    // Manual SGD step: w <- w - lr * dw. Avoids pulling in a Burn optimizer
    // configuration so the comparison is forward + backward + apply only.
    use burn::optim::Optimizer;
    let mut sgd = burn::optim::SgdConfig::new().init::<B, Mlp<B>>();
    let model = sgd.step(lr.into(), model, grads);
    (model, loss_value)
}
