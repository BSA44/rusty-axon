//! Fully-connected linear layer backed by a fused [`MatMulTape`].
//!
//! `Linear` is the user-facing wrapper around the Phase 1 fused matmul: one
//! call to [`Linear::forward`] performs a single `y = W @ x + b` matmul,
//! mints one [`Operation::MatMul`]-tagged `Node` per output, and (optionally)
//! runs the activation as scalar `Node` ops on top.  The weight and bias
//! buffers live as flat `Vec<f32>` inside an `Rc<MatMulTape>` so that the
//! Phase 4 `matrixmultiply::sgemm` swap is a drop-in change at the kernel
//! call sites — and so optimizers see the parameters as `Vec<Node>` via
//! [`ParamView`] leaves.
//!
//! Phase 6 carves the train-only forward (returns `Vec<Node>`) away from the
//! always-on pure-`&[f32]` [`Linear::infer_into_f32`].  Both routes share
//! the same `sgemm_rm` kernel; the inference path skips the `Node`-graph
//! bookkeeping so it can ship in `--features inference` builds without the
//! engine.

use std::cell::Ref;
use std::rc::Rc;

use rand::Rng;

use crate::nn::activations::Activations;
use crate::nn::matmul::MatMulTape;
#[cfg(feature = "train")]
use crate::engine::value::Node;
#[cfg(feature = "train")]
use crate::nn::matmul::{ParamKind, ParamView};

/// How a [`Linear`] layer holds its weight matrix.
///
/// `F32` is the default — weights live in `tape.weights` and the layer is
/// trainable.  `I8` is the Phase 7 inference-only path: `tape.weights` is
/// emptied (memory savings are the whole point of PTQ) and the i8 buffer
/// + per-tensor scale move into the layer.  Biases stay f32 in either case.
#[derive(Debug)]
pub enum WeightStorage {
    /// Weights live in `tape.weights` as a flat row-major `Vec<f32>`.
    F32,
    /// Per-tensor symmetric INT8 quantization.  `qweights` is row-major
    /// `[out_dim, in_dim]`; `scale` is the dequantization multiplier.
    #[cfg(feature = "quant-i8")]
    I8 { qweights: Vec<i8>, scale: f32 },
}

/// Fully-connected layer with optional element-wise activation.
///
/// Layout: weights are row-major `[out_dim, in_dim]`; bias is `[out_dim]`.
/// Under `feature = "train"`, `cached_params` materialises one `ParamView`-
/// backed `Node` per scalar parameter so `parameters()` is a cheap clone.
/// Inference-only builds drop `cached_params` entirely.
pub struct Linear {
    tape: Rc<MatMulTape>,
    activation: Activations,
    storage: WeightStorage,
    #[cfg(feature = "train")]
    cached_params: Vec<Node>,
}

impl Linear {
    /// Random uniform `[-1, 1]` init for both weights and bias.
    pub fn new(in_dim: usize, out_dim: usize, activation: Activations) -> Self {
        let mut rng = rand::rng();
        let weights: Vec<f32> = (0..out_dim * in_dim)
            .map(|_| rng.random_range(-1.0..1.0_f32))
            .collect();
        let bias: Vec<f32> = (0..out_dim)
            .map(|_| rng.random_range(-1.0..1.0_f32))
            .collect();
        Self::with_weights_inner(in_dim, out_dim, weights, bias, activation)
    }

    /// Construct with caller-supplied weights/bias.  Useful for tests and for
    /// `Mlp::load` (Phase 5).
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != out_dim * in_dim` or `bias.len() != out_dim`.
    pub fn with_weights(
        in_dim: usize,
        out_dim: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
        activation: Activations,
    ) -> Self {
        Self::with_weights_inner(in_dim, out_dim, weights, bias, activation)
    }

    fn with_weights_inner(
        in_dim: usize,
        out_dim: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
        activation: Activations,
    ) -> Self {
        let tape = MatMulTape::new(in_dim, out_dim, weights, bias);
        #[cfg(feature = "train")]
        let cached_params = {
            let mut params = Vec::with_capacity(out_dim * in_dim + out_dim);
            for i in 0..out_dim * in_dim {
                params.push(Node::from_param_view(ParamView {
                    tape: Rc::clone(&tape),
                    kind: ParamKind::Weight,
                    index: i,
                }));
            }
            for i in 0..out_dim {
                params.push(Node::from_param_view(ParamView {
                    tape: Rc::clone(&tape),
                    kind: ParamKind::Bias,
                    index: i,
                }));
            }
            params
        };
        Self {
            tape,
            activation,
            storage: WeightStorage::F32,
            #[cfg(feature = "train")]
            cached_params,
        }
    }

    /// Construct a quantized layer directly from an i8 weight buffer + scale.
    /// Used by `Mlp::load` when the `.axn` file already carries I8 tensors.
    /// The underlying `MatMulTape` is created with **empty** f32 weights so
    /// the memory savings are realised immediately.
    ///
    /// # Panics
    ///
    /// Panics if `qweights.len() != out_dim * in_dim` or `bias.len() != out_dim`.
    /// Calls to [`Linear::forward`] on the result panic — quantized layers
    /// are inference-only.
    #[cfg(feature = "quant-i8")]
    pub fn with_quantized_weights(
        in_dim: usize,
        out_dim: usize,
        qweights: Vec<i8>,
        scale: f32,
        bias: Vec<f32>,
        activation: Activations,
    ) -> Self {
        assert_eq!(
            qweights.len(),
            out_dim * in_dim,
            "qweights must be row-major [out_dim, in_dim]"
        );
        assert_eq!(bias.len(), out_dim, "bias must be [out_dim]");
        // Tape carries an empty weight buffer — the I8 path bypasses it.
        // ParamView Nodes would index into an empty vec, but `parameters()`
        // panics for quantized layers, so they're never created.
        let tape = MatMulTape::new(in_dim, out_dim, Vec::new(), bias);
        Self {
            tape,
            activation,
            storage: WeightStorage::I8 { qweights, scale },
            #[cfg(feature = "train")]
            cached_params: Vec::new(),
        }
    }

    /// Train-path forward: returns `out_dim` activated `Node`s, each carrying
    /// `Operation::MatMul { tape, output_index }` underneath the activation
    /// chain (or directly if `activation == None`).
    ///
    /// # Panics
    ///
    /// Panics if this layer is quantized (Phase 7's INT8 path is
    /// inference-only; load f32 weights first to fine-tune).
    #[cfg(feature = "train")]
    pub fn forward(&self, inputs: &[Node]) -> Vec<Node> {
        #[cfg(feature = "quant-i8")]
        if matches!(self.storage, WeightStorage::I8 { .. }) {
            panic!(
                "Cannot train a quantized Linear layer; load f32 weights for fine-tuning"
            );
        }
        let raw = self.tape.forward(inputs);
        // Activations apply scalar Node ops on top of the matmul outputs;
        // CrossEntropy/Softmax then walk those scalar Nodes as usual.
        match self.activation {
            Activations::None => raw,
            _ => raw
                .into_iter()
                .map(|n| self.activation.apply(n))
                .collect(),
        }
    }

    /// Pure-`f32` inference path — always available, including under
    /// `--features inference`.  Computes `y = activation(W @ x + b)` without
    /// allocating any `Node`s.  Routes through the same `sgemm_rm` kernel as
    /// the train-mode forward (matrixmultiply on the default build, naive
    /// fallback under `naive-matmul`).
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != in_dim` or `output.len() != out_dim`.
    pub fn infer_into_f32(&self, input: &[f32], output: &mut [f32]) {
        let in_dim = self.tape.in_dim;
        let out_dim = self.tape.out_dim;
        assert_eq!(input.len(), in_dim, "input length mismatch");
        assert_eq!(output.len(), out_dim, "output length mismatch");
        let bias = self.tape.bias.borrow();
        match &self.storage {
            WeightStorage::F32 => {
                let weights = self.tape.weights.borrow();
                // y = b + W @ x; same shape parameters as the train-mode forward.
                output.copy_from_slice(&bias);
                crate::nn::matmul::sgemm_rm(
                    out_dim, in_dim, 1, 1.0, &weights, in_dim, input, 1, 1.0, output, 1,
                );
            }
            #[cfg(feature = "quant-i8")]
            WeightStorage::I8 { qweights, scale } => {
                // Dequant-fused matvec: y = b + scale * (qW @ x).  Strategy
                // chosen by `quant_matvec` based on out*in vs threshold.
                crate::nn::quant::quant_matvec(
                    out_dim, in_dim, qweights, *scale, &bias, input, output,
                );
            }
        }
        self.activation.apply_f32_inplace(output);
    }

    /// All trainable parameters as `Node`s suitable for `Sgd::new` / `MeProp::new`.
    /// Length is `in_dim * out_dim + out_dim` (weights followed by biases).
    ///
    /// # Panics
    ///
    /// Panics on a quantized layer — its parameter buffers don't exist as
    /// f32 anymore.  Reload f32 weights before constructing an optimizer.
    #[cfg(feature = "train")]
    pub fn parameters(&self) -> Vec<Node> {
        #[cfg(feature = "quant-i8")]
        if matches!(self.storage, WeightStorage::I8 { .. }) {
            panic!(
                "Cannot extract trainable parameters from a quantized Linear layer"
            );
        }
        self.cached_params.clone()
    }

    pub fn in_dim(&self) -> usize {
        self.tape.in_dim
    }

    pub fn out_dim(&self) -> usize {
        self.tape.out_dim
    }

    pub fn weights(&self) -> Ref<'_, Vec<f32>> {
        self.tape.weights.borrow()
    }

    pub fn bias(&self) -> Ref<'_, Vec<f32>> {
        self.tape.bias.borrow()
    }

    pub fn activation(&self) -> &Activations {
        &self.activation
    }

    /// The shared tape backing this layer.  Exposed for advanced use (e.g.
    /// custom optimizers, serialization in Phase 5).
    pub fn tape(&self) -> &Rc<MatMulTape> {
        &self.tape
    }

    /// Whether this layer's weights are stored as INT8 (Phase 7 PTQ).
    pub fn is_quantized(&self) -> bool {
        #[cfg(feature = "quant-i8")]
        {
            matches!(self.storage, WeightStorage::I8 { .. })
        }
        #[cfg(not(feature = "quant-i8"))]
        {
            false
        }
    }

    /// Quantize this layer's f32 weights in place to per-tensor symmetric
    /// INT8.  Frees the f32 weight buffer in the underlying tape.  Bias
    /// stays f32 and the activation choice is preserved.  Idempotent.
    ///
    /// # Panics
    ///
    /// Panics if the layer was constructed from already-quantized weights
    /// (no f32 source to quantize from).  Calling on an already-quantized
    /// layer is a no-op (idempotent).
    #[cfg(feature = "quant-i8")]
    pub fn quantize_to_i8(&mut self) {
        if matches!(self.storage, WeightStorage::I8 { .. }) {
            return;
        }
        let (qweights, scale) = {
            let weights = self.tape.weights.borrow();
            crate::nn::quant::quantize_per_tensor_symmetric(&weights)
        };
        // Drop the f32 weight buffer — the tape holds it via RefCell, so we
        // swap in an empty Vec to release the allocation.  The cached
        // ParamView Nodes still exist but `parameters()` panics for
        // quantized layers, so they're never dereferenced.
        *self.tape.weights.borrow_mut() = Vec::new();
        self.storage = WeightStorage::I8 { qweights, scale };
        #[cfg(feature = "train")]
        {
            self.cached_params.clear();
        }
    }

    /// Borrow the i8 weight buffer + scale for serialization.  Returns
    /// `None` for f32 layers.
    #[cfg(feature = "quant-i8")]
    pub fn quantized_weights(&self) -> Option<(&[i8], f32)> {
        match &self.storage {
            WeightStorage::I8 { qweights, scale } => Some((qweights, *scale)),
            WeightStorage::F32 => None,
        }
    }
}
