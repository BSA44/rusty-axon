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

/// Fully-connected layer with optional element-wise activation.
///
/// Layout: weights are row-major `[out_dim, in_dim]`; bias is `[out_dim]`.
/// Under `feature = "train"`, `cached_params` materialises one `ParamView`-
/// backed `Node` per scalar parameter so `parameters()` is a cheap clone.
/// Inference-only builds drop `cached_params` entirely.
pub struct Linear {
    tape: Rc<MatMulTape>,
    activation: Activations,
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
            #[cfg(feature = "train")]
            cached_params,
        }
    }

    /// Train-path forward: returns `out_dim` activated `Node`s, each carrying
    /// `Operation::MatMul { tape, output_index }` underneath the activation
    /// chain (or directly if `activation == None`).
    #[cfg(feature = "train")]
    pub fn forward(&self, inputs: &[Node]) -> Vec<Node> {
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
        let weights = self.tape.weights.borrow();
        let bias = self.tape.bias.borrow();
        // y = b + W @ x; same shape parameters as the train-mode forward.
        output.copy_from_slice(&bias);
        crate::nn::matmul::sgemm_rm(
            out_dim, in_dim, 1, 1.0, &weights, in_dim, input, 1, 1.0, output, 1,
        );
        self.activation.apply_f32_inplace(output);
    }

    /// All trainable parameters as `Node`s suitable for `Sgd::new` / `MeProp::new`.
    /// Length is `in_dim * out_dim + out_dim` (weights followed by biases).
    #[cfg(feature = "train")]
    pub fn parameters(&self) -> Vec<Node> {
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
}
