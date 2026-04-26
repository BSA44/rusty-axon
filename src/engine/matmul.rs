//! Fused matmul op for the autograd engine.
//!
//! A `Linear` layer in `rusty-axon` runs **one** matrix-vector product per
//! forward pass and **two** matmuls per backward pass, but every individual
//! parameter still appears in the scalar `Value` graph as a leaf (`Param` view
//! in Phase 2).  To keep the per-`Node` overhead tiny we factor all of the
//! shared state — weight matrix, bias, input snapshot, gradient buffers,
//! upstream-Node refs, and the visit-count bookkeeping that drives the fused
//! backward — into [`MatMulTape`].  Each output `Node` produced by a matmul
//! carries `(Rc<MatMulTape>, output_index)` and nothing else.
//!
//! The kernel itself is the naive scalar fallback for Phase 1.  Phase 4 will
//! swap the three call sites in [`MatMulTape::run_backward`] /
//! [`MatMulTape::forward_into`] for `matrixmultiply::sgemm`.
//!
//! ## Backward dispatch
//!
//! For a tape with `out_dim` outputs, every backward pass:
//!
//! 1. Each output `Node` accumulates its incoming gradient into
//!    `tape.d_out[output_index]` and bumps `tape.visit_count`.
//! 2. When `visit_count == out_dim`, the tape fires
//!    [`MatMulTape::run_backward`] exactly once: it accumulates `dW = d_out ⊗
//!    x` into `d_weights`, accumulates `db = d_out` into `d_bias`, and (if the
//!    inputs were not leaves) propagates `dx = Wᵀ d_out` into the upstream
//!    `Node`s via `Node::add_gradient`.
//!
//! This trick assumes every output `Node` is reachable from the loss so that
//! the topo walk pulls it into the backward pass.  For the MLPs we target
//! (softmax + cross-entropy over every logit) that is always true.  A
//! `debug_assert!` in [`Drop`] flags the rare case where it is not.

use std::cell::{Cell, Ref, RefCell};
use std::rc::Rc;

use crate::engine::ops::Operation;
use crate::engine::value::Node;

/// Which buffer in a [`MatMulTape`] a [`ParamView`] points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamKind {
    Weight,
    Bias,
}

/// Read-through / write-through view of one scalar parameter living inside a
/// [`MatMulTape`].  Used by `nn::Linear` to expose its flat `Vec<f32>` weight
/// and bias buffers as a `Vec<Node>` so existing optimizers (`Sgd`, `MeProp`)
/// keep working unchanged.
///
/// A `ParamView` is paired with the [`Node::Param`] storage variant; calls to
/// `Node::get_value`, `set_value`, `get_gradient`, and `add_gradient` route
/// straight into `tape.weights[index]` / `tape.bias[index]` (or the matching
/// gradient buffer) via `RefCell::borrow{,_mut}`.  The fused matmul kernel
/// `matrixmultiply::sgemm` (Phase 4) requires a contiguous `&[f32]`, which is
/// why parameters live in the tape's flat buffer rather than as one
/// `Rc<RefCell<f32>>` per scalar.
#[derive(Debug, Clone)]
pub struct ParamView {
    pub tape: Rc<MatMulTape>,
    pub kind: ParamKind,
    pub index: usize,
}

impl ParamView {
    pub fn get_value(&self) -> f32 {
        match self.kind {
            ParamKind::Weight => self.tape.weights.borrow()[self.index],
            ParamKind::Bias => self.tape.bias.borrow()[self.index],
        }
    }

    pub fn set_value(&self, value: f32) {
        match self.kind {
            ParamKind::Weight => self.tape.weights.borrow_mut()[self.index] = value,
            ParamKind::Bias => self.tape.bias.borrow_mut()[self.index] = value,
        }
    }

    pub fn get_gradient(&self) -> f32 {
        match self.kind {
            ParamKind::Weight => self.tape.d_weights.borrow()[self.index],
            ParamKind::Bias => self.tape.d_bias.borrow()[self.index],
        }
    }

    pub fn set_gradient(&self, gradient: f32) {
        match self.kind {
            ParamKind::Weight => self.tape.d_weights.borrow_mut()[self.index] = gradient,
            ParamKind::Bias => self.tape.d_bias.borrow_mut()[self.index] = gradient,
        }
    }

    pub fn add_gradient(&self, gradient: f32) {
        match self.kind {
            ParamKind::Weight => self.tape.d_weights.borrow_mut()[self.index] += gradient,
            ParamKind::Bias => self.tape.d_bias.borrow_mut()[self.index] += gradient,
        }
    }

    /// Stable identity used by [`Node`]'s `Hash`/`Eq` impls and by
    /// optimizers' tape-deduplication logic.
    pub fn tape_ptr(&self) -> *const MatMulTape {
        Rc::as_ptr(&self.tape)
    }
}

/// Side struct shared by every output `Node` of one fused matmul op.
///
/// All parameter and per-iteration state lives here so the per-`Node` payload
/// stays at one `Rc` plus one `usize`.  A `Linear` layer keeps a single
/// `Rc<MatMulTape>` alive for its lifetime and re-uses it across forward
/// passes (the `input` snapshot and `upstream` refs are overwritten each
/// `forward`).
pub struct MatMulTape {
    pub in_dim: usize,
    pub out_dim: usize,

    /// Row-major weight matrix `[out_dim, in_dim]`.  Lives here permanently
    /// because Phase 2's `ParamView` indexes directly into it.
    pub weights: RefCell<Vec<f32>>,
    /// Bias vector `[out_dim]`.
    pub bias: RefCell<Vec<f32>>,

    /// Snapshot of the input vector taken at forward time.  Overwritten on
    /// every `forward`.
    pub input: RefCell<Vec<f32>>,

    /// Per-output upstream gradient, accumulated by the dispatch loop in
    /// `Node::backward` and consumed by `run_backward`.
    pub d_out: RefCell<Vec<f32>>,
    /// Accumulated parameter gradients.  Reset by an explicit
    /// [`MatMulTape::reset_grads`] call (Phase 2 wires this into
    /// `Optimizer::zero_state`).
    pub d_weights: RefCell<Vec<f32>>,
    pub d_bias: RefCell<Vec<f32>>,
    /// Scratch buffer for `dx = Wᵀ d_out`.  Only populated when the inputs
    /// have a non-leaf operation.
    pub d_input: RefCell<Vec<f32>>,

    /// `Some(inputs)` if any of the input `Node`s carries a non-`None`
    /// operation, i.e. their gradients still need to be back-propagated
    /// through the upstream graph.  `None` when the inputs are pure leaves
    /// (the dx matmul can then be skipped entirely).
    pub upstream: RefCell<Option<Vec<Node>>>,

    /// Number of output `Node`s that have contributed to `d_out` in the
    /// current backward pass.  When this equals `out_dim`, `run_backward`
    /// fires.
    pub visit_count: Cell<usize>,
    /// Set to `true` after `run_backward` has fired this iteration.  Cleared
    /// by `forward` (or by `reset_grads` if forward never re-runs).
    pub backward_done: Cell<bool>,
    /// Guards `build_topo_recursive` against walking `upstream` once per
    /// output `Node` (`out_dim` redundant traversals).  The topo helper
    /// itself is responsible for clearing this flag once the topo is built.
    pub topo_walked: Cell<bool>,
}

impl MatMulTape {
    /// Allocate a tape with the supplied weights and bias.  All gradient and
    /// per-iteration buffers are zero-initialised; `upstream` is `None` until
    /// the first `forward`.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != out_dim * in_dim` or `bias.len() != out_dim`.
    pub fn new(in_dim: usize, out_dim: usize, weights: Vec<f32>, bias: Vec<f32>) -> Rc<Self> {
        assert_eq!(
            weights.len(),
            out_dim * in_dim,
            "weights must be row-major [out_dim, in_dim]"
        );
        assert_eq!(bias.len(), out_dim, "bias must be [out_dim]");
        Rc::new(Self {
            in_dim,
            out_dim,
            weights: RefCell::new(weights),
            bias: RefCell::new(bias),
            input: RefCell::new(vec![0.0; in_dim]),
            d_out: RefCell::new(vec![0.0; out_dim]),
            d_weights: RefCell::new(vec![0.0; out_dim * in_dim]),
            d_bias: RefCell::new(vec![0.0; out_dim]),
            d_input: RefCell::new(vec![0.0; in_dim]),
            upstream: RefCell::new(None),
            visit_count: Cell::new(0),
            backward_done: Cell::new(false),
            topo_walked: Cell::new(false),
        })
    }

    /// Forward pass: snapshot `inputs`, compute `y = W @ x + b`, and emit one
    /// `Operation::MatMul`-tagged output `Node` per row of `W`.
    ///
    /// Per-iteration state (`d_out`, `visit_count`, `backward_done`) is reset
    /// here so a `forward` -> `backward` -> `forward` -> `backward` sequence
    /// works without an explicit `reset_grads` between iterations.  The
    /// accumulated `d_weights` / `d_bias` are *not* touched — the optimizer
    /// owns that lifecycle (Phase 2).
    ///
    /// # Panics
    ///
    /// Panics if `inputs.len() != self.in_dim`.
    pub fn forward(self: &Rc<Self>, inputs: &[Node]) -> Vec<Node> {
        assert_eq!(
            inputs.len(),
            self.in_dim,
            "input length must match in_dim ({})",
            self.in_dim
        );

        {
            let mut input = self.input.borrow_mut();
            for (i, node) in inputs.iter().enumerate() {
                input[i] = node.get_value();
            }
        }

        // Detect whether any input is itself the output of an operation.  If
        // every input is a leaf we can skip the `dx = Wᵀ d_out` matmul on
        // backward — pure inference call sites and parameter-only chains
        // both hit this fast path.
        let has_upstream = inputs
            .iter()
            .any(|n| !matches!(n.get_operation(), Operation::None));
        *self.upstream.borrow_mut() = if has_upstream {
            Some(inputs.to_vec())
        } else {
            None
        };

        // Forward kernel: y = W @ x + b.  Naive scalar fallback for Phase 1;
        // Phase 4 will swap to `matrixmultiply::sgemm`.
        let mut y = vec![0.0_f32; self.out_dim];
        {
            let weights = self.weights.borrow();
            let bias = self.bias.borrow();
            let input = self.input.borrow();
            for i in 0..self.out_dim {
                let mut acc = bias[i];
                let row_offset = i * self.in_dim;
                for j in 0..self.in_dim {
                    acc += weights[row_offset + j] * input[j];
                }
                y[i] = acc;
            }
        }

        // Reset per-iteration state.  `d_weights` / `d_bias` accumulate across
        // iterations until `reset_grads` is called.
        {
            let mut d_out = self.d_out.borrow_mut();
            for v in d_out.iter_mut() {
                *v = 0.0;
            }
        }
        self.visit_count.set(0);
        self.backward_done.set(false);
        self.topo_walked.set(false);

        // Emit one output Node per row of W, each tagged with the shared tape.
        y.into_iter()
            .enumerate()
            .map(|(i, value)| {
                Node::with_operation(
                    value,
                    Operation::MatMul {
                        tape: Rc::clone(self),
                        output_index: i,
                    },
                )
            })
            .collect()
    }

    /// Fused backward pass.  Called by the dispatch loop in `Node::backward`
    /// when every output `Node` of this tape has accumulated its gradient
    /// into `d_out`.
    ///
    /// Accumulates `dW += d_out ⊗ input` and `db += d_out`, then (if the
    /// inputs were not leaves) propagates `dx = Wᵀ d_out` into the upstream
    /// `Node`s via `add_gradient`.  All three operations are naive scalar
    /// fallbacks; Phase 4 substitutes `matrixmultiply::sgemm`.
    pub fn run_backward(self: &Rc<Self>) {
        debug_assert!(
            !self.backward_done.get(),
            "MatMulTape::run_backward fired twice in one pass"
        );

        let weights = self.weights.borrow();
        let input = self.input.borrow();
        let d_out = self.d_out.borrow();

        // dW[i, j] += d_out[i] * input[j]   (rank-1 update)
        {
            let mut d_weights = self.d_weights.borrow_mut();
            for i in 0..self.out_dim {
                let row_offset = i * self.in_dim;
                let g = d_out[i];
                if g == 0.0 {
                    continue;
                }
                for j in 0..self.in_dim {
                    d_weights[row_offset + j] += g * input[j];
                }
            }
        }

        // db[i] += d_out[i]
        {
            let mut d_bias = self.d_bias.borrow_mut();
            for i in 0..self.out_dim {
                d_bias[i] += d_out[i];
            }
        }

        // dx[j] = sum_i W[i, j] * d_out[i]; propagate to upstream Nodes.
        let upstream_borrow = self.upstream.borrow();
        if let Some(upstream) = upstream_borrow.as_ref() {
            let mut d_input = self.d_input.borrow_mut();
            for j in 0..self.in_dim {
                let mut acc = 0.0_f32;
                for i in 0..self.out_dim {
                    acc += weights[i * self.in_dim + j] * d_out[i];
                }
                d_input[j] = acc;
            }
            for (j, node) in upstream.iter().enumerate() {
                node.add_gradient(d_input[j]);
            }
        }

        self.backward_done.set(true);
    }

    /// Clear every transient and accumulated buffer the tape owns.
    /// Optimizers call this at the start of a new mini-batch (Phase 2 wires
    /// it into `Optimizer::zero_state`).
    pub fn reset_grads(&self) {
        for v in self.d_out.borrow_mut().iter_mut() {
            *v = 0.0;
        }
        for v in self.d_weights.borrow_mut().iter_mut() {
            *v = 0.0;
        }
        for v in self.d_bias.borrow_mut().iter_mut() {
            *v = 0.0;
        }
        for v in self.d_input.borrow_mut().iter_mut() {
            *v = 0.0;
        }
        self.visit_count.set(0);
        self.backward_done.set(false);
        self.topo_walked.set(false);
    }

    pub fn weights_ref(&self) -> Ref<'_, Vec<f32>> {
        self.weights.borrow()
    }

    pub fn bias_ref(&self) -> Ref<'_, Vec<f32>> {
        self.bias.borrow()
    }

    pub fn d_weights_ref(&self) -> Ref<'_, Vec<f32>> {
        self.d_weights.borrow()
    }

    pub fn d_bias_ref(&self) -> Ref<'_, Vec<f32>> {
        self.d_bias.borrow()
    }

    pub fn d_input_ref(&self) -> Ref<'_, Vec<f32>> {
        self.d_input.borrow()
    }
}

impl std::fmt::Debug for MatMulTape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatMulTape")
            .field("in_dim", &self.in_dim)
            .field("out_dim", &self.out_dim)
            .field("visit_count", &self.visit_count.get())
            .field("backward_done", &self.backward_done.get())
            .finish()
    }
}

impl Drop for MatMulTape {
    fn drop(&mut self) {
        // If a backward pass started accumulating into `d_out` but the trigger
        // condition (`visit_count == out_dim`) never fired, the user has
        // routed gradients into a tape whose outputs are not all reachable
        // from the loss.  Phase 1 only supports the all-outputs-used path.
        if !self.backward_done.get() && self.visit_count.get() != 0 {
            for &g in self.d_out.borrow().iter() {
                debug_assert!(
                    g == 0.0,
                    "MatMulTape dropped with pending gradient: visit_count={} of out_dim={}; \
                     ensure every matmul output Node is reachable from the loss",
                    self.visit_count.get(),
                    self.out_dim
                );
            }
        }
    }
}
