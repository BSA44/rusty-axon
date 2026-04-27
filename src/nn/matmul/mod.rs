//! Fused matmul tape — relocated from `engine::matmul` in Phase 6.
//!
//! Phase 6 splits the framework into a `train`-only autograd path (which
//! gates `engine`, `optim`, `loss`, and the visualization helpers) and an
//! always-on pure-`&[f32]` inference path.  `MatMulTape` straddles both: the
//! parameter buffers (`weights`, `bias`) are needed by the inference forward,
//! while the gradient buffers, the input snapshot, the upstream `Node` refs,
//! and the visit-count bookkeeping that drives the fused backward are
//! `train`-only.
//!
//! The three GEMM call sites — forward `y = W @ x + b`, backward `dW = d_out
//! @ xᵀ`, backward `dx = Wᵀ @ d_out` — go through the [`kernel::sgemm_rm`]
//! helper.  Phase 4 swaps that helper between [`kernel_naive`] and
//! [`kernel_mm`] (matrixmultiply, auto-NEON on aarch64) at compile time based
//! on the `matrixmultiply` / `naive-matmul` feature flags.
//!
//! ## Backward dispatch (`train` only)
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

use std::cell::{Ref, RefCell};
#[cfg(feature = "train")]
use std::cell::Cell;
use std::rc::Rc;

#[cfg(feature = "train")]
use crate::engine::ops::Operation;
#[cfg(feature = "train")]
use crate::engine::value::Node;

mod kernel;
pub(crate) mod kernel_naive;
#[cfg(feature = "matrixmultiply")]
pub(crate) mod kernel_mm;

pub(crate) use kernel::sgemm_rm;

/// Which buffer in a [`MatMulTape`] a [`ParamView`] points at.
#[cfg(feature = "train")]
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
/// A `ParamView` is paired with the `Node::Param` storage variant; calls to
/// `Node::get_value`, `set_value`, `get_gradient`, and `add_gradient` route
/// straight into `tape.weights[index]` / `tape.bias[index]` (or the matching
/// gradient buffer) via `RefCell::borrow{,_mut}`.  The fused matmul kernel
/// `matrixmultiply::sgemm` requires a contiguous `&[f32]`, which is why
/// parameters live in the tape's flat buffer rather than as one
/// `Rc<RefCell<f32>>` per scalar.
#[cfg(feature = "train")]
#[derive(Debug, Clone)]
pub struct ParamView {
    pub tape: Rc<MatMulTape>,
    pub kind: ParamKind,
    pub index: usize,
}

#[cfg(feature = "train")]
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

    /// Stable identity used by `Node`'s `Hash`/`Eq` impls and by
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
///
/// The gradient buffers, the input snapshot, the upstream `Node` refs, and
/// the visit-count bookkeeping are `cfg(feature = "train")`-only — pure
/// inference builds carry only the parameter buffers (`weights`, `bias`)
/// and the dimension fields.
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
    #[cfg(feature = "train")]
    pub input: RefCell<Vec<f32>>,

    /// Per-output upstream gradient, accumulated by the dispatch loop in
    /// `Node::backward` and consumed by `run_backward`.
    #[cfg(feature = "train")]
    pub d_out: RefCell<Vec<f32>>,
    /// Accumulated parameter gradients.  Reset by an explicit
    /// [`MatMulTape::reset_grads`] call (Phase 2 wires this into
    /// `Optimizer::zero_state`).
    #[cfg(feature = "train")]
    pub d_weights: RefCell<Vec<f32>>,
    #[cfg(feature = "train")]
    pub d_bias: RefCell<Vec<f32>>,
    /// Scratch buffer for `dx = Wᵀ d_out`.  Only populated when the inputs
    /// have a non-leaf operation.
    #[cfg(feature = "train")]
    pub d_input: RefCell<Vec<f32>>,

    /// `Some(inputs)` if any of the input `Node`s carries a non-`None`
    /// operation, i.e. their gradients still need to be back-propagated
    /// through the upstream graph.  `None` when the inputs are pure leaves
    /// (the dx matmul can then be skipped entirely).
    #[cfg(feature = "train")]
    pub upstream: RefCell<Option<Vec<Node>>>,

    /// Number of output `Node`s that have contributed to `d_out` in the
    /// current backward pass.  When this equals `out_dim`, `run_backward`
    /// fires.
    #[cfg(feature = "train")]
    pub visit_count: Cell<usize>,
    /// Set to `true` after `run_backward` has fired this iteration.  Cleared
    /// by `forward` (or by `reset_grads` if forward never re-runs).
    #[cfg(feature = "train")]
    pub backward_done: Cell<bool>,
    /// Guards `build_topo_recursive` against walking `upstream` once per
    /// output `Node` (`out_dim` redundant traversals).  The topo helper
    /// itself is responsible for clearing this flag once the topo is built.
    #[cfg(feature = "train")]
    pub topo_walked: Cell<bool>,
}

impl MatMulTape {
    /// Allocate a tape with the supplied weights and bias.
    ///
    /// Under `feature = "train"` all gradient and per-iteration buffers are
    /// zero-initialised; `upstream` is `None` until the first `forward`.
    /// Under inference-only builds those fields don't exist and the tape
    /// holds only the parameter buffers.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != out_dim * in_dim` or `bias.len() != out_dim`.
    pub fn new(in_dim: usize, out_dim: usize, weights: Vec<f32>, bias: Vec<f32>) -> Rc<Self> {
        // `weights` may be empty when the owning `Linear` is quantized
        // (Phase 7) and the f32 buffer is no longer needed.  Otherwise it
        // must be the full row-major `[out_dim, in_dim]` matrix.
        assert!(
            weights.is_empty() || weights.len() == out_dim * in_dim,
            "weights must be empty or row-major [out_dim, in_dim]"
        );
        assert_eq!(bias.len(), out_dim, "bias must be [out_dim]");
        Rc::new(Self {
            in_dim,
            out_dim,
            weights: RefCell::new(weights),
            bias: RefCell::new(bias),
            #[cfg(feature = "train")]
            input: RefCell::new(vec![0.0; in_dim]),
            #[cfg(feature = "train")]
            d_out: RefCell::new(vec![0.0; out_dim]),
            #[cfg(feature = "train")]
            d_weights: RefCell::new(vec![0.0; out_dim * in_dim]),
            #[cfg(feature = "train")]
            d_bias: RefCell::new(vec![0.0; out_dim]),
            #[cfg(feature = "train")]
            d_input: RefCell::new(vec![0.0; in_dim]),
            #[cfg(feature = "train")]
            upstream: RefCell::new(None),
            #[cfg(feature = "train")]
            visit_count: Cell::new(0),
            #[cfg(feature = "train")]
            backward_done: Cell::new(false),
            #[cfg(feature = "train")]
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
    /// owns that lifecycle.
    ///
    /// # Panics
    ///
    /// Panics if `inputs.len() != self.in_dim`.
    #[cfg(feature = "train")]
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

        // Forward kernel: y = W @ x + b.  Pre-load y with bias, then sgemm
        // with beta = 1 to accumulate.  Sized as [m=out_dim, k=in_dim, n=1]
        // — the column vector x has ldb = 1 because there is exactly one
        // column.
        let mut y = vec![0.0_f32; self.out_dim];
        {
            let weights = self.weights.borrow();
            let bias = self.bias.borrow();
            let input = self.input.borrow();
            y.copy_from_slice(&bias);
            sgemm_rm(
                self.out_dim,
                self.in_dim,
                1,
                1.0,
                &weights,
                self.in_dim,
                &input,
                1,
                1.0,
                &mut y,
                1,
            );
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
    /// `Node`s via `add_gradient`.
    #[cfg(feature = "train")]
    pub fn run_backward(self: &Rc<Self>) {
        debug_assert!(
            !self.backward_done.get(),
            "MatMulTape::run_backward fired twice in one pass"
        );

        let weights = self.weights.borrow();
        let input = self.input.borrow();
        let d_out = self.d_out.borrow();

        // dW += d_out ⊗ x   (rank-1 outer product)
        // Sized as [m=out_dim, k=1, n=in_dim].  d_out is [out, 1] so lda=1;
        // x is [1, in] so ldb=in.  beta=1 to accumulate across mini-batches.
        {
            let mut d_weights = self.d_weights.borrow_mut();
            sgemm_rm(
                self.out_dim,
                1,
                self.in_dim,
                1.0,
                &d_out,
                1,
                &input,
                self.in_dim,
                1.0,
                &mut d_weights,
                self.in_dim,
            );
        }

        // db[i] += d_out[i]
        {
            let mut d_bias = self.d_bias.borrow_mut();
            for i in 0..self.out_dim {
                d_bias[i] += d_out[i];
            }
        }

        // dx = d_outᵀ @ W; m=1, k=out_dim, n=in_dim.  d_out is treated as a
        // [1, out] row, W is [out, in], result is [1, in].  beta=0 because
        // d_input is scratch — we overwrite, then forward each component into
        // the matching upstream Node.
        let upstream_borrow = self.upstream.borrow();
        if let Some(upstream) = upstream_borrow.as_ref() {
            let mut d_input = self.d_input.borrow_mut();
            sgemm_rm(
                1,
                self.out_dim,
                self.in_dim,
                1.0,
                &d_out,
                self.out_dim,
                &weights,
                self.in_dim,
                0.0,
                &mut d_input,
                self.in_dim,
            );
            for (j, node) in upstream.iter().enumerate() {
                node.add_gradient(d_input[j]);
            }
        }

        self.backward_done.set(true);
    }

    /// Clear every transient and accumulated buffer the tape owns.
    /// Optimizers call this at the start of a new mini-batch.
    #[cfg(feature = "train")]
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

    #[cfg(feature = "train")]
    pub fn d_weights_ref(&self) -> Ref<'_, Vec<f32>> {
        self.d_weights.borrow()
    }

    #[cfg(feature = "train")]
    pub fn d_bias_ref(&self) -> Ref<'_, Vec<f32>> {
        self.d_bias.borrow()
    }

    #[cfg(feature = "train")]
    pub fn d_input_ref(&self) -> Ref<'_, Vec<f32>> {
        self.d_input.borrow()
    }
}

impl std::fmt::Debug for MatMulTape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("MatMulTape");
        dbg.field("in_dim", &self.in_dim).field("out_dim", &self.out_dim);
        #[cfg(feature = "train")]
        {
            dbg.field("visit_count", &self.visit_count.get())
                .field("backward_done", &self.backward_done.get());
        }
        dbg.finish()
    }
}

#[cfg(feature = "train")]
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

#[cfg(test)]
mod kernel_tests {
    //! Phase 4 acceptance: `kernel_mm` and `kernel_naive` agree to within
    //! `1e-5` on a randomly populated 64x64 GEMM, and on the three shapes
    //! `MatMulTape` actually issues (mat-vec, outer product, row-times-mat).
    //!
    //! Only runs when `matrixmultiply` is enabled (it has to be for the
    //! `kernel_mm` module to compile).

    #[cfg(feature = "matrixmultiply")]
    use super::kernel_mm::sgemm_rm as sgemm_mm;
    use super::kernel_naive::sgemm_rm as sgemm_naive;

    fn lcg(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((*seed >> 8) as f32 / ((1u32 << 23) as f32)) - 1.0
    }

    fn random_vec(n: usize, seed: &mut u32) -> Vec<f32> {
        (0..n).map(|_| lcg(seed)).collect()
    }

    #[allow(dead_code)] // only referenced when `matrixmultiply` is on
    fn assert_gemms_agree(a: &[f32], b: &[f32], lhs: &[f32], rhs: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{}: length mismatch", label);
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            assert!(
                diff < tol,
                "{} mismatch at {}: naive={} mm={} diff={} \
                 (lhs.len={}, rhs.len={})",
                label,
                i,
                x,
                y,
                diff,
                lhs.len(),
                rhs.len(),
            );
        }
    }

    #[cfg(feature = "matrixmultiply")]
    #[test]
    fn test_kernel_agreement_64x64() {
        let mut seed = 0xDEADBEEF_u32;
        let m = 64;
        let k = 64;
        let n = 64;
        let a = random_vec(m * k, &mut seed);
        let b = random_vec(k * n, &mut seed);

        let mut c_naive = vec![0.0_f32; m * n];
        let mut c_mm = vec![0.0_f32; m * n];

        sgemm_naive(m, k, n, 1.0, &a, k, &b, n, 0.0, &mut c_naive, n);
        sgemm_mm(m, k, n, 1.0, &a, k, &b, n, 0.0, &mut c_mm, n);

        assert_gemms_agree(&c_naive, &c_mm, &a, &b, 1e-3, "64x64 GEMM");
    }

    #[cfg(feature = "matrixmultiply")]
    #[test]
    fn test_kernel_agreement_matvec_shape() {
        // Forward shape: m=out, k=in, n=1.  Shape used by Linear::forward.
        let mut seed = 0x1234_5678_u32;
        let m = 32;
        let k = 17;
        let w = random_vec(m * k, &mut seed);
        let x = random_vec(k, &mut seed);
        let bias = random_vec(m, &mut seed);

        let mut y_naive = bias.clone();
        let mut y_mm = bias.clone();

        sgemm_naive(m, k, 1, 1.0, &w, k, &x, 1, 1.0, &mut y_naive, 1);
        sgemm_mm(m, k, 1, 1.0, &w, k, &x, 1, 1.0, &mut y_mm, 1);

        assert_gemms_agree(&y_naive, &y_mm, &w, &x, 1e-4, "matvec forward");
    }

    #[cfg(feature = "matrixmultiply")]
    #[test]
    fn test_kernel_agreement_outer_product_shape() {
        // Backward dW shape: m=out, k=1, n=in.  Outer product d_out ⊗ x.
        let mut seed = 0xC0FFEE_u32;
        let m = 24;
        let n = 19;
        let d_out = random_vec(m, &mut seed);
        let x = random_vec(n, &mut seed);

        let mut dw_naive = vec![0.0_f32; m * n];
        let mut dw_mm = vec![0.0_f32; m * n];

        sgemm_naive(m, 1, n, 1.0, &d_out, 1, &x, n, 0.0, &mut dw_naive, n);
        sgemm_mm(m, 1, n, 1.0, &d_out, 1, &x, n, 0.0, &mut dw_mm, n);

        assert_gemms_agree(&dw_naive, &dw_mm, &d_out, &x, 1e-5, "outer product dW");
    }

    #[cfg(feature = "matrixmultiply")]
    #[test]
    fn test_kernel_agreement_row_times_matrix_shape() {
        // Backward dx shape: m=1, k=out, n=in.  Row vector times matrix.
        let mut seed = 0xBADBEEF_u32;
        let k = 28;
        let n = 13;
        let d_out = random_vec(k, &mut seed);
        let w = random_vec(k * n, &mut seed);

        let mut dx_naive = vec![0.0_f32; n];
        let mut dx_mm = vec![0.0_f32; n];

        sgemm_naive(1, k, n, 1.0, &d_out, k, &w, n, 0.0, &mut dx_naive, n);
        sgemm_mm(1, k, n, 1.0, &d_out, k, &w, n, 0.0, &mut dx_mm, n);

        assert_gemms_agree(&dx_naive, &dx_mm, &d_out, &w, 1e-4, "row-times-matrix dx");
    }

    #[test]
    fn test_naive_gemm_against_textbook_reference() {
        // Even when matrixmultiply is off, the naive kernel must compute the
        // textbook formula exactly.  Tiny 2x3 . 3x2 case worked out by hand.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // [3, 2]
        let mut c = vec![0.0_f32; 4];
        sgemm_naive(2, 3, 2, 1.0, &a, 3, &b, 2, 0.0, &mut c, 2);
        // c[0,0] = 1*7 + 2*9 + 3*11 = 58
        // c[0,1] = 1*8 + 2*10 + 3*12 = 64
        // c[1,0] = 4*7 + 5*9 + 6*11 = 139
        // c[1,1] = 4*8 + 5*10 + 6*12 = 154
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }
}
