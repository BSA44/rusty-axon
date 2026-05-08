# rusty-axon Paper-Grade Rework — Implementation Plan

This document is the authoritative implementation plan for the v0.3 rework of `rusty-axon` that reframes the project as a **training-capable edge framework**. It is intended to be executed across multiple Claude sessions, one Phase per session.

---

## Context

`rusty-axon` is currently a [micrograd](https://github.com/karpathy/micrograd)-style Rust autograd: every value is `Rc<RefCell<Value>>` holding `f64`, every neuron is a hand-rolled scalar dot product. It works as an educational MLP framework (XOR, MNIST 95%+) but has none of the artifacts a "training-capable edge framework" paper needs: no fused matmul, no kernel acceleration, no model serialization, no quantization, no inference-only build, no on-device demo.

This rework reframes the project around the paper thesis:

> Existing edge runtimes are inference-only (TFLite Micro, tract, MicroFlow, ncnn). Existing Rust ML frameworks that do both training and inference (Candle, Burn) are too heavy for Pi Zero-class hardware or are inference-focused for on-device fine-tuning. **rusty-axon** fills that gap: a minimal, pure-Rust, memory-safe framework that **trains and infers from the same codebase**, demonstrated by an on-device fine-tuning workflow on Raspberry Pi Zero 2 W.

**Hard constraint:** the scalar `Value`-based autograd must stay. The optimization is to add a **single fused MatMul op** inside `Linear` — one matmul forward, two matmuls backward — while everything else (activations, loss, scalar ops) keeps flowing through the existing `Value` graph.

**Decisions captured up front:**
- **RPi target:** `aarch64-unknown-linux-gnu` only (64-bit Pi OS). No hand-written NEON kernels — `matrixmultiply` auto-uses NEON on aarch64.
- **Demo scope:** Two demos — MNIST personalization fine-tune **and** synthetic sensor-drift adaptation.
- **Engine precision:** Migrate the scalar engine from `f64` to `f32` (paper-pure single-precision; matches MatMul, INT8, and inference paths).
- **Paper baselines:** Burn, Candle, TFLite Micro/MicroFlow.

---

## Conventions used throughout

- Crate version bumps to `0.3.0` at Phase 0.
- Engine is `f32` end-to-end after Phase 0.5. No `f64` mixed-precision boundary.
- Cargo features:
  - `default = ["train", "matrixmultiply"]`
  - `train` — engine, autograd, optim, loss, visualization
  - `inference` — pure-`&[f32]` forward path; engine module gated out
  - `matrixmultiply` — link `matrixmultiply` crate (auto-NEON on aarch64)
  - `naive-matmul` — force the naive kernel for the speedup-vs-NEON table
  - `quant-i8` — INT8 PTQ load/save + dequant-fused matmul
- Cargo profiles: stock `release` plus a `release-edge` profile (`lto = "fat"`, `codegen-units = 1`, `panic = "abort"`, `opt-level = "z"`, `strip = "symbols"`) used for the binary-size table and shipped artifacts.
- One Phase ≈ one focused implementation session. The plan totals **13 phases** (0 through 11 plus K-paper).

---

## Phase 0 — Repo hygiene, feature flags, profiles, CI scaffolding

**Goal.** Lay down build infrastructure so subsequent phases land cleanly.

**Touched files.**
- [Cargo.toml](../Cargo.toml) — features, dev-deps, profiles, optional `matrixmultiply` dep.
- [src/lib.rs](../src/lib.rs) — gate `pub mod engine`, `pub mod optim`, `pub mod loss` behind `cfg(feature = "train")`.
- [src/main.rs](../src/main.rs) — gate behind `train` (or move XOR demo into `examples/xor_problem.rs` and remove `main.rs` from the lib package).

**New files.**
- `.cargo/config.toml` — placeholder; populated in Phase 9.
- `rust-toolchain.toml` — pin a stable toolchain for reproducibility.
- `.github/workflows/ci.yml` — `fmt + clippy + test` matrix across `train`, `inference`, `inference,quant-i8`. Add `windows-latest` and `ubuntu-latest`.
- `clippy.toml`, `rustfmt.toml`.

**Acceptance.**
- `cargo check --no-default-features --features train` — green.
- `cargo check --no-default-features --features inference` — green (no-op until Phase 6, but type-checks the gating).
- `cargo build --profile release-edge` — green.
- CI matrix green on host x86_64.

**Risks / open questions.**
- `panic = "abort"` removes unwind metadata (~30–80 KB savings); we don't use `catch_unwind`.
- Pin `matrixmultiply = "=0.3.9"` exactly so timing tables are reproducible across rebuilds.

**Dependencies.** None.

---

## Phase 0.5 — Migrate engine `f64 → f32`

**Goal.** Make `Value::value: f32` and `Value::gradient: f32` everywhere. Update every test, example, optimizer, and loss accordingly.

**Touched files.**
- [src/engine/value.rs](../src/engine/value.rs) — `value: f32`, `gradient: f32`. Update `From<f64>`, `From<i32>`, `From<i64>` impls to cast to `f32`. Update `pow`, `exp`, `log` arg types.
- [src/engine/ops.rs](../src/engine/ops.rs) — only metadata; types are inferred via `Node`. Likely no changes.
- [src/engine/tests.rs](../src/engine/tests.rs) — adjust tolerances from `1e-9` to `1e-5` where needed.
- [src/nn/neuron.rs](../src/nn/neuron.rs), [src/nn/layer.rs](../src/nn/layer.rs), [src/nn/mlp.rs](../src/nn/mlp.rs) — `f32` everywhere.
- [src/nn/activations.rs](../src/nn/activations.rs) — `f32::exp`, etc.
- [src/loss/](../src/loss/) — `f32`.
- [src/optim/](../src/optim/) — `f32`.
- [examples/](../examples/) — sample data literals (`0.0` is fine; explicit casts on inputs to `Node::from`).
- [python-tests/prepare_mnist.py](../python-tests/prepare_mnist.py) — output stays the same (CSV pixel values 0..1); Rust loader parses to `f32`.

**Acceptance.**
- All existing tests green within `1e-5` tolerance.
- All existing examples (XOR, MNIST classifier, bench_*) compile and produce the same qualitative results (MNIST ≥85% in 5 epochs, XOR converges).
- `std::mem::size_of::<Value>()` drops from 16 to 8 bytes.

**Risks.**
- `Node::from(f64)` is used in some examples — keep the impl (cast to `f32` lossily) so callers don't break.
- Numerical regressions on MNIST: f32 cross-entropy gradients can blow up if logits are large. Add `epsilon = 1e-7_f32` clamps already present in `cross_entropy.rs`. Verify.
- `f32` accumulation in MSE for large batches loses precision vs `f64`. For MNIST batch 32 this is fine; document.

**Dependencies.** Phase 0.

---

## Phase 1 — Fused `MatMul` op + `MatMulTape` in the Value engine

**Goal.** Add a `MatMul` variant to `Operation` that captures one weight matrix + bias + input vector as a side struct shared via `Rc`, so each output `Node` only carries `(Rc<MatMulTape>, output_index)`.

**Touched files.**
- [src/engine/ops.rs](../src/engine/ops.rs) — new variant `MatMul { tape: Rc<MatMulTape>, output_index: usize }`.
- [src/engine/value.rs](../src/engine/value.rs) — extend `build_topo_recursive` and `backward` to handle `MatMul`. Update `to_dot` and any equality/hash code.
- [src/engine/mod.rs](../src/engine/mod.rs) — re-exports.
- [src/engine/tests.rs](../src/engine/tests.rs) — gradient correctness tests against a scalar reference.

**New files.**
- `src/engine/matmul.rs` — `MatMulTape` struct + kernel call sites + shared backward.

**Key types.**

```rust
pub struct MatMulTape {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weights: RefCell<Vec<f32>>,        // row-major [out, in]
    pub bias:    RefCell<Vec<f32>>,        // [out]
    pub input:   RefCell<Vec<f32>>,        // [in], snapshot at forward
    pub d_out:   RefCell<Vec<f32>>,        // [out], filled by output Nodes during backward
    pub d_weights: RefCell<Vec<f32>>,      // [out, in]
    pub d_bias:    RefCell<Vec<f32>>,      // [out]
    pub d_input:   RefCell<Vec<f32>>,      // [in]
    pub upstream:  Option<Vec<Node>>,      // None if inputs are leaves
    pub visit_count:   Cell<usize>,        // resets to 0 at start of each backward
    pub backward_done: Cell<bool>,
    pub topo_walked:   Cell<bool>,         // ensures upstream is recursed once per topo build
}

pub enum Operation {
    // existing variants ...
    MatMul { tape: Rc<MatMulTape>, output_index: usize },
}
```

**Backward dispatch (the tricky part).**
Each output Node carrying `MatMul { tape, output_index }` runs:
1. `tape.d_out[output_index] += grad`
2. `tape.visit_count += 1`
3. If `visit_count == out_dim`, fire `tape.run_backward()` once:
   - `dW = d_out · x^T` via `sgemm` (outer product)
   - if `tape.upstream.is_some()`: `dx = W^T · d_out` via `sgemm`, then `upstream[j].add_gradient(dx[j])` for each j
   - `db += d_out`
4. After `loss.backward()` finishes, the optimizer's `zero_state` resets `visit_count`, `backward_done`, `d_out`, `d_weights`, `d_bias`, `d_input` to zero (Phase 2 wires this in).

**Topo recursion.** In `build_topo_recursive`, the `MatMul` arm recurses into `tape.upstream` (if `Some`) **once across all output Nodes that share the tape** — guarded by `tape.topo_walked`. Reset `topo_walked` to `false` at the start of every `backward()`.

**Acceptance criteria.**
- Random `8 × 4` weight, random `[4]` input: gradient matches a scalar reference (sum of `w_ij * x_j + b_i` per output) within `1e-5`. Includes a chained-input case (input vector itself produced by a prior `(Node + Node) * Node` chain) so upstream propagation is tested.
- Existing scalar engine tests still green.
- `std::mem::size_of::<Operation>() <= 64` bytes (regression test).

**Risks / open questions.**
- The `visit_count` trick assumes every output Node is reachable from the loss. For MNIST + softmax + cross-entropy, this is true. Document the constraint in `MatMulTape`'s doc-comment; add a `Drop` impl that `debug_assert!`s `backward_done == true` if any `d_out` entry is non-zero.
- Reset semantics must align with `Optimizer::zero_state` — Phase 2 is responsible for hooking `MatMulTape::reset_grads()` into the optimizer.

**Dependencies.** Phase 0.5.

---

## Phase 2 — `Linear` layer using fused MatMul; `ParamView` for `parameters()`

**Goal.** Implement `Linear` whose weights live as a flat `Vec<f32>` inside `Rc<MatMulTape>`, but whose `parameters()` continues to return `Vec<Node>` so `Sgd`/`MeProp` work unchanged.

**Touched files.**
- [src/engine/value.rs](../src/engine/value.rs) — refactor `Node` into an enum-backed type:

  ```rust
  enum NodeStorage {
      Owned(Rc<RefCell<Value>>),
      Param(ParamView),
  }
  pub struct Node { storage: NodeStorage }
  ```

  Every accessor (`get_value`, `set_value`, `add_gradient`, `zero_gradient`, `get_gradient`, `get_operation`) becomes a `match`. `Param` Nodes return `Operation::None` from `get_operation()` (they are leaves) and route reads/writes into the tape.
- [src/nn/mod.rs](../src/nn/mod.rs) — `pub mod linear; pub use linear::Linear;`
- [src/lib.rs](../src/lib.rs) — re-export `Linear`.

**New files.**
- `src/nn/linear.rs` — the `Linear` layer.
- `src/nn/param_view.rs` — `ParamView { tape, kind, index }` plus the `Node::from_param_view` constructor.

**API surface.**

```rust
pub struct Linear {
    tape: Rc<MatMulTape>,
    activation: Activations,
    cached_params: Vec<Node>,
}

impl Linear {
    pub fn new(in_dim: usize, out_dim: usize, activation: Activations) -> Self;
    pub fn with_weights(weights: Vec<f32>, bias: Vec<f32>, activation: Activations) -> Self;
    pub fn forward(&self, inputs: &[Node]) -> Vec<Node>;          // train path
    pub fn parameters(&self) -> Vec<Node>;                         // for optimizer
    pub fn in_dim(&self) -> usize;
    pub fn out_dim(&self) -> usize;
    pub fn weights(&self) -> std::cell::Ref<'_, Vec<f32>>;
    pub fn bias(&self) -> std::cell::Ref<'_, Vec<f32>>;
    pub fn infer_into_f32(&self, input: &[f32], output: &mut [f32]);  // always-on
}
```

**Why `ParamView`, not `Rc<RefCell<f32>>` per parameter.** A `Vec<RefCell<f32>>` is contiguous in `RefCell<f32>` cells, but reinterpret-casting `&[RefCell<f32>]` to `&[f32]` to feed `sgemm` is UB by language rule. The flat `Vec<f32>` in the tape is required for `matrixmultiply`, so it must be the source of truth. `ParamView` is the only sound option that keeps the optimizer API (`Vec<Node>`) unchanged.

**Linear::forward flow.**
1. Snapshot `inputs[*].get_value()` into `tape.input`.
2. Detect upstream: if any input Node has `Operation != None`, store `Some(inputs.to_vec())` in `tape.upstream`; else `None`.
3. `y.copy_from_slice(&bias); sgemm(y += W @ x)`.
4. For each `i` in `0..out_dim`, build a Node with `Operation::MatMul { tape, output_index: i }` and value `y[i]`, then apply `activation` (which produces a separate scalar `Operation::Exp/Sub/Div` chain — only Linear is fused).

**Optimizer integration.** Add a `reset_grads()` method on `MatMulTape` and call it from `Sgd::zero_state` and `MeProp::zero_state` for every parameter Node that is a `Param` view (skip duplicates by deduplicating tape `Rc` pointers).

**Acceptance.**
- Train `Linear(2, 1, None)` for 1000 steps on `y = 2x₁ + 3x₂ + 1`; learned weights within 1% of `[2, 3]`, bias within 1% of `1`.
- `Linear::parameters().len() == in*out + out`.
- MeProp on `Linear(8, 4)` selects top-k gradients correctly (test against a hand-computed reference).

**Risks.**
- `Node` enum bump from 8 → 24 bytes per stack/struct field. Acceptable; verify with `mem::size_of`.
- `tape.upstream` holds `Vec<Node>`; gradient cycles are impossible because tape never holds Nodes pointing at itself.
- `cached_params` keeps `Rc<MatMulTape>` clones alive; safe.

**Dependencies.** Phase 1.

---

## Phase 3 — Backward-compat shim: `Mlp` uses `Linear` internally

**Goal.** `Mlp::new(&[784, 64, 32, 10], &[ReLU, ReLU, None])` builds `Vec<Linear>` internally, but every existing example compiles and runs unchanged.

**Touched files.**
- [src/nn/mlp.rs](../src/nn/mlp.rs) — replace `Vec<Layer>` with `Vec<Linear>`. Public API (`new`, `forward`, `parameters`) unchanged.
- [src/nn/layer.rs](../src/nn/layer.rs), [src/nn/neuron.rs](../src/nn/neuron.rs) — leave as-is. They become the **legacy scalar baseline** used by Phase 8's speedup-vs-fused benchmark.

**New API.**
- `Mlp::with_layers(layers: Vec<Linear>) -> Self`
- `Mlp::layer(&self, idx: usize) -> &Linear`
- `Mlp::parameters_for_layers(&self, range: Range<usize>) -> Vec<Node>` — for partial-layer fine-tune in Phase 11.

**Acceptance.**
- All 12 existing examples build and run.
- MNIST classifier reaches ≥85% test accuracy in 5 epochs (sanity for fused Linear correctness).
- Regression test: identical-seed `Mlp` (legacy `Neuron`-based) and `Mlp` (Linear-based) — forward outputs match within `1e-4`, gradients within `1e-4`.

**Risks.** Tolerances at `1e-4` because cumulative `f32` rounding on a 784→64 dot product accumulates ~`784 * eps_f32 ≈ 1e-4`.

**Dependencies.** Phase 2.

---

## Phase 4 — `matrixmultiply` integration + naive fallback kernel

**Goal.** Wire `matrixmultiply::sgemm` into `MatMulTape`'s three call sites. Provide a naive fallback for the `naive-matmul` feature flag (used by the speedup table in Phase 8).

**Touched files.**
- `src/engine/matmul.rs`

**New files.**
- `src/engine/matmul/kernel.rs` (module split)
- `src/engine/matmul/kernel_naive.rs`
- `src/engine/matmul/kernel_mm.rs`

**Internal API.**

```rust
pub(crate) fn sgemm_rm(
    m: usize, k: usize, n: usize,
    alpha: f32, a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32, c: &mut [f32], ldc: usize,
);
```

Selected at compile time:

```rust
#[cfg(all(feature = "matrixmultiply", not(feature = "naive-matmul")))]
pub(crate) use kernel_mm::sgemm_rm;
#[cfg(any(not(feature = "matrixmultiply"), feature = "naive-matmul"))]
pub(crate) use kernel_naive::sgemm_rm;
```

**Three call sites in `MatMulTape`** (all row-major `[m, k] · [k, n] -> [m, n]`):
- Forward `y = W @ x + b`: `m=out, k=in, n=1`. Pre-load `y` with `b`, call with `beta=1`.
- Backward `dW = d_out @ xᵀ` (outer product): `m=out, k=1, n=in`. `beta=0` (or 1 if accumulating across micro-batches).
- Backward `dx = Wᵀ @ d_out`: `m=1, k=out, n=in`. `beta=0`. Skip if `tape.upstream.is_none()`.

The bias gradient is `db += d_out` — no GEMM needed.

**Acceptance.**
- `kernel_mm` and `kernel_naive` agree within `1e-5` on a 64×64 random GEMM.
- `cargo test --no-default-features --features train,naive-matmul` green.
- `benches/matmul_kernel.rs` shows `mm:naive ≥ 4×` for 256×256 on x86_64 host (matrixmultiply's headline speedup).

**Risks.**
- One `unsafe` block per call site (matrixmultiply requires it). Document the safety invariants (slice lengths ≥ `m*k`/`k*n`/`m*n`, valid strides) adjacent to each call.
- Pi Zero 2 W is Cortex-A53. matrixmultiply 0.3 auto-uses NEON on `aarch64`; verify by inspecting symbols (`nm | grep neon`) on a cross-compiled binary in Phase 9.

**Dependencies.** Phases 1, 2.

---

## Phase 5 — `.axn` model serialization (f32 baseline)

**Goal.** Define a stable, minimal binary format and implement `Mlp::save` / `Mlp::load`. Required for the fine-tune demo, the sensor-drift demo, and the INT8 PTQ workflow.

**Touched files.**
- [src/lib.rs](../src/lib.rs) — re-export `format::axn`.
- [src/nn/mlp.rs](../src/nn/mlp.rs) — `save`/`load`.
- `src/nn/linear.rs` — tensor accessors for save/load.

**New files.**
- `src/format/mod.rs`
- `src/format/axn.rs` — writer and reader.
- `src/format/axn_tests.rs`
- `docs/AXN_FORMAT.md` — wire-format spec for paper appendix.

**Wire format** (little-endian):

```
0   4   magic = b"AXN\0"
4   2   version = 0x0001
6   1   flags (bit 0: has_int8_quant; bit 1: per_channel_scales [reserved])
7   1   reserved = 0
8   4   num_tensors: u32
12  4   header_len: u32
16  ... tensor headers, each:
       2   name_len: u16
       N   name: utf8
       1   dtype: u8        (0=F32, 1=I8)
       1   rank: u8
       4r  dims: [u32; rank]
       4   scale: f32       (0.0 if not quantized)
       8   data_offset: u64
       8   data_len: u64
       4   crc32: u32
... raw tensor bytes (4-byte aligned) ...
final 4 bytes: crc32 of header region
```

Tensor names: `layer{N}.weight` (`[out, in]`), `layer{N}.bias` (`[out]`).

**API.**

```rust
pub enum Dtype { F32, I8 }
pub struct AxnWriter<W: Write + Seek> { ... }
pub struct AxnReader<R: Read + Seek> { ... }

impl Mlp {
    pub fn save(&self, path: &Path) -> io::Result<()>;
    pub fn load(path: &Path, activations: &[Activations]) -> io::Result<Self>;
}
```

`load` takes `activations` because v1 doesn't serialize activation choice (deliberate — kept simple, can extend in v2).

**Acceptance.**
- Round-trip random `Mlp` → `save` → `load` → bit-exact forward outputs.
- Corruption detection: flip a byte, reader returns `Err(Crc32Mismatch)`.
- 784→64→32→10 model serializes to ~217 KB f32 + ~200 B headers.

**Risks.**
- Endianness: spec is little-endian. `compile_error!` on `cfg(target_endian = "big")`.
- No serde / bincode dependency. CRC32 is a tiny inline IEEE polynomial implementation (~25 LoC, no `crc32fast` crate).

**Dependencies.** Phase 3.

---

## Phase 6 — Inference-only feature gating + pure-`&[f32]` forward

**Goal.** `cargo build --no-default-features --features inference` produces a binary with **zero `Rc<RefCell<Value>>`** code. `Mlp::infer(&[f32]) -> Vec<f32>` is the only forward path.

**Touched files.**
- [src/lib.rs](../src/lib.rs) — gate `pub mod engine`, `pub mod optim`, `pub mod loss` behind `cfg(feature = "train")`.
- `src/nn/linear.rs` — split `forward` (train, gated) from `infer_into_f32` (always on).
- [src/nn/mlp.rs](../src/nn/mlp.rs) — gate Node-based `forward` behind `train`; `Mlp::infer` and `Mlp::infer_into` always-on.
- [src/nn/activations.rs](../src/nn/activations.rs) — split `apply(Node)` (train) from `apply_f32_inplace(&mut [f32])` (always on).
- [src/nn/visualization.rs](../src/nn/visualization.rs) — gate behind `train`.
- **Move** `MatMulTape` from `src/engine/matmul.rs` to `src/nn/matmul_tape.rs` so it's available without the engine. Inside the struct, gate the gradient-only fields (`d_out`, `d_input`, `d_weights`, `d_bias`, `upstream`, `visit_count`, `backward_done`, `topo_walked`) behind `cfg(feature = "train")`.

**New API.**

```rust
impl Activations {
    pub fn apply_f32_inplace(&self, x: &mut [f32]);
}
impl Linear {
    pub fn infer_into_f32(&self, input: &[f32], output: &mut [f32]);
}
impl Mlp {
    pub fn infer(&self, input: &[f32]) -> Vec<f32>;
}
```

**Examples.** Add `#![cfg(feature = "train")]` to every example that uses `Node`. The new `examples/min_inference.rs` and `examples/rpi_inference.rs` build under `--features inference`.

**Acceptance.**
- `cargo build --no-default-features --features inference` succeeds.
- `cargo bloat --no-default-features --features inference --crates --release` shows zero bytes attributable to the `engine` module.
- Symbol audit: `nm target/release/examples/min_inference | grep -E "RefCell|Rc<.*Value"` returns 0 lines.
- `Mlp::infer` matches `Mlp::forward` (train mode) numerically within `1e-5` on the same inputs after `load`.

**Risks.**
- `MatMulTape` allocates differently in train vs inference (different field set). Document; verify size with `mem::size_of` test.
- `Sigmoid`/`Tanh`/`Swish` need `f32::exp` — works fine on glibc aarch64.

**Dependencies.** Phases 1, 2, 5.

---

## Phase 7 — INT8 weights-only post-training quantization

**Goal.** Quantize Linear weights to `Vec<i8>` + per-tensor `f32` scale (per-tensor symmetric, `scale = max(|W|) / 127`). Integrate into `.axn`. Add a dequant-fused matmul path. Keep biases as `f32`.

**Touched files.**
- `src/nn/linear.rs` — `WeightStorage` enum.
- [src/nn/mlp.rs](../src/nn/mlp.rs) — `quantize_to_i8`, `is_quantized`, `infer` dispatches on storage.
- `src/format/axn.rs` — already supports I8 dtype from Phase 5.

**New files.**
- `src/nn/quant.rs` — quantize/dequantize, dequant-fused matmul kernel.
- `src/nn/quant_tests.rs`.

**API.**

```rust
pub fn quantize_per_tensor_symmetric(w: &[f32]) -> (Vec<i8>, f32);
pub fn dequantize(qw: &[i8], scale: f32, out: &mut [f32]);

enum WeightStorage {
    F32(Vec<f32>),
    I8 { qweights: Vec<i8>, scale: f32 },
}

impl Mlp {
    #[cfg(feature = "quant-i8")]
    pub fn quantize_to_i8(&mut self);
    #[cfg(feature = "quant-i8")]
    pub fn save_quantized(&self, path: &Path) -> io::Result<()>;
    pub fn is_quantized(&self) -> bool;
}
```

**Dequant-fused inference kernel.** `matrixmultiply` is f32/f64 only — no int8 GEMM. Two strategies, choose at runtime via a size threshold:
- Small layers (`m * k <= 4096`): scalar loop `out[i] = bias[i] + scale * Σⱼ (qw[i,j] as f32) * x[j]`.
- Large layers (`m * k > 4096`): cast the i8 row to a scratch `Vec<f32>` (one allocation reused via the arena), then `sgemm`, then scale + bias-add.

**Optimizer policy.** **INT8 = inference only.** `Mlp::forward` panics with a clear message if any layer is `WeightStorage::I8`. Fine-tuning requires loading f32 `.axn`. After fine-tune, the user can re-quantize. No QAT/STE.

**Acceptance.**
- Quantize-dequantize round trip: max abs error ≤ `scale = max(|w|)/127`.
- MNIST: f32 model at 95% test acc → quantize → re-evaluate → ≤ 0.5 percentage point drop (paper-target number).
- Quantized 784→64→32→10 `.axn` ≤ 60 KB (vs ~217 KB f32).

**Risks.**
- Per-tensor symmetric is the simplest; per-channel symmetric usually loses less accuracy. Per-tensor first; per-channel listed as v0.4 stretch in PAPER.md if needed.
- The dequant-fused matmul is **slower per FLOP** than f32 sgemm. Paper benefit is binary size + memory footprint, **not** latency. State this explicitly.
- Saturation: clip `(w/scale).round()` to `[-127, 127]` (not `[-128, 127]`) to keep symmetry.

**Dependencies.** Phases 5, 6.

---

## Phase 8 — Static arena allocator + Criterion benchmark suite

**Goal.** Eliminate per-call heap allocation in inference; produce the latency table the paper hangs on.

**Touched files.**
- [Cargo.toml](../Cargo.toml) — `[[bench]]` entries, `[dev-dependencies] criterion = "0.5"`, `tempfile`.
- [src/nn/mlp.rs](../src/nn/mlp.rs) — arena type + `infer_into`.

**New files.**
- `src/nn/arena.rs`
- `benches/forward_train.rs`
- `benches/forward_infer_f32.rs`
- `benches/forward_infer_i8.rs`
- `benches/training_step.rs`
- `benches/finetune_step.rs`
- `benches/matmul_kernel.rs` — host-only, isolates the kernel speedup.

**Arena API.**

```rust
pub struct InferArena {
    buffer: Vec<f32>,
    slots: Vec<Range<usize>>,  // one slot per layer output (+ input slot at index 0)
}

impl InferArena {
    pub fn for_mlp(mlp: &Mlp) -> Self;
    pub fn buffer_bytes(&self) -> usize;
}

impl Mlp {
    pub fn infer_into(&self, input: &[f32], output: &mut [f32], arena: &mut InferArena);
}
```

`infer_into` does zero allocation per call. Default one slot per layer; ping-pong optimization deferred (RAM is plentiful for the 217 KB MNIST model).

**Bench cases** (all under criterion, both x86_64 host and aarch64 RPi Zero 2 W):
1. `forward_train::mnist_784_64_32_10` — `Mlp::forward(&[Node])`, includes graph build.
2. `forward_train::mnist_legacy_neuron` — same arch via `Layer<Neuron>`. The "scalar autograd reference" for the speedup ratio.
3. `forward_infer_f32::mnist_784_64_32_10` — `Mlp::infer_into` with arena. **Headline edge inference number.**
4. `forward_infer_i8::mnist_784_64_32_10` — quantized model.
5. `training_step::mnist_batch_32` — full forward + backward + SGD step. Demonstrates training is feasible.
6. `finetune_step::last_layer_only` — fine-tune only the last Linear of a pretrained model. **Phase 11 demo target.**
7. `matmul_kernel::sgemm_vs_naive_{64,256,784x64}` — kernel-only speedup.

`scripts/parse_criterion.py` consumes `target/criterion/**/estimates.json` and produces a wide CSV (one row per bench × target).

**Acceptance.**
- `forward_train::mnist_legacy_neuron` ≥ 10× slower than `forward_infer_f32::mnist_784_64_32_10` on x86_64 host.
- `finetune_step::last_layer_only` < 50 ms per step on host (paper-baseline number).
- Bench CSV regenerable from a single command.

**Dependencies.** Phases 1–7.

---

## Phase 9 — Cross-compile for `aarch64-unknown-linux-gnu`

**Goal.** Reproducible cross-compiled binaries for Pi Zero 2 W with 64-bit Pi OS, via both `cross` (Docker, CI-friendly) and `cargo-zigbuild` (Docker-free).

**Touched files.**
- `.cargo/config.toml`
- `.github/workflows/ci.yml` — add cross-compile matrix entry.

**New files.**
- `Cross.toml`
- `scripts/build_rpi.sh`, `scripts/build_rpi.ps1`
- `docs/RPI_DEPLOY.md` — flashing 64-bit Pi OS Lite, copying binaries via `scp`, running.

**`.cargo/config.toml`.**

```toml
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"
rustflags = ["-C", "target-cpu=cortex-a53", "-C", "target-feature=+neon"]
```

Cortex-A53 is the exact CPU in Pi Zero 2 W (Broadcom BCM2710A1, 4× A53 @ 1 GHz). The binary still runs on Pi 3/4/5 (newer cores) but is tuned for A53.

**Build commands.**

```sh
cross build --profile release-edge \
  --no-default-features --features inference \
  --target aarch64-unknown-linux-gnu --example rpi_inference
cross build --profile release-edge \
  --no-default-features --features train,matrixmultiply \
  --target aarch64-unknown-linux-gnu --example rpi_finetune_mnist
aarch64-linux-gnu-strip target/aarch64-unknown-linux-gnu/release-edge/examples/*
```

**Acceptance.**
- `cross build … --example rpi_inference` produces a runnable ELF; smoke-test with `qemu-aarch64-static` and on real hardware.
- Cross-compile CI job green.
- `nm` / `objdump -d` confirms NEON instructions are present in the matmul code (verifies matrixmultiply's auto-NEON kicked in).

**Risks.**
- glibc version skew between cross's Ubuntu 20.04 base (glibc 2.31) and Pi OS Bookworm (glibc 2.36). Older symbol set links forward-compatibly. `cargo-zigbuild` lets us pin: `cargo zigbuild … --target aarch64-unknown-linux-gnu.2.31`. Document both.
- 32-bit Pi OS will not run the binary. `RPI_DEPLOY.md` calls this out at the top.

**Dependencies.** Phases 6, 7.

---

## Phase 10 — Binary-size measurement automation

**Goal.** Reproducibly measure binary size across a fixed combo matrix, on both host and RPi cross-compile.

**Touched files.**
- [Cargo.toml](../Cargo.toml) — register `examples/min_inference.rs`.

**New files.**
- `examples/min_inference.rs` — minimal ~50-line load + infer demo, the smallest realistic binary.
- `scripts/measure_binary_size.sh`, `scripts/measure_binary_size.ps1`
- `scripts/sizes_to_md.py`
- `docs/BINARY_SIZE.md` (auto-generated)

**Combo matrix.**

| ID | Profile        | Features                 | Target  | Strip |
|----|----------------|--------------------------|---------|-------|
| A  | `release`      | default (train)          | host    | no    |
| B  | `release`      | inference                | host    | no    |
| C  | `release-edge` | inference                | host    | yes   |
| D  | `release-edge` | inference + quant-i8     | host    | yes   |
| E  | `release-edge` | inference                | aarch64 | yes   |
| F  | `release-edge` | inference + quant-i8     | aarch64 | yes   |

For each: build → measure (`wc -c` / `Get-Item .Length`) → append CSV row → render Markdown table.

**Acceptance.**
- `bash scripts/measure_binary_size.sh` (Linux) and `pwsh scripts/measure_binary_size.ps1` (Windows) both populate `binary_sizes.csv` and regenerate `docs/BINARY_SIZE.md`.
- The table reports actual numbers, not placeholders.

**Risks.**
- `cargo bloat` is informative but slow; runs as a non-blocking secondary.
- `strip = "symbols"` profile setting works on stable since 1.59 — simpler than a post-step.

**Dependencies.** Phase 9.

---

## Phase 11 — On-device demos: MNIST personalization + sensor-drift adaptation

**Goal.** Two reproducibility artifacts that anchor the paper:
1. **MNIST personalization fine-tune** on Pi Zero 2 W: load pretrained `.axn`, fine-tune the final Linear on a small user-personalization subset, save adapted `.axn`, report before/after accuracy and per-step wall-clock.
2. **Synthetic sensor-drift adaptation**: a regression model trained on an initial sensor distribution that progressively drifts; demonstrate that periodic on-device fine-tuning recovers accuracy.

**Touched files.**
- [Cargo.toml](../Cargo.toml) — register the new examples and gate them on `features = ["train"]`.

**New files.**
- `examples/rpi_inference.rs` — pure-inference companion (used by Phase 10's binary-size matrix).
- `examples/mnist_personalize_pretrain.rs` — host-side training of the personalization base model (`Mlp::new(&[784, 256, 128, 10], &[ReLU, ReLU, None])`); saves `mnist_pretrained.axn`. Kept separate from `examples/mnist_classifier.rs`, which stays at 784→64→32→10 as the Phase 3 regression baseline.
- `examples/rpi_finetune_mnist.rs` — MNIST personalization demo.
- `examples/rpi_sensor_drift.rs` — sensor-drift adaptation demo.
- `python-tests/generate_personalize_data.py` — apply a fixed per-user affine + photometric transform to a held-out MNIST subset to simulate one user's consistent handwriting drift. Outputs three CSVs at `python-tests/mnist/`: `mnist_personalize_train.csv` (200 augmented samples for fine-tune), `mnist_personalize_test.csv` (500 augmented samples for eval), `mnist_personalize_clean.csv` (the same 500 indices un-augmented, for the domain-shift baseline row in the paper). Also writes `personalize_preview.png` showing clean-vs-augmented sample pairs.
- `python-tests/generate_sensor_drift.py` — synthesize a drifting-sensor dataset (e.g., temperature sensor with monotonic offset over time, plus Gaussian noise) → `sensor_train.csv`, `sensor_drift_t1.csv`, `sensor_drift_t2.csv`, `sensor_drift_t3.csv`.
- `scripts/run_paper_artifacts.sh` — end-to-end driver for all paper measurements.

**MNIST personalization flow.**
1. On host: train `Mlp::new(&[784, 256, 128, 10], &[ReLU, ReLU, None])` on full MNIST via `examples/mnist_personalize_pretrain.rs`; target ≥ 97% on the un-augmented test set; save `mnist_pretrained.axn` (~937 KB f32).
2. Run `python-tests/generate_personalize_data.py` to produce the three personalization CSVs from the held-out MNIST test set.
3. On RPi: load `mnist_pretrained.axn`. Evaluate on `mnist_personalize_clean.csv` (500 un-augmented samples) — should match host accuracy within rounding.
4. Evaluate on `mnist_personalize_test.csv` (500 augmented samples) — accuracy drops because the user's distribution differs from training. This is the **before** number.
5. Load `mnist_personalize_train.csv` (200 augmented samples).
6. `Sgd::new(0.01, mlp.parameters_for_layers(2..3))` — only the final 128→10 Linear (~1290 params). Detach inputs to that layer so backward does not propagate `dx` into frozen layers (otherwise upstream gradients accumulate uselessly into layer 0/1 tapes).
7. Fine-tune 50 epochs × 200 samples, batch 4. Print per-step wall-clock and running loss.
8. Re-evaluate on `mnist_personalize_test.csv` — the **after** number.
9. Re-evaluate on `mnist_personalize_clean.csv` — confirm we have not catastrophically forgotten clean digits.
10. `mlp.save("mnist_finetuned.axn")`.
11. Print RSS via `sysinfo` before and after.

**Sensor-drift flow.**
1. Train a small `Mlp::new(&[1, 8, 8, 1], &[ReLU, ReLU, None])` on `sensor_train.csv` (host); save `sensor_initial.axn`.
2. On RPi Zero 2 W, evaluate MSE on `sensor_drift_t1.csv` → high error (drift kicked in).
3. Fine-tune the full small model (it's tiny, 50 params total) for 200 steps on a 100-sample buffer of recent drifted samples.
4. Re-evaluate on `sensor_drift_t1.csv` → MSE drops.
5. Repeat for `t2`, `t3` to show progressive adaptation.
6. Save `sensor_adapted_t3.axn`.
7. Report wall-clock per fine-tune cycle and final-vs-initial MSE.

**Acceptance.**
- Both demos run end-to-end on Pi Zero 2 W (manual + qemu-aarch64 smoke test in CI).
- MNIST personalization, 784→256→128→10 model: per-step wall-clock ≤ 3 s for batch=4 on Pi Zero 2 W. Augmented test accuracy after fine-tune ≥ pre-fine-tune augmented accuracy + 4 percentage points. Clean test accuracy stays within 0.5 percentage points of the pretrained baseline (no catastrophic forgetting).
- Sensor-drift: full fine-tune cycle (200 steps, 100 samples) completes in < 10 s on Pi Zero 2 W. MSE on drifted distribution drops by ≥ 30% post-adaptation.
- Both produce `.axn` files that round-trip through `Mlp::load` and infer correctly.

**Risks.**
- 512 MB RAM on Pi Zero 2 W is plentiful for these workloads — not a concern.
- The "user" is simulated by a fixed affine + photometric transform on held-out MNIST images, giving a real distribution shift to recover from. PAPER.md must list the augmentation parameters (rotation, shift, brightness, contrast) so reviewers can reproduce the user persona; in a real deployment this would be the user's actual handwriting. Sensor drift is fully synthetic but well-motivated by the IoT-sensor-drift literature cited in the paper (BrainyEdge etc.).
- Fine-tuning the last layer requires the layer-2 → layer-3 boundary to act as a leaf for backward, otherwise `dx` accumulates into layer-0/1 tapes that the optimizer never reads. Either add a `Node::detach()` primitive (Phase 11 gap to fill) or rebuild the head layer with input values snapshotted into fresh leaf Nodes each forward pass.
- Tape allocation during training holds an input snapshot per Linear; for the 784→256→128→10 model peak RSS contribution from tape buffers is ~700 KB. Confirm total RSS stays well under 50 MB on Pi.

**Dependencies.** Phases 1–9. Practically requires Phase 10 (cross-compile) before testing on real hardware.

---

## Phase K-paper — Paper-grade artifacts

**Goal.** Pull all measurements into a single document and rewrite README around the thesis. Off the critical path; can run in parallel with Phase 11 or after.

**Touched files.**
- [README.md](../README.md) — rewrite around the thesis and link to `docs/PAPER.md`.
- [docs/CHANGELOG.md](CHANGELOG.md) — `0.3.0` entry.

**New files.**
- `docs/PAPER.md` — thesis, claims, methodology, results tables, reproduction instructions.
- `docs/COMPARISON.md` — side-by-side with Burn, Candle, TFLite Micro, MicroFlow.
- `scripts/compare_burn/` — separate Cargo project building the same MLP in [Burn](https://burn.dev/), times forward + training step.
- `scripts/compare_candle/` — same for [Candle](https://github.com/huggingface/candle).
- `scripts/compare_tflite_micro/` — small C harness building TFLite Micro for an equivalent MLP (inference only); measures binary size + inference latency.
- `scripts/compare_microflow/` — Rust harness for [MicroFlow](https://github.com/matteocarnelos/microflow-rs) (inference only).

**Result tables (filled with measured numbers from Phases 8 + 10 + 11):**
1. **Forward latency, MNIST 784→64→32→10, single sample.** Cols: legacy scalar Value, fused MatMul (train), inference f32, inference i8, Burn, Candle, MicroFlow, TFLite Micro. Rows: host x86_64, RPi Zero 2 W aarch64.
2. **Single training step, batch=32.** Cols: rusty-axon SGD, rusty-axon MeProp, Burn, Candle. (TFLite Micro / MicroFlow inapplicable — categorical difference.)
3. **Binary size, stripped.** Combos A–F + Burn + Candle + MicroFlow + TFLite Micro.
4. **RSS during training and inference.** RPi Zero 2 W only.
5. **Fine-tune wall-clock per step on RPi Zero 2 W.** Last-layer-only (MNIST) + full-model (sensor drift).
6. **PTQ accuracy delta.** f32 vs i8 test accuracy on MNIST.
7. **Sensor-drift adaptation.** MSE vs time, pre- and post-adaptation.

**Acceptance.**
- Every table cell links to (a) the producing script and (b) the raw CSV.
- `bash scripts/run_paper_artifacts.sh` reproduces every table on a clean host + RPi.
- README leads with the thesis and a single command to run the smallest demo.

**Risks.**
- Burn and Candle are slow to build — cross-compile from host. Don't try to build them on the Pi.
- Burn's MNIST docs use a CNN; we hand-write a 784→64→32→10 MLP for fairness.
- On host x86_64, Burn/Candle may outperform rusty-axon (they have BLAS). The paper's positioning is **edge**, not host. Make this explicit in the methodology section: "rusty-axon is not designed to beat Candle on a workstation; it is designed to fit and fine-tune on a Pi Zero 2 W where Candle does not."
- TFLite Micro / MicroFlow are inference-only; comparison is inference latency + binary size only, with explicit categorical-difference framing on training.

**Dependencies.** Phases 8, 10, 11.

---

## Cross-cutting design decisions

### How `MatMul` stores captured tensors without bloating every Node

`Operation::MatMul { tape: Rc<MatMulTape>, output_index: usize }`. The `Rc` is one fat pointer (16 B on 64-bit) and `usize` is 8 B. All `out_dim` output Nodes share one allocation; the captured `Vec<f32>` weight matrix and input vector exist exactly once. `MatMulTape::weights` is `RefCell<Vec<f32>>` so the optimizer can write through after backward.

### How `Linear::parameters()` returns `Vec<Node>` while weights live in a flat `Vec<f32>`

The `Node` struct becomes `enum NodeStorage { Owned(Rc<RefCell<Value>>), Param(ParamView) }`. `Param` Nodes route reads and writes into a `MatMulTape`'s flat buffers via `(tape, kind, index)`. They report `Operation::None` so they're treated as leaves. The `Sgd`/`MeProp` interface is unchanged. Justification over `Rc<RefCell<f32>>` per parameter: a `Vec<RefCell<f32>>` cannot be safely reinterpret-cast to `&[f32]` for `sgemm` (UB).

### How the `inference` feature flag fully removes `Rc<RefCell<Value>>`

`mod engine`, `mod optim`, `mod loss`, and `mod nn::visualization` are gated behind `cfg(feature = "train")`. `MatMulTape` lives outside `engine` (in `nn/matmul_tape.rs`); its gradient-only fields are `cfg(train)`-gated. `Linear::infer_into_f32` and `Mlp::infer` are always-on. Verified by `nm | grep RefCell` returning zero matches.

### CrossEntropy / Softmax stay scalar after Linear's output

Linear returns `Vec<Node>` where each output Node is a `MatMul` shell wrapped by `Activations::apply` (which produces scalar `Operation::Exp/Sub/Div` chains). `CrossEntropy::forward(&[Node], &[Node])` walks those Nodes scalar-style — none of its code touches MatMul. Backward propagates scalar gradients through softmax into each MatMul output Node's `add_gradient`, and only when `visit_count == out_dim` does `MatMulTape::run_backward()` fire. The fused-matmul boundary is exactly between Linear's output Nodes and the rest of the scalar graph.

### `matrixmultiply::sgemm` API and call sites

```rust
pub unsafe fn sgemm(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: *const f32, rsa: isize, csa: isize,
    b: *const f32, rsb: isize, csb: isize,
    beta:  f32,
    c: *mut f32,   rsc: isize, csc: isize,
);
```

For row-major `W: [out, in]`, vector `x: [in]`, `y: [out]`:
- **Forward** `y = W @ x + b`: pre-load `y` with `b`; `sgemm(m=out, k=in, n=1, A=W rsa=in csa=1, B=x rsb=1 csb=1, beta=1, C=y rsc=1 csc=1)`.
- **Backward dW** `dW = d_out @ xᵀ`: `sgemm(m=out, k=1, n=in, A=d_out rsa=1 csa=1, B=x rsb=in csb=1, beta=accumulate?1:0, C=dW rsc=in csc=1)`.
- **Backward dx** `dx = Wᵀ @ d_out`: `sgemm(m=1, k=out, n=in, A=d_out rsa=out csa=1, B=W rsb=in csb=1, beta=0, C=dx rsc=in csc=1)`.
- **Bias gradient** `db += d_out`: scalar copy / accumulate, no GEMM.

These are the only `unsafe` blocks in the crate.

### INT8 PTQ × optimizer policy

**INT8 = inference only.** `Mlp::forward(&[Node])` panics with `"Cannot train a quantized model; load f32 weights for fine-tuning"` if any layer is `WeightStorage::I8`. Two clean stories:
1. Load f32, fine-tune, save f32 — the fine-tune demo.
2. Load f32, quantize → i8, save quantized — the size-and-memory-reduction story.

No QAT, no STE, no fake-quant. Out of scope for v0.3 / paper v1.

### Why no Conv2d

The user did not ask for it; MNIST hits 95–97% with the 784→64→32→10 MLP; Conv2d would require a multi-dim tensor type that contradicts "scalar Value-based autograd stays". The fused-MatMul technique generalizes to fused Conv2d (im2col → matmul) and is listed as v0.4 future work in PAPER.md.

---

## Cross-cutting risks

- **Numerical precision.** f32 engine end-to-end (Phase 0.5) means accumulation in MSE/CrossEntropy over big batches loses ~`1e-4` precision vs f64. MNIST batch-32 is fine; document.
- **`Rc` cycles.** `MatMulTape::upstream` holds `Vec<Node>` for the *previous* layer's output Nodes; those Nodes hold `Rc<MatMulTape>` for that previous layer's tape. No cycle (different tapes). Verify with `Rc::strong_count` checks in tests.
- **`build_topo_recursive` correctness.** Upstream Nodes must be **before** MatMul output Nodes in forward topo (so they appear **after** in reverse). Add an explicit ordering test on a chained 2-Linear network.
- **Windows builds.** All scripts have `.ps1` mirrors. CI matrix includes `windows-latest` for `train` and `inference`. Cross-compile to aarch64 runs on `ubuntu-latest` only.
- **Reproducibility.** Pin `matrixmultiply = "=0.3.9"` and `criterion = "=0.5.1"` exactly so timing tables are stable across rebuilds.

---

## Phase ordering and session estimate

| Session | Phase | Focus                                                  | Critical |
|--------:|-------|--------------------------------------------------------|:--------:|
| 1       | 0     | hygiene, features, profiles, CI                        | yes      |
| 2       | 0.5   | f64 → f32 engine migration                             | yes      |
| 3       | 1     | fused MatMul op + tape                                 | yes      |
| 4       | 2     | Linear layer + ParamView Node enum                     | yes      |
| 5       | 3     | Mlp shim, regression test against legacy Neuron path   | yes      |
| 6       | 4     | matrixmultiply integration + naive fallback            | yes      |
| 7       | 5     | `.axn` format                                          | yes      |
| 8       | 6     | inference feature gating                               | yes      |
| 9       | 7     | INT8 PTQ                                               | yes      |
| 10      | 8     | arena + criterion benches                              | yes      |
| 11      | 9     | aarch64 cross-compile                                  | yes      |
| 12      | 10    | binary-size automation                                 | yes      |
| 13      | 11    | RPi demos (MNIST personalize + sensor drift)           | yes      |
| (par.)  | K     | PAPER.md, COMPARISON.md, Burn/Candle/TFLM/MicroFlow    | no       |

If compressed: 0+0.5 → one session, 4+8 → one session, 9+10 → one session. Minimum ≈ 9 sessions.

---

## Critical files

These are the files most central to executing the plan; they are touched in nearly every phase.

- [Cargo.toml](../Cargo.toml)
- [src/engine/value.rs](../src/engine/value.rs)
- [src/engine/ops.rs](../src/engine/ops.rs)
- `src/engine/matmul.rs` (new in Phase 1; relocated to `src/nn/matmul_tape.rs` in Phase 6)
- `src/nn/linear.rs` (new in Phase 2)
- `src/nn/param_view.rs` (new in Phase 2)
- [src/nn/mlp.rs](../src/nn/mlp.rs)
- [src/nn/activations.rs](../src/nn/activations.rs)
- `src/format/axn.rs` (new in Phase 5)
- `src/nn/quant.rs` (new in Phase 7)
- `src/nn/arena.rs` (new in Phase 8)

Existing functions/utilities to reuse rather than rewrite:
- `Node::add_gradient`, `Node::zero_gradient` — already the right interface for `Param` views.
- `Activations::Sigmoid/ReLU/Tanh/Swish` — keep `apply(Node)` for train; add a sibling `apply_f32_inplace(&mut [f32])` for inference.
- `Optimizer::step` / `zero_state` trait in [src/optim/optimizer.rs](../src/optim/optimizer.rs) — unchanged; `Sgd` and `MeProp` work as-is once `parameters()` returns `ParamView`-backed Nodes.
- CSV loaders in [examples/mnist_classifier.rs](../examples/mnist_classifier.rs) and `examples/bench_*` — pattern reused for sensor-drift demo.
- `sysinfo` crate is already a dep — used directly for RSS reporting in the demos.

---

## End-to-end verification

A reviewer (or you in a future session) verifies the rework end-to-end with this sequence:

```bash
# 1. Build everything cleanly under each feature combo (host)
cargo check --no-default-features --features train
cargo check --no-default-features --features inference
cargo check --no-default-features --features inference,quant-i8
cargo build --profile release-edge --no-default-features --features inference

# 2. Run the test suite under each combo
cargo test --no-default-features --features train
cargo test --no-default-features --features inference
cargo test --no-default-features --features train,quant-i8
cargo test --no-default-features --features train,naive-matmul

# 3. Train + save the MNIST baseline on host
cargo run --release --example mnist_classifier   # produces mnist_pretrained.axn

# 4. Quantize and verify accuracy delta
cargo run --release --features quant-i8 --example quantize_mnist  # produces mnist_q8.axn

# 5. Run the criterion benches on host; populate the CSV
cargo bench
python scripts/parse_criterion.py > docs/BENCH_HOST.csv

# 6. Cross-compile both demos for RPi Zero 2 W (aarch64)
bash scripts/build_rpi.sh

# 7. SCP to a real Pi Zero 2 W (or qemu-aarch64-static)
scp target/aarch64-unknown-linux-gnu/release-edge/examples/{rpi_inference,rpi_finetune_mnist,rpi_sensor_drift} pi@rpi-zero:/home/pi/

# 8. On the Pi: run the demos, capture wall-clock and accuracy deltas
ssh pi@rpi-zero ./rpi_finetune_mnist mnist_pretrained.axn mnist_personalize.csv
ssh pi@rpi-zero ./rpi_sensor_drift   sensor_initial.axn sensor_drift_t1.csv

# 9. Regenerate the binary-size and paper tables
bash scripts/measure_binary_size.sh
bash scripts/run_paper_artifacts.sh
```

If every step in steps 1-8 succeeds and step 9 produces `docs/BINARY_SIZE.md` plus a `docs/PAPER.md` with all tables filled in, the rework is complete.
