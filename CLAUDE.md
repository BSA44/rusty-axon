# Rusty-Axon: Agent Cheat Sheet

> Pure-Rust, memory-safe ML framework that **trains and infers from one
> codebase** on edge devices (Raspberry Pi Zero 2 W class). v0.3.0 reframes
> the original micrograd-style scalar autograd around a fused `MatMul` op
> inside `Linear`, with optional `inference`-only and `quant-i8` builds.

The authoritative implementation plan lives in
[`docs/PAPER_REWORK_PLAN.md`](docs/PAPER_REWORK_PLAN.md) — read it before
starting any phase.

## Phase status

| Phase | Focus                                                      | Status |
|------:|------------------------------------------------------------|--------|
| 0     | Repo hygiene, feature flags, profiles, CI                  | ✅ |
| 0.5   | `f64 → f32` engine migration                               | ✅ |
| 1     | Fused `MatMul` op + `MatMulTape`                           | ✅ |
| 2     | `Linear` layer + `ParamView` Node enum                     | ⏳ next |
| 3     | `Mlp` shim over `Linear`; legacy regression test           | ⏳ |
| 4     | `matrixmultiply` integration + naive fallback              | ⏳ |
| 5     | `.axn` model serialization                                 | ⏳ |
| 6     | Inference-only feature gating + pure-`&[f32]` forward      | ⏳ |
| 7     | INT8 PTQ (weights-only, per-tensor symmetric)              | ⏳ |
| 8     | Static arena + criterion benchmark suite                   | ⏳ |
| 9     | aarch64 cross-compile (Pi Zero 2 W)                        | ⏳ |
| 10    | Binary-size automation                                     | ⏳ |
| 11    | RPi demos: MNIST personalize + sensor-drift adapt          | ⏳ |
| K     | `PAPER.md`, `COMPARISON.md`, Burn/Candle/TFLM/MicroFlow    | ⏳ |

> **Note:** the fused `MatMul` op + `MatMulTape` exist in the engine but
> nothing user-facing uses them yet. `Mlp` still composes `Layer` →
> `Neuron` (legacy scalar dot products). Phase 2 introduces `Linear` and
> Phase 3 swaps `Mlp` over.

## File structure

```
src/
├── engine/
│   ├── value.rs               # Node, Value, operators, backward(), to_dot()
│   ├── ops.rs                 # Operation enum (incl. MatMul variant)
│   ├── matmul.rs              # MatMulTape: fused matmul forward/backward       (Phase 1)
│   └── tests.rs               # autograd + matmul correctness tests
├── nn/
│   ├── neuron.rs              # legacy scalar single neuron
│   ├── layer.rs               # legacy scalar fully-connected layer
│   ├── mlp.rs                 # multi-layer perceptron (still on Layer/Neuron)
│   ├── activations.rs         # Sigmoid, Tanh, ReLU, Swish, None
│   ├── visualization.rs       # layer-oriented network diagrams
│   ├── linear.rs              # fused Linear layer                              (Phase 2)
│   ├── param_view.rs          # ParamView leaf for Node                         (Phase 2)
│   ├── arena.rs               # static inference arena                          (Phase 8)
│   ├── quant.rs               # INT8 PTQ                                        (Phase 7)
│   └── tests.rs
├── optim/                     # Optimizer trait, Sgd, MeProp
├── loss/                      # Loss trait, Mse, Rmse, CrossEntropy
├── format/axn.rs              # .axn model serialization                        (Phase 5)
├── lib.rs                     # public exports (gated on cfg(feature = "train"))
└── main.rs                    # XOR demo

examples/                      # all gated on required-features = ["train"]
├── xor_problem.rs, xor_relu.rs, basic_autograd.rs, neural_network.rs
├── graph_visualization.rs, network_visualization.rs, custom_colors.rs
├── mnist_classifier.rs        # 95%+ MNIST baseline
└── bench_*.rs                 # diabetes (cls) + housing (reg) × {sgd, meprop}

docs/
├── PAPER_REWORK_PLAN.md       # AUTHORITATIVE 13-phase implementation plan
└── (PAPER.md, BINARY_SIZE.md, AXN_FORMAT.md, RPI_DEPLOY.md — later phases)
```

## Cargo features and profiles

```
default        = ["train", "matrixmultiply"]
train          # engine, autograd, optim, loss, visualization, nn (Node-based)
inference      # pure-&[f32] forward path; engine module gated out (Phase 6)
matrixmultiply # link the matrixmultiply crate (auto-NEON on aarch64; Phase 4)
naive-matmul   # force the naive kernel for the speedup table (Phase 4)
quant-i8       # INT8 PTQ load/save + dequant-fused matmul (Phase 7)
```

Until Phase 6 ships, `--features inference` builds an empty lib (the gating
is in place; the public inference surface lands later). Every example sets
`required-features = ["train"]`.

Profiles: stock `release` plus `release-edge` (`lto = "fat"`,
`codegen-units = 1`, `panic = "abort"`, `opt-level = "z"`,
`strip = "symbols"`) used for the binary-size table and shipped artifacts.

## Core architecture

Engine is `f32` end-to-end (Phase 0.5). `Operation` carries the
`MatMul { tape, output_index }` variant (Phase 1) alongside the scalar
ops; `MatMulTape` is the side struct shared by every output `Node` of one
fused matmul, holding weights, bias, input snapshot, and gradient buffers
exactly once.

### Node

```rust
pub struct Node { value: Rc<RefCell<Value>> }
```

Cheap clone, interior mutability for gradient updates. Phase 2 turns
`Node` into an enum `Owned(Rc<RefCell<Value>>) | Param(ParamView)`, where
`Param` views route reads/writes into a flat `Vec<f32>` inside a
`MatMulTape` (so `sgemm` can consume the buffer directly).

### Operation

```rust
pub enum Operation {
    Add { left: Node, right: Node },
    Sub { minuend: Node, subtrahend: Node },
    Mul { left: Node, right: Node },
    Div { dividend: Node, divisor: Node },
    Pow { base: Node, exponent: f32 },
    Exp { exponent: Node },
    Neg { operand: Node },
    Log { base: f32, operand: Node },
    ReLU { input: Node },
    MatMul { tape: Rc<MatMulTape>, output_index: usize },  // Phase 1
    None,
}
```

### MatMul backward dispatch

Each output `Node` of one matmul carries `(Rc<MatMulTape>, output_index)`:

1. `Node::backward` accumulates the incoming grad into
   `tape.d_out[output_index]` and bumps `tape.visit_count`.
2. When `visit_count == out_dim`, `MatMulTape::run_backward` fires once:
   `dW += d_out ⊗ x`, `db += d_out`, and (if inputs are not leaves)
   `dx = Wᵀ d_out` is propagated into upstream Nodes via `add_gradient`.
3. `build_topo_recursive` walks `tape.upstream` once per matmul (guarded
   by `tape.topo_walked`); `Node::backward` resets that flag at the end.

`d_weights` and `d_bias` accumulate across backward passes until
`MatMulTape::reset_grads()` is called (Phase 2 wires this into
`Optimizer::zero_state`). `d_out`, `visit_count`, `backward_done`, and
`topo_walked` reset every `forward()`.

### Training pattern

```rust
let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
let mut optimizer = Sgd::new(lr, mlp.parameters());
for _ in 0..epochs {
    optimizer.zero_state();
    let output = mlp.forward(&input);
    let mut loss = /* compute */;
    loss.backward();
    optimizer.step();
}
```

## Hard constraints (paper thesis)

- **Scalar `Value`-based autograd stays.** The single optimization is one
  fused `MatMul` op inside `Linear` (one matmul forward, two matmuls
  backward). Activations, loss, and everything else keep flowing through
  the scalar `Value` graph.
- **Engine is `f32` end-to-end.** No `f64` mixed-precision boundary.
- **Edge target:** `aarch64-unknown-linux-gnu` only (64-bit Pi OS). No
  hand-written NEON; `matrixmultiply` auto-uses NEON on aarch64.
- **No `Conv2d`** (out of scope for v0.3 / paper v1).
- **INT8 = inference only.** No QAT, no STE. Fine-tuning requires loading
  f32 weights; re-quantize after.

## Dependencies

```toml
rand = "0.9.2"                                            # weight init
csv = "1.3"                                               # CSV loaders
sysinfo = "0.30"                                          # RSS reporting
matrixmultiply = { version = "=0.3.9", optional = true }  # pinned for repro
# criterion = "=0.5.1"                                    # dev-dep, Phase 8
```

## Build / test commands

```bash
cargo build                                                  # default (train + matrixmultiply)
cargo test                                                   # full suite
cargo test engine                                            # autograd + matmul only
cargo check --no-default-features --features inference       # empty lib until Phase 6
cargo build --profile release-edge

cargo run --example xor_problem
cargo run --release --example mnist_classifier
```

## Lint policy

- `cargo fmt --all -- --check` is enforced in CI; the v0.2 codebase has
  been reformatted to match `rustfmt.toml`.
- `cargo clippy` is **advisory** during the rework (no `-D warnings`). The
  legacy `Neuron`/`Layer`/`Mlp` modules carry cosmetic warnings
  (`needless_return`, `module_inception`, `redundant_field_names`, …)
  that will be swept up naturally as those files are rewritten in Phase
  3. Flip back to deny-warnings after Phase 3.
