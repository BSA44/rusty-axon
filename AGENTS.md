# Rusty-Axon: Agent Cheat Sheet

> Pure-Rust, memory-safe ML framework that **trains and infers from one
> codebase** on edge devices (Raspberry Pi Zero 2 W class). v0.3.0 reframes
> the original micrograd-style scalar autograd around a fused `MatMul` op
> inside `Linear`, with optional `inference`-only and `quant-i8` builds.

The authoritative implementation plan lives in
[`docs/PAPER_REWORK_PLAN.md`](docs/PAPER_REWORK_PLAN.md) — read it before
starting any phase.

## Phase status

| Phase | Focus | Status |
|------:|-------|--------|
| 0     | Repo hygiene, feature flags, profiles, CI | ✅ done |
| 0.5   | `f64 → f32` engine migration | ⏳ next |
| 1     | Fused `MatMul` op + `MatMulTape` | ⏳ |
| 2     | `Linear` layer + `ParamView` Node enum | ⏳ |
| 3     | `Mlp` shim over `Linear`; legacy regression test | ⏳ |
| 4     | `matrixmultiply` integration + naive fallback | ⏳ |
| 5     | `.axn` model serialization | ⏳ |
| 6     | Inference-only feature gating + pure-`&[f32]` forward | ⏳ |
| 7     | INT8 PTQ (weights-only, per-tensor symmetric) | ⏳ |
| 8     | Static arena + criterion benchmark suite | ⏳ |
| 9     | aarch64 cross-compile (Pi Zero 2 W) | ⏳ |
| 10    | Binary-size automation | ⏳ |
| 11    | RPi demos: MNIST personalize + sensor-drift adapt | ⏳ |
| K     | `PAPER.md`, `COMPARISON.md`, Burn/Candle/TFLM/MicroFlow | ⏳ |

## Quick component overview

| Component | Location | Status |
|-----------|----------|--------|
| Scalar autograd engine | `src/engine/` | ✅ Complete (`f64`; migrates to `f32` in Phase 0.5) |
| Neural networks (`Neuron`/`Layer`/`Mlp`) | `src/nn/` | ✅ Complete (legacy scalar; `Linear` lands in Phase 2) |
| Optimizers | `src/optim/` | ✅ SGD, MeProp |
| Loss functions | `src/loss/` | ✅ MSE, RMSE, CrossEntropy |
| Visualization | `src/nn/visualization.rs` | ✅ Complete |
| Fused `MatMul` op | `src/engine/matmul.rs` | ⏳ Phase 1 |
| `Linear` layer | `src/nn/linear.rs` | ⏳ Phase 2 |
| `.axn` serialization | `src/format/axn.rs` | ⏳ Phase 5 |
| INT8 PTQ | `src/nn/quant.rs` | ⏳ Phase 7 |
| Inference arena | `src/nn/arena.rs` | ⏳ Phase 8 |

## Cargo features

```
default        = ["train", "matrixmultiply"]
train          # engine, autograd, optim, loss, visualization, nn (Node-based)
inference      # pure-&[f32] forward path; engine module gated out (Phase 6)
matrixmultiply # link the matrixmultiply crate (auto-NEON on aarch64)
naive-matmul   # force the naive kernel (paper speedup-vs-NEON table; Phase 4)
quant-i8       # INT8 PTQ load/save + dequant-fused matmul (Phase 7)
```

Until Phase 6 ships, `--features inference` builds an empty lib (the gating
is in place; the public inference surface lands later). Every example sets
`required-features = ["train"]`.

## Cargo profiles

- `release` — stock.
- `release-edge` — `lto = "fat"`, `codegen-units = 1`, `panic = "abort"`,
  `opt-level = "z"`, `strip = "symbols"`. Used for the binary-size table and
  every shipped artifact.

## Repo scaffolding (Phase 0)

```
.cargo/config.toml          # placeholder; Phase 9 fills aarch64 cross-compile block
.github/workflows/ci.yml    # fmt + clippy (advisory) + test matrix + release-edge build
rust-toolchain.toml         # pinned to 1.87.0 + rustfmt + clippy
clippy.toml                 # msrv 1.87.0; relaxed too-many-arguments + cognitive-complexity
rustfmt.toml                # max_width 100, edition 2021, Unix newlines
Cargo.toml                  # 0.3.0; features, optional matrixmultiply (=0.3.9), profiles
Cargo.lock                  # checked in (reproducibility)
```

## File structure

```
src/
├── engine/
│   ├── value.rs               # Node, Value, operators, backward(), visualization
│   ├── ops.rs                 # Operation enum (Add, Mul, Pow, Exp, ReLU, ...)
│   └── tests.rs               # 30+ autograd tests
├── nn/
│   ├── neuron.rs              # Single neuron (legacy scalar; baseline for Phase 8)
│   ├── layer.rs               # Fully connected layer (legacy scalar)
│   ├── mlp.rs                 # Multi-layer perceptron
│   ├── activations.rs         # Sigmoid, Tanh, ReLU, Swish, None
│   ├── visualization.rs       # Layer-oriented network diagrams
│   └── tests.rs               # 15+ NN tests
├── optim/
│   ├── optimizer.rs           # Optimizer trait
│   ├── sgd.rs                 # Stochastic Gradient Descent
│   └── meprop.rs              # Sparse backprop (top-k% gradients)
├── loss/
│   ├── loss.rs                # Loss trait
│   ├── mse.rs                 # Mean Squared Error
│   ├── rmse.rs                # Root Mean Squared Error
│   └── cross_entropy.rs       # CrossEntropy + label smoothing
├── lib.rs                     # Public exports (all gated on cfg(feature = "train"))
└── main.rs                    # XOR demo (gated via [[bin]] required-features)

examples/                      # all gated on required-features = ["train"]
├── basic_autograd.rs
├── neural_network.rs
├── xor_problem.rs             # Complete training loop (Tanh)
├── xor_relu.rs                # XOR with ReLU activation
├── graph_visualization.rs     # Computation graphs
├── network_visualization.rs   # Layer diagrams
├── custom_colors.rs           # Custom theme demo
├── mnist_classifier.rs        # 95%+ MNIST baseline
└── bench_*.rs                 # Diabetes (cls) + Housing (reg) × {sgd, meprop}

docs/
├── PAPER_REWORK_PLAN.md       # AUTHORITATIVE 13-phase implementation plan
└── (PAPER.md, BINARY_SIZE.md, AXN_FORMAT.md, RPI_DEPLOY.md — added in later phases)
```

## Core Architecture (current — pre-Phase 0.5)

### Node (Smart Pointer)
```rust
pub struct Node { value: Rc<RefCell<Value>> }
```
- Cheap clone (reference counted), interior mutability for gradient updates.
- Key methods: `get_value()`, `get_gradient()`, `set_value()`, `backward()`.
- Phase 2 turns `Node` into an enum: `Owned(Rc<RefCell<Value>>) | Param(ParamView)`,
  where `Param` views route reads/writes into a flat `Vec<f32>` inside a
  `MatMulTape` (so `sgemm` can consume them directly).

### Operations
```rust
pub enum Operation {
    Add { left: Node, right: Node },
    Sub { minuend: Node, subtrahend: Node },
    Mul { left: Node, right: Node },
    Div { dividend: Node, divisor: Node },
    Pow { base: Node, exponent: f64 },   // f32 after Phase 0.5
    Exp { exponent: Node },
    Neg { operand: Node },
    Log { base: f64, operand: Node },    // f32 after Phase 0.5
    ReLU { input: Node },
    None,                                 // Leaf nodes
    // Phase 1: MatMul { tape: Rc<MatMulTape>, output_index: usize }
}
```

### Neural Network
```rust
let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
let output = mlp.forward(&inputs);
output.backward();
let params = mlp.parameters();  // Vec<Node>
```

### Training Pattern
```rust
let mut optimizer = Sgd::new(learning_rate, mlp.parameters());
for epoch in 0..epochs {
    optimizer.zero_state();           // 1. Zero gradients
    let output = mlp.forward(&input); // 2. Forward
    let mut loss = /* compute */;     // 3. Loss
    loss.backward();                  // 4. Backward
    optimizer.step();                 // 5. Update
}
```

## Key Traits

```rust
// src/optim/optimizer.rs
pub trait Optimizer {
    fn step(&mut self);       // Update parameters
    fn zero_state(&mut self); // Zero gradients
}

// src/loss/loss.rs
pub trait Loss {
    fn forward(&self, predictions: &[Node], targets: &[Node]) -> Node;
}
```

## Hard constraints (paper thesis)

- **Scalar `Value`-based autograd stays.** The single optimization is one
  fused `MatMul` op inside `Linear` (one matmul forward, two matmuls
  backward). Activations, loss, and everything else keep flowing through the
  scalar `Value` graph.
- **Engine is `f32` end-to-end** after Phase 0.5 (no `f64` mixed-precision
  boundary). Single precision matches MatMul, INT8, and inference paths.
- **Edge target:** `aarch64-unknown-linux-gnu` only (64-bit Pi OS). No
  hand-written NEON; `matrixmultiply` auto-uses NEON on aarch64.
- **No `Conv2d`** (out of scope for v0.3 / paper v1).
- **INT8 = inference only.** No QAT, no STE. Fine-tuning requires loading
  f32 weights; re-quantize after.

## Dependencies

```toml
rand = "0.9.2"                                # weight init
csv = "1.3"                                   # MNIST/diabetes/housing loaders
sysinfo = "0.30"                              # RSS reporting in demos
matrixmultiply = { version = "=0.3.9", optional = true }  # pinned for repro
# criterion = "=0.5.1"                        # added in Phase 8 (dev-dep)
```

## Build / test commands

```bash
# Default (train + matrixmultiply)
cargo build
cargo test

# Feature combos
cargo check --no-default-features --features train
cargo check --no-default-features --features inference            # empty lib until Phase 6
cargo check --no-default-features --features inference,quant-i8   # ditto
cargo build --profile release-edge

# Targeted tests
cargo test engine          # autograd only
cargo test nn              # neural networks only
```

## Running examples

```bash
cargo run --example xor_problem        # XOR with Tanh + MeProp
cargo run --example xor_relu           # XOR with ReLU
cargo run --example basic_autograd
cargo run --example neural_network
cargo run --release --example mnist_classifier
```

## Lint policy

- `cargo fmt --all -- --check` is enforced in CI; the v0.2 codebase has been
  reformatted to match `rustfmt.toml`.
- `cargo clippy` is **advisory** during the rework (no `-D warnings`). The
  legacy `Neuron`/`Layer`/`Mlp`/`engine` modules carry cosmetic warnings
  (`needless_return`, `module_inception`, `redundant_field_names`, …) that
  will be swept up naturally as those files are rewritten in Phases 0.5 / 3.
  Flip back to deny-warnings after Phase 3.
