# Changelog

All notable changes to the Rusty-Axon project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Unreleased

The 0.3 line reframes the project as a training-capable edge framework,
following the 13-phase plan in [`docs/PAPER_REWORK_PLAN.md`](PAPER_REWORK_PLAN.md).
Phases 0 and 0.5 are complete; remaining phases land incrementally.

### Phase 0 — Repo hygiene, feature flags, profiles, CI

- ✅ `default = ["train", "matrixmultiply"]`; new feature flags: `train`,
  `inference`, `matrixmultiply` (`=0.3.9` pinned), `naive-matmul`, `quant-i8`.
- ✅ `release-edge` profile (`lto = "fat"`, `codegen-units = 1`,
  `panic = "abort"`, `opt-level = "z"`, `strip = "symbols"`).
- ✅ `rust-toolchain.toml` (1.87.0 + rustfmt + clippy), `clippy.toml`,
  `rustfmt.toml`, `.cargo/config.toml` placeholder, `.github/workflows/ci.yml`
  (fmt + clippy advisory + test matrix).
- ✅ Every `[[bin]]`/`[[example]]` gated on `required-features = ["train"]`.

### Phase 0.5 — Engine `f64 → f32`

- ✅ `Value::value` and `Value::gradient` are now `f32`. `Node::new`,
  accessors, `pow`, `exp`, `log`, and the `Operation::{Pow, Log}` payloads
  all moved to `f32`.
- ✅ `From<f64> for Node` retained as a lossy convenience so legacy callers
  keep compiling; `From<f32>`, `From<i32>`, `From<i64>` all cast to `f32`.
- ✅ Scalar arithmetic macros now resolve only against `f32` (untyped float
  literals constrain to `f32` through the single available impl; adding a
  parallel `f64` impl would create literal ambiguity).
- ✅ `Sgd::learning_rate`, `MeProp::learning_rate`, and
  `CrossEntropy::label_smoothing` migrated to `f32`. Loss aggregations divide
  by `len() as f32`.
- ✅ Engine tests bumped from `1e-6` (f64) to `1e-5` (f32) tolerance; all 45
  unit tests green. XOR demo still converges to a perfect truth table.
- ✅ New `test_value_struct_is_f32_packed` regression locks the size win.

## [0.2.0] - 2024-11-29

### Added - Optimizers
- ✅ `Optimizer` trait for extensible optimizer design
- ✅ `Sgd` - Stochastic Gradient Descent with learning rate
- ✅ `MeProp` - Sparse backpropagation optimizer ([ICML 2017 paper](https://proceedings.mlr.press/v70/sun17c.html))
  - Updates only top-k% parameters by gradient magnitude
  - Configurable sparsity (top_k parameter)
  - Reduces computation while often improving generalization

### Added - Loss Functions
- ✅ `Loss` trait for extensible loss function design
- ✅ `MeanSquaredError` (MSE) - Standard regression loss
- ✅ `RootMeanSquaredError` (RMSE) - MSE with sqrt for interpretable units
- ✅ `CrossEntropy` - Classification loss with:
  - Softmax computation (numerically stable)
  - Label smoothing support
  - Multi-class support

### Added - Training Examples
- ✅ XOR problem example (`examples/xor_problem.rs`)
- ✅ Complete training loop demonstration
- ✅ Comparison between SGD and MeProp optimizers

### Changed
- Updated `lib.rs` to export new loss and optim modules
- Added `set_value()` method to `Node` for parameter updates

---

## [0.1.0] - 2024-11-26

### Added - Core Engine
- ✅ Automatic differentiation engine with forward and backward passes
- ✅ 8 operations: Add, Sub, Mul, Div, Pow, Exp, Neg, Log
- ✅ Gradient accumulation for multi-path graphs
- ✅ Topological sort for correct gradient propagation
- ✅ Smart pointer architecture using `Rc<RefCell<Value>>`
- ✅ Scalar operations (Node × f64, f64 × Node, etc.)
- ✅ 25+ comprehensive tests for all operations

### Added - Neural Networks
- ✅ `Neuron` implementation with weights, bias, and activation
- ✅ `Layer` (fully connected) supporting multiple neurons
- ✅ `MLP` (Multi-Layer Perceptron) supporting deep networks
- ✅ Random weight initialization using `rand` crate
- ✅ Parameter collection methods for optimization
- ✅ 15+ tests for neural network components

### Added - Activation Functions
- ✅ Sigmoid: σ(x) = 1/(1+e⁻ˣ)
- ✅ Tanh: (e²ˣ-1)/(e²ˣ+1)
- ✅ Swish: x·σ(x)
- ✅ None (Linear/Identity)
- ⚠️ ReLU family (placeholder - needs gradient handling)

### Added - Graph Visualization
- ✅ DOT file generation (Graphviz format)
- ✅ Automatic rendering to PNG, SVG, PDF, JPG
- ✅ Color-coded nodes by gradient magnitude
  - Red: High gradients (|grad| > 1.0)
  - Yellow: Medium gradients (|grad| > 0.1)
  - Blue: Low gradients (|grad| > 1e-10)
  - Gray: Zero gradients
- ✅ Operation visualization with colors
  - Orange: +, -, negation
  - Green: ×, ÷
  - Purple: ^, exp, log
- ✅ `to_dot()` - Generate DOT string
- ✅ `save_graph()` - Save to .dot file
- ✅ `render_png()`, `render_svg()`, `render_pdf()` - Auto-render
- ✅ `render_to()` - Custom format rendering
- ✅ `check_graphviz()` - System check for Graphviz installation

### Added - Documentation
- ✅ AGENTS.md - Complete architecture documentation
- ✅ README.md - User-friendly quick start guide
- ✅ VISUALIZATION.md - Graph visualization guide
- ✅ Examples in `examples/graph_visualization.rs`
- ✅ Inline documentation for all public APIs
- ✅ LICENSE (MIT)
- ✅ This CHANGELOG

### Added - Project Structure
- ✅ Modular architecture (engine, nn, optim modules)
- ✅ Comprehensive test suite (40+ tests)
- ✅ Example programs in `main.rs`
- ✅ Proper `.gitignore` for generated files

### Technical Details
- **Language**: Rust 2021 edition
- **Dependencies**: `rand = "0.9.2"` (only dependency)
- **Architecture**: Enum-based operations with pattern matching
- **Memory Model**: Reference counted nodes with interior mutability
- **Design Pattern**: Computation graph with topological sort

### Known Limitations
- ⚠️ ReLU family needs special gradient handling
- ⚠️ Scalar operations only (no tensors/matrices)
- ⚠️ No GPU support
- ⚠️ No model serialization
- ⚠️ No mini-batch support (but MeProp provides similar benefits)

## [Unreleased] - Planned Features

### To Be Implemented
- [ ] Adam optimizer
- [ ] Proper ReLU implementation
- [ ] More activation functions (LeakyReLU, ELU, GELU)
- [ ] Model serialization (save/load)
- [ ] Gradient clipping
- [ ] Learning rate scheduling
- [ ] More examples (classification, regression)

### Future Enhancements
- [ ] Tensor operations (move beyond scalars)
- [ ] Performance optimizations
- [ ] Graph optimization (eliminate redundant nodes)
- [ ] Custom layer types (Conv, RNN, etc.)
- [ ] Data loaders and preprocessing
- [ ] More sophisticated visualizations

---

## Version History

- **v0.2.0** (2024-11-29): Added optimizers (SGD, MeProp), loss functions (MSE, RMSE, CrossEntropy), training examples
- **v0.1.0** (2024-11-26): Initial release with core engine, neural networks, and visualization
- **v0.0.1** (2024-11-XX): Project skeleton

---

**Note**: This is an educational project. Breaking changes may occur between versions.

