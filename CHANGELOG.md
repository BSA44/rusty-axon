# Changelog

All notable changes to the Rusty-Axon project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- ⚠️ No optimizer implementation yet (manual parameter updates)
- ⚠️ No loss functions
- ⚠️ ReLU family needs special gradient handling
- ⚠️ Scalar operations only (no tensors/matrices)
- ⚠️ No GPU support
- ⚠️ No model serialization
- ⚠️ No mini-batch support

## [Unreleased] - Planned Features

### To Be Implemented
- [ ] SGD optimizer with learning rate
- [ ] Loss functions (MSE, Cross-Entropy)
- [ ] Training loop utilities
- [ ] Proper ReLU implementation
- [ ] More activation functions (LeakyReLU, ELU, GELU)
- [ ] Model serialization (save/load)
- [ ] Mini-batch support
- [ ] Gradient clipping
- [ ] Learning rate scheduling
- [ ] Real-world examples (XOR, classification)

### Future Enhancements
- [ ] Tensor operations (move beyond scalars)
- [ ] Performance optimizations
- [ ] Graph optimization (eliminate redundant nodes)
- [ ] Custom layer types (Conv, RNN, etc.)
- [ ] Data loaders and preprocessing
- [ ] More sophisticated visualizations

---

## Version History

- **v0.1.0** (2024-11-26): Initial release with core engine, neural networks, and visualization
- **v0.0.1** (2024-11-XX): Project skeleton

---

**Note**: This is an educational project. Breaking changes may occur between versions.

