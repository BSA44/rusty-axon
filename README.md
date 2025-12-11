# 🦀 Rusty-Axon

**A micrograd-inspired automatic differentiation engine and neural network library in pure Rust.**

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An educational project implementing automatic differentiation (autograd) and neural networks from scratch in Rust, inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd).

## ✨ Features

- ✅ **Scalar Autograd Engine** - Automatic differentiation with gradient accumulation
- ✅ **Neural Networks** - Neuron, Layer, and MLP (Multi-Layer Perceptron)
- ✅ **Optimizers** - SGD and MeProp (sparse backpropagation from [ICML 2017](https://proceedings.mlr.press/v70/sun17c.html))
- ✅ **Loss Functions** - MSE, RMSE, CrossEntropy (with label smoothing)
- ✅ **Activation Functions** - Sigmoid, Tanh, ReLU, Swish
- ✅ **Dual Visualization** - Layer-oriented architecture + detailed computation graphs (PNG, SVG, PDF)
- ✅ **Pure Rust** - Zero-cost abstractions, no Python dependencies
- ✅ **Comprehensive Tests** - 45+ tests covering all operations

## 🚀 Quick Start

### Installation

Clone and run:

```bash
git clone https://github.com/BSA44/rusty-axon.git
cd rusty-axon
cargo run
```

### Quick Example

```rust
use rusty_axon::engine::Node;

let x = Node::from(2.0);
let mut y = x.clone().pow(3.0);  // y = x³
y.backward();
println!("dy/dx = {}", x.get_gradient()); // 12.0 (3x² at x=2)
```

## 📂 Examples

Run examples with `cargo run --example <name>`:

| Example | Description | Command |
|---------|-------------|---------|
| **basic_autograd** | Core autograd operations | `cargo run --example basic_autograd` |
| **neural_network** | Creating and using MLPs | `cargo run --example neural_network` |
| **xor_problem** | Complete training loop with XOR (Tanh) | `cargo run --example xor_problem` |
| **xor_relu** | XOR training with ReLU activation | `cargo run --example xor_relu` |
| **graph_visualization** | Computation graph rendering | `cargo run --example graph_visualization` |
| **network_visualization** | Layer-oriented network diagrams | `cargo run --example network_visualization` |
| **custom_colors** | Custom visualization themes | `cargo run --example custom_colors` |

### Example Output (XOR Training)

```
Network Architecture: [2, 4, 1]
Epoch    0 | Loss: 0.312
Epoch  500 | Loss: 0.001
Epoch  999 | Loss: 0.0003

Testing: [0,0]→0.01 ✓  [0,1]→0.98 ✓  [1,0]→0.98 ✓  [1,1]→0.02 ✓
```

![Network Architecture](network_architecture.png)

## 📚 Documentation

- **[AGENTS.md](AGENTS.md)** - Complete architecture and implementation details
- **[BENCHMARK.md](BENCHMARK.md)** - Performance comparison vs micrograd (NEW! 🚀)
- **[NETWORK_VISUALIZATION.md](NETWORK_VISUALIZATION.md)** - Layer-oriented network visualization guide
- **[VISUALIZATION.md](VISUALIZATION.md)** - Computation graph visualization guide
- **[Examples](examples/)** - Code examples and tutorials

## 🎯 What Can You Build?

This library is perfect for:
- 🧠 **Learning** automatic differentiation and neural networks
- 📊 **Visualizing** computation graphs and gradient flow
- 🔬 **Experimenting** with custom activation functions
- 🎓 **Teaching** deep learning fundamentals
- 🧪 **Prototyping** simple neural network architectures

## 📖 Core Concepts

| Component | Description |
|-----------|-------------|
| **Node** | Scalar value with automatic gradient tracking |
| **Mlp** | Multi-layer perceptron (feedforward network) |
| **Optimizer** | SGD or MeProp for weight updates |
| **Loss** | MSE, RMSE, or CrossEntropy |

**Supported Operations:** `+`, `-`, `*`, `/`, `pow`, `exp`, `log`, `neg`, `relu`

**Activations:** Sigmoid, Tanh, ReLU, Swish, Linear

## 📊 Visualization

Two visualization modes (requires [Graphviz](https://graphviz.org/download/)):

| Type | Method | Use Case |
|------|--------|----------|
| **Layer View** | `mlp.render_network_png("net")` | Architecture diagrams |
| **Computation Graph** | `output.render_png("graph")` | Debugging gradients |

See [examples/network_visualization.rs](examples/network_visualization.rs) for details.

## 🧪 Running Tests

```bash
# Run all tests
cargo test

# Run specific tests
cargo test engine::tests  # Test autograd engine
cargo test nn::tests      # Test neural networks

# With output
cargo test -- --nocapture
```

**Test Coverage:**
- ✅ 30+ engine tests (operations, chain rule, gradient accumulation, ReLU)
- ✅ 15+ neural network tests (forward/backward passes)
- ✅ Integration tests (deep networks, complex graphs)

## 🏗️ Project Structure

```
rusty-axon/
├── src/
│   ├── engine/          # Autograd engine
│   │   ├── value.rs     # Node and Value types
│   │   ├── ops.rs       # Operations enum
│   │   └── tests.rs     # Engine tests
│   ├── nn/              # Neural network components
│   │   ├── neuron.rs    # Single neuron
│   │   ├── layer.rs     # Fully connected layer
│   │   ├── mlp.rs       # Multi-layer perceptron
│   │   ├── activations.rs # Activation functions
│   │   ├── visualization.rs # Network visualization
│   │   └── tests.rs     # NN tests
│   ├── optim/           # Optimizers
│   │   ├── optimizer.rs # Optimizer trait
│   │   ├── sgd.rs       # Stochastic Gradient Descent
│   │   └── meprop.rs    # Sparse backpropagation (MeProp)
│   └── loss/            # Loss functions
│       ├── loss.rs      # Loss trait
│       ├── mse.rs       # Mean Squared Error
│       ├── rmse.rs      # Root Mean Squared Error
│       └── cross_entropy.rs # CrossEntropy with label smoothing
├── examples/            # Example code
│   ├── xor_problem.rs   # XOR training (Tanh activation)
│   ├── xor_relu.rs      # XOR training (ReLU activation)
│   └── ...
└── AGENTS.md           # Architecture documentation
```

## 🎓 Learning Path

1. `cargo run --example basic_autograd` - Understand autograd
2. `cargo run --example neural_network` - Build networks
3. `cargo run --example xor_problem` - Train a model
4. `cargo run --example graph_visualization` - Visualize gradients

## 🚧 Roadmap

### Current Status
- ✅ Autograd engine with 9 operations
- ✅ Neural network building blocks
- ✅ Graph visualization (layer + computation)
- ✅ SGD optimizer
- ✅ MeProp optimizer (sparse backpropagation)
- ✅ Loss functions (MSE, RMSE, CrossEntropy)
- ✅ XOR training example
- ✅ Comprehensive testing
- ✅ Added multithreading on branch `multithreading`

### Coming Soon
- ⏳ Adam optimizer
- ⏳ Learning rate scheduling
- ⏳ More activation functions (LeakyReLU, GELU)
- ⏳ Model serialization (save/load)
- ⏳ More examples (classification, regression)

## ⚠️ Limitations

This is an **educational project**. Not suitable for production:

- **Scalar only** - No tensor/matrix operations
- **CPU only** - No GPU acceleration
- **No batching** - One example at a time
- **Memory intensive** - Stores full computation graph
- **Not optimized** - Focus on clarity over performance


## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- **Andrej Karpathy** - For [micrograd](https://github.com/karpathy/micrograd) and the amazing [tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0)
- **Rust Community** - For excellent documentation and tools


**Built with ❤️ in Rust for learning and education.**

