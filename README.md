# 🦀 Rusty-Axon

**A micrograd-inspired automatic differentiation engine and neural network library in pure Rust.**

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An educational project implementing automatic differentiation (autograd) and neural networks from scratch in Rust, inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd).

## ✨ Features

- ✅ **Scalar Autograd Engine** - Automatic differentiation with gradient accumulation
- ✅ **Neural Networks** - Neuron, Layer, and MLP (Multi-Layer Perceptron)
- ✅ **Activation Functions** - Sigmoid, Tanh, Swish
- ✅ **Graph Visualization** - Beautiful computation graph rendering (PNG, SVG, PDF)
- ✅ **Pure Rust** - Zero-cost abstractions, no Python dependencies
- ✅ **Comprehensive Tests** - 40+ tests covering all operations

## 🚀 Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rusty-axon = "0.1.0"
```

Or clone and run:

```bash
git clone https://github.com/yourusername/rusty-axon.git
cd rusty-axon
cargo run
```

### Basic Example

```rust
use rusty_axon::engine::Node;

fn main() {
    // Create scalar values
    let a = Node::from(2.0);
    let b = Node::from(-3.0);
    let c = Node::from(10.0);
    
    // Build computation graph
    let d = a.clone() * b.clone();
    let e = d + c.clone();
    let mut f = e.pow(2.0);
    
    // Backpropagation
    f.backward();
    
    // Access gradients
    println!("df/da = {}", a.get_gradient()); // -60.0
    println!("df/db = {}", b.get_gradient()); // 40.0
    println!("df/dc = {}", c.get_gradient()); // 8.0
}
```

### Neural Network Example

```rust
use rusty_axon::engine::Node;
use rusty_axon::nn::{Mlp, Activations};

fn main() {
    // Create a 2-4-4-1 neural network
    let mlp = Mlp::new(
        &[2, 4, 4, 1],
        &[Activations::Tanh, Activations::Tanh, Activations::Sigmoid]
    );
    
    // Forward pass
    let inputs = vec![Node::from(1.0), Node::from(2.0)];
    let outputs = mlp.forward(&inputs);
    
    // Backward pass
    let mut output = outputs[0].clone();
    output.backward();
    
    println!("Output: {}", output.get_value());
    println!("Parameters: {}", mlp.parameters().len());
}
```

### Visualization Example

```rust
use rusty_axon::engine::Node;

fn main() {
    let a = Node::from(2.0);
    let b = Node::from(3.0);
    let mut c = (a.clone() + b.clone()).pow(2.0);
    c.backward();
    
    // Generate visualization
    c.render_png("graph").unwrap();  // Creates graph.png
    c.render_svg("graph").unwrap();  // Creates graph.svg
}
```

![Example Graph](examples/neural_network.png)

## 📚 Documentation

- **[AGENTS.md](AGENTS.md)** - Complete architecture and implementation details
- **[VISUALIZATION.md](VISUALIZATION.md)** - Graph visualization guide
- **[Examples](examples/)** - Code examples and tutorials

## 🎯 What Can You Build?

This library is perfect for:
- 🧠 **Learning** automatic differentiation and neural networks
- 📊 **Visualizing** computation graphs and gradient flow
- 🔬 **Experimenting** with custom activation functions
- 🎓 **Teaching** deep learning fundamentals
- 🧪 **Prototyping** simple neural network architectures

## 📖 Core Concepts

### Autograd Engine

The engine automatically builds a computation graph and computes gradients:

```rust
let x = Node::from(2.0);
let y = x.pow(3.0);  // y = x³
y.backward();         // dy/dx = 3x² = 12.0
println!("{}", x.get_gradient()); // 12.0
```



### Neural Networks

Build deep networks with ease:

```rust
// Architecture: 3 inputs → 8 hidden → 8 hidden → 4 hidden → 1 output
let network = Mlp::new(
    &[3, 8, 8, 4, 1],
    &[
        Activations::Tanh,
        Activations::Tanh,
        Activations::Tanh,
        Activations::Sigmoid
    ]
);
```

## 📊 Visualization

Visualize your computation graphs with color-coded gradients:

```rust
// After backward pass, visualize the graph
output.render_svg("my_network").unwrap();
```

**Features:**
- 🎨 Color-coded nodes by gradient magnitude (red=high, blue=low)
- 📦 Shows values and gradients in each node
- 🔄 Displays operation types and connections
- 📈 Multiple formats: PNG, SVG, PDF, JPG

Requires [Graphviz](https://graphviz.org/download/) for automatic rendering.

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
- ✅ 25+ engine tests (operations, chain rule, gradient accumulation)
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
│   │   └── tests.rs     # NN tests
│   └── optim/           # Optimizers (TODO)
├── examples/            # Example code
└── AGENTS.md           # Architecture documentation
```

## 🎓 Learning Path

1. **Start with basics** - `cargo run` to see examples
2. **Read AGENTS.md** - Understand the architecture
3. **Run tests** - `cargo test` to see how it works
4. **Visualize** - Create graphs to see gradient flow
5. **Build networks** - Experiment with different architectures
6. **Implement optimizer** - Add SGD as next step

## 🚧 Roadmap

### Current Status
- ✅ Autograd engine with 8 operations
- ✅ Neural network building blocks
- ✅ Graph visualization
- ✅ Comprehensive testing

### Coming Soon
- ⏳ SGD optimizer
- ⏳ Loss functions (MSE, Cross-Entropy)
- ⏳ Training loop utilities
- ⏳ Real-world examples (XOR, classification)
- ⏳ Model serialization

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
- **Graphviz** - For visualization capabilities

**Built with ❤️ in Rust for learning and education.**

*Star ⭐ this repo if you find it helpful!*
