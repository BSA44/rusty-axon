# Rusty-Axon: Agent Cheat Sheet

> Micrograd-style autograd engine + neural networks in Rust. Educational project.

## Quick Overview

| Component | Location | Status |
|-----------|----------|--------|
| Autograd Engine | `src/engine/` | ✅ Complete |
| Neural Networks | `src/nn/` | ✅ Complete |
| Optimizers | `src/optim/` | ✅ SGD, MeProp |
| Loss Functions | `src/loss/` | ✅ MSE, RMSE, CrossEntropy |
| Visualization | `src/nn/visualization.rs` | ✅ Complete |

## File Structure

```
src/
├── engine/
│   ├── value.rs      # Node, Value, operators, backward(), visualization
│   ├── ops.rs        # Operation enum (Add, Mul, Pow, Exp, etc.)
│   └── tests.rs      # 30+ autograd tests
├── nn/
│   ├── neuron.rs     # Single neuron (weights, bias, activation)
│   ├── layer.rs      # Fully connected layer
│   ├── mlp.rs        # Multi-layer perceptron
│   ├── activations.rs # Sigmoid, Tanh, ReLU, Swish, None
│   ├── visualization.rs # Layer-oriented network diagrams
│   └── tests.rs      # 15+ NN tests
├── optim/
│   ├── optimizer.rs  # Optimizer trait
│   ├── sgd.rs        # Stochastic Gradient Descent
│   └── meprop.rs     # Sparse backprop (top-k% gradients)
├── loss/
│   ├── loss.rs       # Loss trait
│   ├── mse.rs        # Mean Squared Error
│   ├── rmse.rs       # Root Mean Squared Error
│   └── cross_entropy.rs # CrossEntropy + label smoothing
├── lib.rs            # Public exports
└── main.rs           # Demo

examples/
├── basic_autograd.rs       # Core autograd demo
├── neural_network.rs       # MLP usage
├── xor_problem.rs          # Complete training loop (Tanh)
├── xor_relu.rs             # XOR with ReLU activation
├── graph_visualization.rs  # Computation graphs
├── network_visualization.rs # Layer diagrams
└── custom_colors.rs        # Custom themes
```

## Core Architecture

### Node (Smart Pointer)
```rust
pub struct Node { value: Rc<RefCell<Value>> }
```
- Cheap clone (reference counted)
- Interior mutability for gradient updates
- Key methods: `get_value()`, `get_gradient()`, `set_value()`, `backward()`

### Operations
```rust
pub enum Operation {
    Add { left: Node, right: Node },
    Sub { minuend: Node, subtrahend: Node },
    Mul { left: Node, right: Node },
    Div { dividend: Node, divisor: Node },
    Pow { base: Node, exponent: f64 },
    Exp { exponent: Node },
    Neg { operand: Node },
    Log { base: f64, operand: Node },
    ReLU { input: Node },
    None,  // Leaf nodes
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

## Implemented vs TODO

### ✅ Done
- All arithmetic ops (+, -, *, /, pow, exp, log, neg, relu)
- Neuron, Layer, MLP
- Activations: Sigmoid, Tanh, ReLU, Swish
- SGD optimizer
- MeProp optimizer (sparse backprop)
- MSE, RMSE, CrossEntropy loss
- Graph visualization (DOT → PNG/SVG/PDF)
- XOR training examples (Tanh & ReLU)

### ⏳ TODO
- Model save/load
- Learning rate scheduling
- Multi-threading

## Dependencies

Only `rand = "0.9.2"` for weight initialization.

## Testing

```bash
cargo test              # All tests
cargo test engine       # Engine only
cargo test nn           # Neural networks only
```

## Running Examples

```bash
cargo run --example xor_problem    # XOR with Tanh
cargo run --example xor_relu       # XOR with ReLU
cargo run --example basic_autograd
cargo run --example neural_network
```
