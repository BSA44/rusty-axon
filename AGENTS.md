# Rusty-Axon: Micrograd Autograd Engine in Rust

This repo contains the code for the micrograd autograd engine implementation in Rust - an educational project for automatic differentiation and neural network building.

## Project Status

**Core Engine:** ✅ **Complete and Working**
- Forward pass with automatic graph construction
- Backward pass with gradient accumulation
- Supports: Add, Sub, Mul, Div, Pow, Exp, Neg, Log operations
- 40+ comprehensive tests covering all operations

**Neural Networks:** ✅ **Complete and Working**
- ✅ Neuron with weights, bias, and activation functions
- ✅ Layer (fully connected) with parameter collection
- ✅ MLP (Multi-Layer Perceptron) supporting deep networks
- ✅ Activation functions: Sigmoid, Tanh, Swish, None

**Graph Visualization:** ✅ **Complete and Working**
- ✅ DOT file generation (Graphviz format)
- ✅ Auto-render to PNG, SVG, PDF, JPG
- ✅ Color-coded gradients (red=high, blue=low, gray=zero)
- ✅ Interactive visualization of computation graphs

**Optimizers:** ⚠️ **Not Yet Implemented**

## Architecture Overview

### Design Philosophy

This implementation uses a **Rust-native approach** with enums and pattern matching, rather than mimicking Python's closure-based design. This provides:
- Zero-cost abstractions via compile-time pattern matching
- Better type safety and exhaustive checking
- Easier debugging and maintenance
- No dynamic dispatch overhead

### Core Components

#### 1. `Node` - Smart Pointer Wrapper
```rust
pub struct Node {
    value: Rc<RefCell<Value>>
}
```
- Cheap-to-clone handle using reference counting (`Rc`)
- Interior mutability for gradient updates (`RefCell`)
- Multiple nodes can point to the same underlying `Value`

#### 2. `Value` - The Actual Data
```rust
pub struct Value {
    value: f64,
    gradient: f64,
    operation: Operation,
}
```
- Stores the scalar value and accumulated gradient
- Contains operation enum that holds parent references

#### 3. `Operation` - Enhanced Enum Design
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
    None,  // Leaf nodes
}
```
**Key Design Decision:** Parents are stored **inside** the operation enum, not in a separate field.

**Why this works:**
- Operations are self-documenting (e.g., `minuend` vs `subtrahend`)
- Pattern matching makes backward pass explicit and type-safe
- No need for boxed closures or dynamic dispatch
- Compiler ensures all operations are handled

### Backpropagation Algorithm

#### Forward Pass
Operations build the computation graph automatically:
```rust
let a = Node::from(2.0);
let b = Node::from(3.0);
let c = a.clone() + b.clone();  // Stores a and b in Operation::Add
```

#### Backward Pass
1. **Topological Sort:** Build list of nodes from inputs to output using post-order DFS
2. **Initialize:** Set output gradient to 1.0 (`dL/dL = 1`)
3. **Reverse Iteration:** Process nodes in reverse topological order
4. **Pattern Match:** For each node, match its operation and propagate gradients:
   - `Add`: gradient flows equally to both parents
   - `Sub`: gradient flows positively to minuend, negatively to subtrahend
   - `Mul`: chain rule with other operand's value
   - `Div`: chain rule with division derivatives
   - `Pow`: power rule with chain rule
   - `Exp`: exponential derivative with chain rule
   - `Neg`: negates the gradient
   - `Log`: logarithm derivative with chain rule

#### Gradient Accumulation
**Critical:** Gradients are **accumulated** (`+=`), not set (`=`).

**Why:** A node used multiple times receives gradients through multiple paths:
```rust
let a = Node::from(2.0);
let b = a.clone() * a.clone();  // a appears twice!
// After backward: a.grad = 2*a = 4.0 (accumulated from both paths)
```

### Memory Model

```
Node (cheap handle)                           Node (another handle)
  │                                                                       │
  ├───────────────────────────┘
  ▼
Rc (reference counter = 2)
  │
  ▼
RefCell (interior mutability)
  │
  ▼
Value { value: 2.0, gradient: 4.0, operation: ... }
```

Multiple `Node` handles can share the same `Value`, enabling the DAG structure needed for autograd.

## Neural Network Architecture

### Neuron
```rust
pub struct Neuron {
    weights: Vec<Node>,
    bias: Node,
    activation: Activations,
}
```
- Stores learnable parameters (weights and bias)
- Applies activation function during forward pass
- Supports parameter collection for optimization

### Layer
```rust
pub struct Layer {
    neurons: Vec<Neuron>,
}
```
- Collection of neurons with shared activation function
- All neurons receive the same inputs
- Output is vector of neuron outputs

### MLP (Multi-Layer Perceptron)
```rust
pub struct Mlp {
    layers: Vec<Layer>,
}
```
- Sequential stack of layers
- Supports deep networks (tested with 4+ layers)
- Flexible architecture with per-layer activations

### Activation Functions
```rust
pub enum Activations {
    Sigmoid,    // σ(x) = 1 / (1 + e^(-x))
    Tanh,       // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    Swish,      // swish(x) = x * σ(x)
    None,       // Linear (identity)
    // ReLU family (placeholder - needs special handling)
}
```

## Graph Visualization

### Features
- Generate DOT files for Graphviz
- Auto-render to PNG, SVG, PDF, JPG
- Color-coded by gradient magnitude
- Shows values, gradients, and operations

### Usage
```rust
let a = Node::from(2.0);
let b = Node::from(3.0);
let mut c = a.clone() + b.clone();
c.backward();

// Save as DOT
c.save_graph("graph.dot")?;

// Render to image (requires Graphviz)
c.render_png("graph")?;
c.render_svg("graph")?;
c.render_pdf("graph")?;

// Visualize entire neural network
let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
let inputs = vec![Node::from(1.0), Node::from(2.0)];
let mut output = mlp.forward(&inputs)[0].clone();
output.backward();
output.render_svg("mlp_graph")?;
```

See `VISUALIZATION.md` for detailed documentation.

## File Structure

```
rusty-axon/
├── src/
│   ├── engine/
│   │   ├── mod.rs       - Module exports
│   │   ├── value.rs     - Node, Value, operators, backward pass, visualization
│   │   ├── ops.rs       - Operation enum definition
│   │   ├── graph.rs     - Computation graph utilities (placeholder)
│   │   └── tests.rs     - 25+ tests for autograd engine
│   ├── nn/
│   │   ├── mod.rs       - Module exports
│   │   ├── neuron.rs    - Single neuron implementation ✅
│   │   ├── layer.rs     - Fully connected layer ✅
│   │   ├── mlp.rs       - Multi-layer perceptron ✅
│   │   ├── activations.rs - Activation functions ✅
│   │   └── tests.rs     - 15+ tests for neural networks
│   ├── optim/
│   │   ├── mod.rs       - Module exports
│   │   └── sgd.rs       - SGD optimizer (TODO)
│   ├── lib.rs           - Library entry point
│   └── main.rs          - Demo with visualization
├── examples/
│   └── graph_visualization.rs - Comprehensive visualization examples
├── AGENTS.md            - This file (architecture documentation)
├── README.md            - Quick start guide
├── VISUALIZATION.md     - Graph visualization guide
└── Cargo.toml          - Dependencies (only rand for initialization)
```

## API Usage

### Basic Operations
```rust
use rusty_axon::engine::Node;

let a = Node::from(2.0);
let b = Node::from(-3.0);
let c = a.clone() + b.clone();
let d = c.clone() * c.clone();
let mut e = d.pow(2.0);

// Forward pass is done, now backward
e.backward();

// Access gradients
println!("a.grad: {}", a.get_gradient());
println!("b.grad: {}", b.get_gradient());
```

### Neural Network
```rust
use rusty_axon::nn::{Mlp, Activations};
use rusty_axon::engine::Node;

// Create a 2-4-4-1 network
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

// Access all parameters
let params = mlp.parameters();
println!("Total parameters: {}", params.len());
```

### Gradient Lifecycle
```rust
// 1. Zero gradients (before each training iteration)
for param in mlp.parameters() {
    param.zero_gradient();
}

// 2. Forward pass (builds graph automatically)
let output = mlp.forward(&inputs);

// 3. Compute loss (MSE example)
let target = Node::from(1.0);
let diff = output[0].clone() - target;
let mut loss = diff.pow(2.0);

// 4. Backward pass (accumulates gradients)
loss.backward();

// 5. Update parameters (manual SGD)
for param in mlp.parameters() {
    let new_val = param.get_value() - learning_rate * param.get_gradient();
    // Note: Currently requires creating new nodes for parameter updates
}
```

## Testing

### Test Coverage
- **Engine Tests (25 tests)**: All operations, chain rule, gradient accumulation
- **Neural Network Tests (15 tests)**: Neuron, Layer, MLP forward/backward passes
- **Integration Tests**: Deep networks, multiple forward passes, gradient flow

### Running Tests
```bash
# Run all tests
cargo test

# Run specific module
cargo test engine::tests
cargo test nn::tests

# Run with output
cargo test -- --nocapture
```

## Completed Features

### Phase 1: Operations ✅
- ✅ `pow(exponent: f64)` - Power operation
- ✅ `exp()` - Exponential
- ✅ `log(base: f64)` - Logarithm
- ✅ `neg` - Negation (unary minus)
- ✅ Scalar operations (Node * f64, f64 * Node, etc.)

### Phase 2: Neural Network Components ✅
- ✅ Implement `Neuron` with weights, bias, and activation
- ✅ Implement `Layer` as collection of neurons
- ✅ Implement `MLP` as stack of layers
- ✅ Add parameter collection methods
- ✅ Random weight initialization (using `rand` crate)

### Phase 3: Visualization ✅
- ✅ DOT file generation
- ✅ Automatic rendering to PNG/SVG/PDF
- ✅ Color-coded gradients
- ✅ Operation visualization
- ✅ Neural network graph visualization

## Next Steps

### Phase 4: Optimizers
- [ ] Implement SGD with learning rate
- [ ] Parameter update mechanism
- [ ] Add momentum to SGD
- [ ] Implement Adam optimizer
- [ ] Add learning rate scheduling

### Phase 5: Training Utilities
- [ ] Loss functions (MSE, CrossEntropy)
- [ ] Mini-batch handling
- [ ] Training loop abstraction
- [ ] Gradient clipping
- [ ] Early stopping
- [ ] Model checkpointing

### Phase 6: Activation Functions
- [ ] ReLU with proper gradient handling
- [ ] LeakyReLU
- [ ] ELU
- [ ] GELU
- [ ] Softmax (for classification)

### Phase 7: Examples
- [ ] XOR problem (classic)
- [ ] Binary classification
- [ ] Simple regression
- [ ] Multi-class classification
- [ ] Real dataset example

## Known Limitations

1. **No GPU support** - CPU only, scalar operations
2. **No automatic batching** - Process one example at a time
3. **No graph optimization** - Builds full graph every forward pass
4. **Scalar only** - No tensor/matrix operations
5. **No serialization** - Can't save/load models yet
6. **No optimizer** - Manual parameter updates required
7. **ReLU gradient issue** - Conditional operations break autograd

## Performance Considerations

- **Memory**: Each operation creates new nodes (O(n) for graph size)
- **Speed**: Pure Rust, no Python overhead, but not optimized
- **Scalability**: Suitable for small networks and educational purposes
- **Visualization**: Large graphs (>100 nodes) may be slow to render

## Dependencies

```toml
[dependencies]
rand = "0.9.2"  # For random weight initialization
```

No other dependencies! Pure Rust implementation.

## Contributing

This is an educational project. Key areas for contribution:
1. Implement optimizers (SGD, Adam)
2. Add proper ReLU/LeakyReLU with subgradients
3. Add serialization (save/load models)
4. Optimize performance (reduce allocations)
5. Add more examples and tutorials

## References

- Original micrograd (Python): https://github.com/karpathy/micrograd
- Andrej Karpathy's micrograd video tutorial: https://www.youtube.com/watch?v=VMj-3S1tku0
- "Automatic Differentiation" concepts and algorithms
- Graphviz visualization: https://graphviz.org/

## Acknowledgments

This project is inspired by Andrej Karpathy's micrograd and created as part of an AI course term project. The goal is to understand automatic differentiation and neural networks from first principles using Rust's type system and memory safety guarantees.
