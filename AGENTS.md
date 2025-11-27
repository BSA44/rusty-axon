# Rusty-Axon: Micrograd Autograd Engine in Rust

This repo contains the code for the micrograd autograd engine implementation in Rust - an educational project for automatic differentiation and neural network building.


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

Rusty-Axon provides **two complementary visualization systems**:

### 1. Layer-Oriented Network Visualization (NEW!)

**Purpose:** Visualize neural network **architecture** with clear layer structure.

**Features:**
- Shows Input, Hidden, and Output layers as grouped units
- Color-coded layers (Blue=Input, Yellow=Hidden, Green=Output)
- Displays activation functions (Tanh, Sigmoid, etc.)
- Clean, presentation-ready diagrams
- Perfect for understanding network structure

**Implementation:**
- Module: `src/nn/visualization.rs`
- Generates DOT files with subgraphs for each layer
- Uses Graphviz clusters to group neurons by layer
- Renders to PNG, SVG, PDF formats

**Usage:**
```rust
// Create network
let mlp = Mlp::new(&[2, 4, 4, 1], &[Activations::Tanh, Activations::Tanh, Activations::Sigmoid]);

// Visualize architecture
mlp.render_network_png("network")?;     // Creates network.png
mlp.render_network_svg("network")?;     // Creates network.svg
mlp.render_network_pdf("network")?;     // Creates network.pdf

// Or use the main method
mlp.visualize_network("network", "png")?;
```

### 2. Computation Graph Visualization (Original)

**Purpose:** Visualize the **computation graph** showing every scalar operation.

**Features:**
- Shows individual scalar values and operations
- Color-coded by gradient magnitude (red=high, blue=low, gray=zero)
- Displays values, gradients, and operation types
- Perfect for debugging backpropagation

**Implementation:**
- Methods: `Node::to_dot()`, `Node::render_png()`, etc.
- Recursively builds DOT representation of entire computation DAG
- Each operation creates intermediate nodes
- Shows the micrograd-style detailed view

**Usage:**
```rust
// Build computation
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

// Or visualize neural network computation
let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
let inputs = vec![Node::from(1.0), Node::from(2.0)];
let mut output = mlp.forward(&inputs)[0].clone();
output.backward();
output.render_svg("mlp_computation_graph")?;
```


### Documentation

- See `NETWORK_VISUALIZATION.md` for layer-oriented visualization guide
- See `VISUALIZATION.md` for computation graph visualization guide

## File Structure

```
rusty-axon/
├── src/
│   ├── engine/
│   │   ├── mod.rs       - Module exports
│   │   ├── value.rs     - Node, Value, operators, backward pass, computation graph viz
│   │   ├── ops.rs       - Operation enum definition
│   │   ├── graph.rs     - Computation graph utilities (placeholder)
│   │   └── tests.rs     - 25+ tests for autograd engine
│   ├── nn/
│   │   ├── mod.rs       - Module exports
│   │   ├── neuron.rs    - Single neuron implementation ✅
│   │   ├── layer.rs     - Fully connected layer ✅
│   │   ├── mlp.rs       - Multi-layer perceptron ✅
│   │   ├── activations.rs - Activation functions ✅
│   │   ├── visualization.rs - Layer-oriented network visualization ✅
│   │   └── tests.rs     - 15+ tests for neural networks
│   ├── optim/
│   │   ├── mod.rs       - Module exports
│   │   └── sgd.rs       - SGD optimizer (TODO)
│   ├── lib.rs           - Library entry point
│   └── main.rs          - Demo with dual visualization
├── examples/
│   ├── graph_visualization.rs - Computation graph examples (original)
│   └── network_visualization.rs - Layer-oriented network examples (NEW)
├── AGENTS.md            - This file (architecture documentation)
├── README.md            - Quick start guide
├── NETWORK_VISUALIZATION.md - Layer-oriented visualization guide (NEW)
├── VISUALIZATION.md     - Computation graph visualization guide
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




## References

- Original micrograd (Python): https://github.com/karpathy/micrograd
- Andrej Karpathy's micrograd video tutorial: https://www.youtube.com/watch?v=VMj-3S1tku0
- "Automatic Differentiation" concepts and algorithms
- Graphviz visualization: https://graphviz.org/
