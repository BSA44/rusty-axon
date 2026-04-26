# Neural Network Architecture Visualization

Rusty-Axon provides **two types** of visualization:

1. **Layer-Oriented Architecture View** (NEW) - Shows network structure with clear layers
2. **Computation Graph View** (Original) - Shows detailed scalar operations

## Layer-Oriented Visualization

### Overview

The layer-oriented visualization displays your neural network as a high-level architecture diagram, similar to what you'd draw on a whiteboard. It clearly shows:

- **Input Layer** (blue)
- **Hidden Layers** (yellow/gold)
- **Output Layer** (green)
- **Activation functions** for each layer
- **All connections** between neurons

### Quick Start

```rust
use rusty_axon::nn::{Mlp, Activations};

// Create a neural network
let mlp = Mlp::new(
    &[2, 4, 4, 1],
    &[Activations::Tanh, Activations::Tanh, Activations::Sigmoid]
);

// Visualize the architecture
mlp.render_network_png("my_network").unwrap();  // Creates my_network.png
```

### API Methods

#### `Mlp::visualize_network(output_name, format)`
Main visualization method that renders the network architecture.

**Parameters:**
- `output_name: &str` - Base name for output files (without extension)
- `format: &str` - Output format: "png", "svg", "pdf", "jpg"

**Example:**
```rust
mlp.visualize_network("network", "png").unwrap();
mlp.visualize_network("network", "svg").unwrap();
mlp.visualize_network("network", "pdf").unwrap();
```

#### Convenience Methods

```rust
// PNG format
mlp.render_network_png("network").unwrap();

// SVG format (best for papers/documents)
mlp.render_network_svg("network").unwrap();

// PDF format
mlp.render_network_pdf("network").unwrap();
```

### Architecture Information

Get information about your network:

```rust
// Get layer sizes
let architecture = mlp.get_architecture();  // &[2, 4, 4, 1]

// Get number of layers (including input)
let num_layers = mlp.num_layers();  // 4

// Get total parameters
let params = mlp.parameters();
println!("Total parameters: {}", params.len());
```

## Computation Graph Visualization

The original micrograd-style visualization shows **every scalar operation** in detail, including:
- Individual additions, multiplications
- Activation function operations
- Values and gradients at each step

### When to Use Each Type

| Use Case | Visualization Type |
|----------|-------------------|
| Understanding network architecture | **Layer-oriented** |
| Presentations/Papers | **Layer-oriented** (SVG/PDF) |
| Debugging network structure | **Layer-oriented** |
| Understanding gradient flow | **Computation graph** |
| Debugging backpropagation | **Computation graph** |
| Learning autograd internals | **Computation graph** |

### Example: Both Visualizations

```rust
use rusty_axon::engine::Node;
use rusty_axon::nn::{Mlp, Activations};

// Create network
let mlp = Mlp::new(&[2, 3, 1], &[Activations::Tanh, Activations::Sigmoid]);

// 1. Visualize architecture (layer view)
mlp.render_network_png("architecture").unwrap();

// 2. Do a forward pass
let inputs = vec![Node::from(1.0), Node::from(2.0)];
let outputs = mlp.forward(&inputs);

// 3. Backward pass
let mut output = outputs[0].clone();
output.backward();

// 4. Visualize computation graph (scalar operations)
output.render_png("computation_graph").unwrap();
```

This creates two files:
- `architecture.png` - Clean layer diagram
- `computation_graph.png` - Detailed scalar operations

## Visualization Features

### Layer-Oriented View

**Colors:**
- 🔵 **Light Blue** - Input layer background
- 💛 **Light Yellow** - Hidden layer backgrounds
- 💚 **Light Green** - Output layer background
- 🔵 **Blue circles** - Input neurons
- 💛 **Gold circles** - Hidden neurons
- 💚 **Green circles** - Output neurons

**Labels:**
- Layer names: "Input Layer", "Hidden Layer 1", "Output Layer"
- Activation functions shown in parentheses: "(Tanh)", "(Sigmoid)"
- Neuron IDs: L0N0, L1N0, etc.

### Computation Graph View

**Colors:**
- 🔴 **Red** - High gradient magnitude (> 1.0)
- 💛 **Yellow** - Medium gradient (0.1 - 1.0)
- 🔵 **Blue** - Low gradient (> 1e-10)
- ⚫ **Gray** - Zero gradient

**Labels:**
- Value and gradient for each node
- Operation type (+, ×, ÷, ^, exp, log, -)

## Examples

### Example 1: Simple Network

```rust
let network = Mlp::new(
    &[2, 4, 1],
    &[Activations::Tanh, Activations::Sigmoid]
);
network.render_network_svg("simple").unwrap();
```

Output: 2-input → 4-hidden → 1-output network

### Example 2: Deep Network

```rust
let network = Mlp::new(
    &[3, 8, 8, 4, 1],
    &[
        Activations::Tanh,
        Activations::Tanh,
        Activations::Tanh,
        Activations::Sigmoid
    ]
);
network.render_network_png("deep").unwrap();
```

Output: Deep network with multiple hidden layers

### Example 3: Custom Activations

```rust
let network = Mlp::new(
    &[4, 6, 3],
    &[Activations::Swish, Activations::None]  // None = Linear
);
network.render_network_pdf("custom").unwrap();
```

Shows different activation functions per layer

## Requirements

### Graphviz

Both visualization types require [Graphviz](https://graphviz.org/download/) to be installed:

**Windows:**
```powershell
winget install graphviz
# or
choco install graphviz
```

**macOS:**
```bash
brew install graphviz
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install graphviz
```

### Fallback

If Graphviz is not installed, the library will:
1. Generate `.dot` files (text format)
2. Provide instructions for installation
3. Suggest viewing at [webgraphviz.com](http://www.webgraphviz.com/)

## File Formats

| Format | Best For | File Size |
|--------|----------|-----------|
| PNG | Quick preview, sharing | Medium |
| SVG | Papers, scaling, web | Small |
| PDF | Documents, printing | Medium |
| JPG | Photos (not recommended) | Small |

**Recommendation:** Use **SVG** for documents and presentations, **PNG** for quick viewing.

## Advanced Configuration

Currently, the visualization uses default settings. Future versions may support:

```rust
use rusty_axon::nn::visualization::NetworkVisualizationConfig;

let config = NetworkVisualizationConfig {
    show_weights: true,      // Show weight values on edges
    show_bias: true,         // Show bias values
    neuron_size: 0.8,        // Adjust neuron circle size
    layer_spacing: 3.0,      // Horizontal spacing
    neuron_spacing: 1.0,     // Vertical spacing within layer
};
```

*Note: Custom config is not yet exposed in the API.*

## Comparison: Old vs New

### Before (Detailed Computation Graph Only)

```rust
let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
let output = mlp.forward(&inputs)[0].clone();
output.backward();
output.render_png("graph").unwrap();
```

Result: Hundreds of nodes showing every scalar operation (hard to see structure)

### After (Layer-Oriented + Computation Graph)

```rust
// Architecture view (clean!)
mlp.render_network_png("architecture").unwrap();

// Still available: detailed view for debugging
output.render_png("detailed").unwrap();
```

Result: Two complementary views - one for structure, one for details

## Troubleshooting

### "Graphviz not found"

Install Graphviz (see Requirements above) and ensure `dot` is in your PATH.

### Large networks are slow to render

For networks with >100 neurons:
- Use SVG format (faster rendering)
- Layer view is much faster than computation graph view

### Edges overlap

This is normal for fully-connected networks. The visualization prioritizes showing all connections. For clarity:
- Use layer-oriented view for architecture
- Computation graph view is better for small subgraphs

## Running Examples

```bash
# Run the demo
cargo run

# Run the visualization example
cargo run --example network_visualization
```

## See Also

- `VISUALIZATION.md` - Original computation graph visualization docs
- `examples/network_visualization.rs` - Comprehensive examples
- `examples/graph_visualization.rs` - Original computation graph examples

