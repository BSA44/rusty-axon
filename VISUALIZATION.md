# 📊 Graph Visualization Guide

Rusty-Axon includes powerful visualization features inspired by Python's micrograd, allowing you to see your computation graphs with gradients.

## Features

- ✅ Generate DOT files (Graphviz format)
- ✅ Auto-render to PNG, SVG, PDF, JPG
- ✅ Color-coded gradients (red = high, blue = low)
- ✅ Show values and gradients in nodes
- ✅ Display operation types
- ✅ Handle complex multi-path graphs

## Quick Start

```rust
use rusty_axon::engine::Node;

let a = Node::from(2.0);
let b = Node::from(3.0);
let mut c = a.clone() + b.clone();
c.backward();

// Save as DOT file
c.save_graph("graph.dot").unwrap();

// Render to PNG (requires Graphviz)
c.render_png("graph").unwrap();

// Render to SVG (vector graphics)
c.render_svg("graph").unwrap();
```

## Installation Requirements

### Installed Graphviz

## API Reference

### Basic Methods

#### `to_dot() -> String`
Generate DOT format string representation.

```rust
let dot_string = node.to_dot();
println!("{}", dot_string);
```

#### `save_graph(filename: &str) -> Result<()>`
Save computation graph as DOT file.

```rust
node.save_graph("output.dot")?;
```

#### `render_png(name: &str) -> Result<()>`
Render graph as PNG image.

```rust
node.render_png("output")?;  // Creates output.png
```

#### `render_svg(name: &str) -> Result<()>`
Render graph as SVG (scalable vector graphics).

```rust
node.render_svg("output")?;  // Creates output.svg
```

#### `render_pdf(name: &str) -> Result<()>`
Render graph as PDF document.

```rust
node.render_pdf("output")?;  // Creates output.pdf
```

#### `render_to(name: &str, format: &str) -> Result<()>`
Render graph in custom format.

```rust
node.render_to("output", "jpg")?;   // Creates output.jpg
node.render_to("output", "gif")?;   // Creates output.gif
```

Supported formats: `png`, `svg`, `pdf`, `jpg`, `jpeg`, `gif`

#### `check_graphviz() -> bool`
Check if Graphviz is installed.

```rust
if Node::check_graphviz() {
    println!("Graphviz is available!");
}
```

## Examples

### Example 1: Simple Expression

```rust
let a = Node::from(2.0);
let b = Node::from(-3.0);
let c = a.clone() + b.clone();
let mut d = c.pow(2.0);
d.backward();

d.render_svg("expression")?;
```

**Output:** Shows nodes for a, b, c, d with their values and gradients, connected by operation nodes (+, ^).

### Example 2: Neural Network Forward Pass

```rust
let x = Node::from(1.0);
let w = Node::from(0.5);
let b = Node::from(0.1);

let z = w * x + b;
let mut a = 1.0 / (1.0 + (-z).exp());  // Sigmoid
a.backward();

a.render_png("neuron")?;
```

**Output:** Visualizes the neuron computation: `sigmoid(w*x + b)`

### Example 3: Tanh Activation

```rust
let x = Node::from(0.5);
let two_x = x.clone() * 2.0;
let exp_2x = two_x.exp();
let mut tanh = (exp_2x.clone() - 1.0) / (exp_2x + 1.0);
tanh.backward();

tanh.render_svg("tanh")?;
```

**Output:** Shows the complete computation graph for tanh approximation.

## Understanding the Visualization

### Node Colors

Nodes are color-coded by gradient magnitude:

- 🔴 **Red (lightcoral)**: High gradient (|grad| > 1.0)
- 🟡 **Yellow (lightyellow)**: Medium gradient (|grad| > 0.1)
- 🔵 **Blue (lightblue)**: Low gradient (|grad| > 1e-10)
- ⚪ **Gray (lightgray)**: Zero gradient (unused in backprop)

### Node Shapes

- **Rectangle**: Value nodes (show `val=X` and `grad=Y`)
- **Circle**: Operation nodes (show operation symbol)

### Operation Colors

- 🟠 **Orange**: Addition, Subtraction, Negation
- 🟢 **Green**: Multiplication, Division
- 🟣 **Purple**: Power, Exponential, Logarithm

## Viewing Without Graphviz

If you don't have Graphviz installed:

1. **Save DOT file**: `node.save_graph("graph.dot")?`
2. **Copy contents** of `graph.dot`
3. **Paste at**: http://www.webgraphviz.com/
4. **Click** "Generate Graph!"

This renders your graph in the browser without installing anything.

## Advanced Usage

### Customizing Visualization

Currently, visualization uses default colors and styles. Future enhancements:

- [ ] Custom color schemes
- [ ] Node labels/names
- [ ] Hide zero-gradient nodes
- [ ] Cluster by layers
- [ ] Show tensor shapes (when extended beyond scalars)

### Troubleshooting

**Problem**: "Graphviz not found" error

**Solution**: Install Graphviz (see above) and ensure `dot` is in your PATH.

**Problem**: DOT file generated but no image

**Solution**: 
```bash
# Manually render
dot -Tpng graph.dot -o graph.png

# Check for errors
dot -v graph.dot
```

**Problem**: Graph too large/complex

**Solution**: Render as SVG instead of PNG for better zoom/pan capabilities.

## Performance Notes

- Graph traversal is O(n) where n = number of nodes
- DOT generation is fast even for large graphs
- Image rendering time depends on graph complexity
- SVG is faster than PNG for large graphs

## Examples in Repository

Run the visualization example:

```bash
cargo run --example graph_visualization
```

See also:
- `examples/graph_visualization.rs` - Comprehensive examples
- `src/main.rs` - Simple demo with multiple examples

## Integration with Neural Networks

Visualize entire neural network forward passes:

```rust
use rusty_axon::nn::{Mlp, Activations};

let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
let inputs = vec![Node::from(1.0), Node::from(2.0)];
let mut output = mlp.forward(&inputs)[0].clone();
output.backward();

// Visualize the entire network!
output.render_svg("mlp_forward")?;
```

This will show all neurons, weights, biases, and activations with their gradients!

