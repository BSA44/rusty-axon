//! Example demonstrating neural network architecture visualization.
//!
//! This example shows how to visualize neural network architectures
//! with a layer-oriented view (not the detailed computation graph).

use rusty_axon::engine::Node;
use rusty_axon::Activations;
use rusty_axon::Mlp;

fn main() {
    println!("=== Neural Network Architecture Visualization Examples ===\n");

    // Example 1: Simple network (XOR-like architecture)
    println!("Example 1: Simple XOR Network (2-4-1)");
    let xor_network = Mlp::new(
        &[10, 20, 10, 10, 4],
        &[
            Activations::Tanh,
            Activations::Tanh,
            Activations::Tanh,
            Activations::Tanh,
            Activations::Sigmoid,
        ],
    );

    println!("  Architecture: {:?}", xor_network.get_architecture());
    println!("  Total parameters: {}", xor_network.parameters().len());
    println!("  Visualizing...");
    xor_network.render_network_png("xor_network").unwrap();
    println!("  [+] Saved to xor_network.png\n");
    /*
    // Example 2: Deep network
    println!("Example 2: Deep Network (3-8-8-4-1)");
    let deep_network = Mlp::new(
        &[3, 8, 8, 4, 1],
        &[Activations::Tanh, Activations::Tanh, Activations::Tanh, Activations::Sigmoid]
    );

    println!("  Architecture: {:?}", deep_network.get_architecture());
    println!("  Total parameters: {}", deep_network.parameters().len());
    println!("  Visualizing...");
    deep_network.render_network_svg("deep_network").unwrap();
    println!("  [+] Saved to deep_network.svg\n");

    // Example 3: Wide network
    println!("Example 3: Wide Network (4-16-8-2)");
    let wide_network = Mlp::new(
        &[4, 16, 8, 2],
        &[Activations::Swish, Activations::Tanh, Activations::Sigmoid]
    );

    println!("  Architecture: {:?}", wide_network.get_architecture());
    println!("  Total parameters: {}", wide_network.parameters().len());
    println!("  Visualizing...");
    wide_network.render_network_png("wide_network").unwrap();
    println!("  [+] Saved to wide_network.png\n");

    // Example 4: Binary classification network
    println!("Example 4: Binary Classification (5-8-4-1)");
    let classifier = Mlp::new(
        &[5, 8, 4, 1],
        &[Activations::Tanh, Activations::Tanh, Activations::Sigmoid]
    );

    println!("  Architecture: {:?}", classifier.get_architecture());
    println!("  Total parameters: {}", classifier.parameters().len());
    println!("  Visualizing...");
    classifier.render_network_pdf("classifier").unwrap();
    println!("  [+] Saved to classifier.pdf\n");

    // Example 5: Forward pass visualization
    println!("Example 5: Forward Pass + Computation Graph");
    let network = Mlp::new(&[2, 3, 1], &[Activations::Tanh, Activations::Sigmoid]);

    // First, visualize the architecture
    println!("  Visualizing network architecture...");
    network.render_network_png("architecture").unwrap();

    // Then do a forward pass and visualize the computation graph
    let inputs = vec![Node::from(1.5), Node::from(-0.5)];
    let outputs = network.forward(&inputs);
    let mut output = outputs[0].clone();

    println!("  Forward pass: {} -> {}", inputs[0].get_value(), output.get_value());

    // Backward pass
    output.backward();
    println!("  Backward pass completed");

    // Visualize the detailed computation graph (micrograd-style)
    println!("  Visualizing computation graph...");
    output.render_png("detailed_computation_graph").unwrap();
    println!("  [+] Saved architecture.png (layer view)");
    println!("  [+] Saved detailed_computation_graph.png (scalar operations)\n");

    println!("=== Summary ===");
    println!("Generated visualizations:");
    println!("  • xor_network.png - Simple 2-4-1 network");
    println!("  • deep_network.svg - Deep 3-8-8-4-1 network");
    println!("  • wide_network.png - Wide 4-16-8-2 network");
    println!("  • classifier.pdf - Binary classification network");
    println!("  • architecture.png - Network architecture (layer view)");
    println!("  • detailed_computation_graph.png - Computation graph (scalar view)");
    println!("\nNote: Layer-oriented views show network structure.");
    println!("      Computation graphs show individual scalar operations.");
     */
}
