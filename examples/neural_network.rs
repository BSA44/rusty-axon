/// Example: Neural Network Basics
///
/// Demonstrates creating and using neural networks.
/// Run with: cargo run --example neural_network
use rusty_axon::engine::Node;
use rusty_axon::nn::activations::Activations;
use rusty_axon::nn::mlp::Mlp;

fn main() {
    println!("=== Neural Network Examples ===\n");

    // Example 1: Simple network
    println!("1. Creating a 2-4-1 network");
    let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
    println!("   Architecture: {:?}", mlp.get_architecture());
    println!("   Parameters: {}", mlp.parameters().len());

    // Forward pass
    let inputs = vec![Node::from(1.0), Node::from(2.0)];
    let outputs = mlp.forward(&inputs);
    println!("   Input: [1.0, 2.0]");
    println!("   Output: {:.4}", outputs[0].get_value());

    // Backward pass
    let mut output = outputs[0].clone();
    output.backward();
    println!("   Backward pass completed!");

    // Example 2: Deep network
    println!("\n2. Creating a deep 3-8-8-4-1 network");
    let deep_mlp = Mlp::new(
        &[3, 8, 8, 4, 1],
        &[
            Activations::Tanh,
            Activations::Tanh,
            Activations::Tanh,
            Activations::Sigmoid,
        ],
    );
    println!("   Architecture: {:?}", deep_mlp.get_architecture());
    println!("   Parameters: {}", deep_mlp.parameters().len());

    // Example 3: Different activations
    println!("\n3. Available activation functions:");
    println!("   - Sigmoid: σ(x) = 1/(1+e^(-x))");
    println!("   - Tanh: tanh(x) = (e^(2x)-1)/(e^(2x)+1)");
    println!("   - Swish: x * σ(x)");
    println!("   - None: linear (identity)");

    // Example 4: Accessing gradients
    println!("\n4. Accessing parameter gradients:");
    let params = mlp.parameters();
    let mut count_nonzero = 0;
    for p in &params {
        if p.get_gradient().abs() > 1e-10 {
            count_nonzero += 1;
        }
    }
    println!(
        "   {} of {} parameters have non-zero gradients",
        count_nonzero,
        params.len()
    );

    println!("\n✨ Examples completed!");
}
