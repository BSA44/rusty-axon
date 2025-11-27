use rusty_axon::engine::{ComputationGraph, Node};
use rusty_axon::nn::mlp::Mlp;
use rusty_axon::nn::activations::Activations;

fn main() {
    println!("=== Rusty-Axon: Autograd Engine Demo ===\n");

    // Sanity check that core structs are visible to the binary target.
    let _ = std::mem::size_of::<Node>();
    let _ = std::mem::size_of::<ComputationGraph>();
    /*
    // Example 1: Simple expression
    println!("Example 1: Simple expression (a + b)^2");
    let a = Node::from(2.0);
    let b = Node::from(-3.0);
    let c = a.clone() + b.clone();
    let mut d = c.pow(2.0);
    
    println!("Before backward:");
    println!("  a: {}", a);
    println!("  b: {}", b);
    println!("  c: {}", c);
    println!("  d: {}", d);
    
    d.backward();
    
    println!("\nAfter backward:");
    println!("  a: {}", a);
    println!("  b: {}", b);
    println!("  c: {}", c);
    println!("  d: {}", d);
    
    // Save and render visualization
    println!("\n📊 Saving and rendering computation graph...");
    d.save_graph("example1_graph.dot").unwrap();
    d.render_png("example1_graph").unwrap();
    d.render_svg("example1_svg").unwrap();
    
    // Example 2: Tanh approximation
    println!("\n\nExample 2: Tanh approximation");
    let x = Node::from(0.5);
    let two_x = x.clone() * 2.0;
    let exp_2x = two_x.exp();
    let numerator = exp_2x.clone() - 1.0;
    let denominator = exp_2x + 1.0;
    let mut tanh_approx = numerator / denominator;
    
    println!("Before backward:");
    println!("  x: {}", x);
    println!("  tanh(x): {}", tanh_approx);
    
    tanh_approx.backward();
    
    println!("\nAfter backward:");
    println!("  x: {}", x);
    println!("  tanh(x): {}", tanh_approx);
    
    println!("\n📊 Saving and rendering computation graph...");
    tanh_approx.save_graph("example2_tanh.dot").unwrap();
    tanh_approx.render_svg("example2_tanh").unwrap();
    
    // Example 3: Neural network-like computation
    println!("\n\nExample 3: Neural network-like computation");
    let x1 = Node::from(2.0);
    let x2 = Node::from(3.0);
    let w1 = Node::from(0.5);
    let w2 = Node::from(-0.3);
    let bias = Node::from(1.0);
    
    let term1 = w1.clone() * x1.clone();
    let term2 = w2.clone() * x2.clone();
    let sum = term1 + term2 + bias.clone();
    let mut output = sum.pow(2.0);
    
    println!("Before backward:");
    println!("  x1: {}", x1);
    println!("  x2: {}", x2);
    println!("  w1: {}", w1);
    println!("  w2: {}", w2);
    println!("  output: {}", output);
    
    output.backward();
    
    println!("\nAfter backward:");
    println!("  x1: {}", x1);
    println!("  x2: {}", x2);
    println!("  w1: {}", w1);
    println!("  w2: {}", w2);
    println!("  output: {}", output);
    
    println!("\n📊 Saving and rendering computation graph...");
    output.save_graph("example3_neuron.dot").unwrap();
    output.render_png("example3_neuron").unwrap();
    
    // Check if graphviz is installed
    println!("\n🔍 Checking Graphviz installation...");
    if Node::check_graphviz() {
        println!("[+] Graphviz is installed!");
        println!("\n✨ Done! Generated files:");
        println!("  • example1_graph.png (and .svg)");
        println!("  • example2_tanh.svg");
        println!("  • example3_neuron.png");
    } else {
        println!("[-] Graphviz not found.");
        println!("\n✨ Done! Generated .dot files:");
        println!("  • example1_graph.dot");
        println!("  • example2_tanh.dot");
        println!("  • example3_neuron.dot");
        println!("\n   View them at: http://www.webgraphviz.com/");
    }
 */
println!("=== Example 1: Simple Neural Network (2-4-1) ===");
let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
println!("Architecture: {:?}", mlp.get_architecture());
println!("Total parameters: {}", mlp.parameters().len());

// Visualize the network architecture (layer-oriented)
println!("\n📊 Generating layer-oriented network visualization...");
mlp.render_network_png("network_architecture").unwrap();
mlp.render_network_svg("network_architecture_svg").unwrap();

// Forward pass
let inputs = vec![Node::from(1.0), Node::from(2.0)];
let outputs = mlp.forward(&inputs);
println!("\nForward pass:");
println!("  Input: [{}, {}]", inputs[0].get_value(), inputs[1].get_value());
println!("  Output: {}", outputs[0].get_value());

// Backward pass
let mut output = outputs[0].clone();
output.backward();
println!("\nBackward pass completed!");
println!("  Output gradient: {}", output.get_gradient());

// You can still visualize the detailed computation graph if needed
println!("\n📊 Generating detailed computation graph...");
output.render_png("computation_graph").unwrap();

println!("\n=== Example 2: Deep Neural Network (3-8-8-4-1) ===");
let deep_mlp = Mlp::new(
    &[3, 8, 8, 4, 1],
    &[Activations::Tanh, Activations::Tanh, Activations::Tanh, Activations::Sigmoid]
);
println!("Architecture: {:?}", deep_mlp.get_architecture());
println!("Total parameters: {}", deep_mlp.parameters().len());

println!("\n📊 Generating deep network visualization...");
deep_mlp.render_network_png("deep_network").unwrap();

println!("\n✨ Done! Generated visualizations:");
println!("  • network_architecture.png - Simple 2-4-1 network (layer view)");
println!("  • network_architecture_svg.svg - Same as SVG");
println!("  • computation_graph.png - Detailed scalar computation graph");
println!("  • deep_network.png - Deep 3-8-8-4-1 network (layer view)");
}
