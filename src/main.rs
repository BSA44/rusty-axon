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
let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
let inputs = vec![Node::from(1.0), Node::from(2.0)];
let mut output = mlp.forward(&inputs)[0].clone();
output.backward();

output.render_png("neural_network").unwrap();
}
