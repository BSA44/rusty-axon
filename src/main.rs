use rusty_axon::engine::Node;
use rusty_axon::nn::mlp::Mlp;
use rusty_axon::nn::activations::Activations;
use rusty_axon::optim::optimizer::Optimizer;  // Import the trait

fn main() {
    println!("=== Rusty-Axon: Autograd Engine Demo ===\n");

    // Sanity check that core structs are visible to the binary target.
    let _ = std::mem::size_of::<Node>();
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

 */
    // ===========================================
    // TRAINING EXAMPLE: Learning XOR Problem
    // ===========================================
    println!("=== Training Example: XOR Problem ===\n");

    // XOR Dataset (hardcoded)
    // Input: [x1, x2] -> Output: x1 XOR x2
    let training_data: Vec<(Vec<f64>, f64)> = vec![
        (vec![0.0, 0.0], 0.0),  // 0 XOR 0 = 0
        (vec![0.0, 1.0], 1.0),  // 0 XOR 1 = 1
        (vec![1.0, 0.0], 1.0),  // 1 XOR 0 = 1
        (vec![1.0, 1.0], 0.0),  // 1 XOR 1 = 0
    ];

    // Create network: 2 inputs -> 4 hidden -> 1 output
    let mlp = Mlp::new(
        &[2, 4, 1],
        &[Activations::Tanh, Activations::Sigmoid]
    );
    println!("Network Architecture: {:?}", mlp.get_architecture());
    println!("Total parameters: {}", mlp.parameters().len());

    // Create optimizer
    let learning_rate = 0.5;
    let mut optimizer = rusty_axon::optim::sgd::Sgd::new(learning_rate, mlp.parameters());

    // Training hyperparameters
    let epochs = 1000;
    let print_every = 100;

    println!("\nStarting training...");
    println!("Learning rate: {}", learning_rate);
    println!("Epochs: {}\n", epochs);

    // Training loop
    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (inputs_raw, target_raw) in &training_data {
            // 1. Zero gradients
            optimizer.zero_state();

            // 2. Convert inputs to Nodes
            let inputs: Vec<Node> = inputs_raw.iter()
                .map(|&x| Node::from(x))
                .collect();
            let target = Node::from(*target_raw);

            // 3. Forward pass
            let outputs = mlp.forward(&inputs);
            let prediction = outputs[0].clone();

            // 4. Compute loss (MSE for single output)
            let diff = prediction.clone() - target;
            let mut loss = diff.pow(2.0);

            // 5. Backward pass
            loss.backward();

            // 6. Update weights
            optimizer.step();

            total_loss += loss.get_value();
        }

        // Print progress
        if epoch % print_every == 0 || epoch == epochs - 1 {
            let avg_loss = total_loss / training_data.len() as f64;
            println!("Epoch {:4} | Loss: {:.6}", epoch, avg_loss);
        }
    }

    // Test the trained network
    println!("\n=== Testing Trained Network ===\n");
    println!("  Input     | Target | Prediction | Correct?");
    println!("  ----------|--------|------------|----------");

    for (inputs_raw, target_raw) in &training_data {
        let inputs: Vec<Node> = inputs_raw.iter()
            .map(|&x| Node::from(x))
            .collect();
        
        let outputs = mlp.forward(&inputs);
        let prediction = outputs[0].get_value();
        let rounded = if prediction > 0.5 { 1.0 } else { 0.0 };
        let correct = if (rounded - target_raw).abs() < 0.01 { "✓" } else { "✗" };

        println!(
            "  [{:.0}, {:.0}]    |   {:.0}    |   {:.4}   |    {}",
            inputs_raw[0], inputs_raw[1], target_raw, prediction, correct
        );
    }

    println!("\n✨ Training complete!");


}
