use rusty_axon::engine::Node;
use rusty_axon::nn::activations::Activations;
use rusty_axon::nn::mlp::Mlp;
use rusty_axon::nn::visualization::NetworkVisualizationConfig;
use rusty_axon::optim::optimizer::Optimizer; // Import the trait

fn main() {
    println!("=== Rusty-Axon: Autograd Engine Demo ===\n");

    println!("=== Training Example: XOR Problem ===\n");

    // XOR Dataset
    // Input: [x1, x2] -> Output: x1 XOR x2
    let training_data: Vec<(Vec<f64>, f64)> = vec![
        (vec![0.0, 0.0], 0.0), // 0 XOR 0 = 0
        (vec![0.0, 1.0], 1.0), // 0 XOR 1 = 1
        (vec![1.0, 0.0], 1.0), // 1 XOR 0 = 1
        (vec![1.0, 1.0], 0.0), // 1 XOR 1 = 0
    ];

    // Create network: 2 inputs -> 4 hidden -> 1 output
    let mlp = Mlp::new(&[2, 4, 1], &[Activations::Tanh, Activations::Sigmoid]);
    println!("Network Architecture: {:?}", mlp.get_architecture());
    println!("Total parameters: {}", mlp.parameters().len());
    let edge_config = NetworkVisualizationConfig::with_colors(
        "aliceblue",
        "royalblue",
        "lavenderblush",
        "orchid",
        "honeydew",
        "mediumseagreen",
    )
    .with_edges("navy", 1.5);
    mlp.visualize_network_with_config("network_xor", "png", &edge_config)
        .unwrap();
    // Create optimizer
    let learning_rate = 0.5;
    // Try different top_k values:
    // 1.0 = 100% (same as SGD)
    // 0.5 = 50% of parameters
    //0.2 = 20% of parameters
    let top_k = 0.5; // Update only 50% of parameters with largest gradients
    let mut optimizer =
        rusty_axon::optim::meprop::MeProp::new(learning_rate, mlp.parameters(), top_k);
    println!("MeProp top_k: {}%", top_k * 100.0);

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
            let inputs: Vec<Node> = inputs_raw.iter().map(|&x| Node::from(x)).collect();
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
            let avg_loss = total_loss / training_data.len() as f32;
            println!("Epoch {:4} | Loss: {:.6}", epoch, avg_loss);
        }
    }

    // Test the trained network
    println!("\n=== Testing Trained Network ===\n");
    println!("  Input     | Target | Prediction | Correct?");
    println!("  ----------|--------|------------|----------");

    for (inputs_raw, target_raw) in &training_data {
        let inputs: Vec<Node> = inputs_raw.iter().map(|&x| Node::from(x)).collect();

        let outputs = mlp.forward(&inputs);
        let prediction = outputs[0].get_value();
        let rounded = if prediction > 0.5 { 1.0 } else { 0.0 };
        let correct = if (rounded - target_raw).abs() < 0.01 {
            "+"
        } else {
            "-"
        };

        println!(
            "  [{:.0}, {:.0}]    |   {:.0}    |   {:.4}   |    {}",
            inputs_raw[0], inputs_raw[1], target_raw, prediction, correct
        );
    }

    println!("\nTraining complete!");
}
