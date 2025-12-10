use rusty_axon::engine::Node;
use rusty_axon::nn::mlp::Mlp;
use rusty_axon::nn::activations::Activations;
use rusty_axon::optim::optimizer::Optimizer;
use rusty_axon::nn::visualization::NetworkVisualizationConfig;

fn main() {
    println!("=== Rusty-Axon: XOR with ReLU Activation ===\n");

    println!("=== Training Example: XOR Problem (ReLU-based) ===\n");

    // XOR Dataset
    // Input: [x1, x2] -> Output: x1 XOR x2
    let training_data: Vec<(Vec<f64>, f64)> = vec![
        (vec![0.0, 0.0], 0.0),  // 0 XOR 0 = 0
        (vec![0.0, 1.0], 1.0),  // 0 XOR 1 = 1
        (vec![1.0, 0.0], 1.0),  // 1 XOR 0 = 1
        (vec![1.0, 1.0], 0.0),  // 1 XOR 1 = 0
    ];

    // Create network: 2 inputs -> 4 hidden (ReLU) -> 1 output (Sigmoid)
    // Using ReLU in hidden layer instead of Tanh
    let mlp = Mlp::new(
        &[2, 4, 1],
        &[Activations::ReLU, Activations::Sigmoid]
    );
    println!("Network Architecture: {:?}", mlp.get_architecture());
    println!("Total parameters: {}", mlp.parameters().len());
    println!("Activations: ReLU (hidden) -> Sigmoid (output)");
    
    // Visualize the network
    let edge_config = NetworkVisualizationConfig::with_colors(
        "aliceblue", "royalblue",
        "mistyrose", "coral",  // Different colors to distinguish from Tanh version
        "honeydew", "mediumseagreen",
    ).with_edges("darkslategray", 1.5);
    mlp.visualize_network_with_config("network_xor_relu", "png", &edge_config).unwrap();

    // Create optimizer - using standard SGD for ReLU
    // ReLU can benefit from slightly lower learning rate to avoid "dying ReLU"
    let learning_rate = 0.3;
    let mut optimizer = rusty_axon::optim::sgd::Sgd::new(learning_rate, mlp.parameters());

    // Training hyperparameters
    let epochs = 2000;
    let print_every = 100;

    println!("\nStarting training...");
    println!("Learning rate: {}", learning_rate);
    println!("Optimizer: SGD");
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

    let mut correct_count = 0;
    for (inputs_raw, target_raw) in &training_data {
        let inputs: Vec<Node> = inputs_raw.iter()
            .map(|&x| Node::from(x))
            .collect();
        
        let outputs = mlp.forward(&inputs);
        let prediction = outputs[0].get_value();
        let rounded = if prediction > 0.5 { 1.0 } else { 0.0 };
        let is_correct = (rounded - target_raw).abs() < 0.01;
        let correct = if is_correct { "+" } else { "-" };
        if is_correct { correct_count += 1; }

        println!(
            "  [{:.0}, {:.0}]    |   {:.0}    |   {:.4}   |    {}",
            inputs_raw[0], inputs_raw[1], target_raw, prediction, correct
        );
    }

    println!("\nAccuracy: {}/{} ({:.0}%)", correct_count, training_data.len(), 
             100.0 * correct_count as f64 / training_data.len() as f64);
    println!("\nTraining complete!");
}

