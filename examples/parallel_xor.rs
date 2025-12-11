//! Parallel XOR Training Example
//!
//! This example demonstrates parallel batch training using data parallelism.
//! It trains an MLP to solve the XOR problem, processing all 4 examples
//! simultaneously across multiple threads.
//!
//! Run with: cargo run --release --example parallel_xor

use rusty_axon::engine::Node;
use rusty_axon::nn::mlp::Mlp;
use rusty_axon::nn::activations::Activations;
use rusty_axon::nn::parallel::ParallelTrainer;
use rusty_axon::loss::mse::MeanSquaredError;
use std::time::Instant;

fn main() {
    println!("          Rusty-Axon: Parallel XOR Training Demo              ");

    // Show thread configuration
    let num_threads = rusty_axon::get_num_threads();
    println!(" Using {} threads for parallel training\n", num_threads);

    // XOR dataset as batch
    let batch: Vec<(Vec<f64>, Vec<f64>)> = vec![
        (vec![0.0, 0.0], vec![0.0]),  // 0 XOR 0 = 0
        (vec![0.0, 1.0], vec![1.0]),  // 0 XOR 1 = 1
        (vec![1.0, 0.0], vec![1.0]),  // 1 XOR 0 = 1
        (vec![1.0, 1.0], vec![0.0]),  // 1 XOR 1 = 0
    ];

    // Network architecture
    let architecture = vec![2, 4, 1];
    let activations = vec![Activations::Tanh, Activations::Sigmoid];

    println!("   Network Architecture: {:?}", architecture);
    println!("   Activations: Tanh → Sigmoid");

    // Create network and trainer
    let mut mlp = Mlp::new(&architecture, &activations);
    let trainer = ParallelTrainer::new(0.5, architecture.clone(), activations.clone());
    let loss_fn = MeanSquaredError;

    println!("   Total parameters: {}\n", mlp.parameters().len());

    // Training configuration
    let epochs = 1000;
    let print_every = 100;

    println!("   Training Configuration:");
    println!("   Learning rate: 0.5");
    println!("   Epochs: {}", epochs);
    println!("   Batch size: {} (full batch)\n", batch.len());

    // ============ PARALLEL TRAINING ============
    println!("                    PARALLEL TRAINING");

    let start = Instant::now();

    for epoch in 0..epochs {
        let loss = trainer.train_batch(&mut mlp, &batch, &loss_fn);

        if epoch % print_every == 0 || epoch == epochs - 1 {
            println!("  Epoch {:4} │ Loss: {:.6}", epoch, loss);
        }
    }

    let parallel_time = start.elapsed();
    println!("\n  Parallel training time: {:?}\n", parallel_time);

    // ============ TEST RESULTS ============
    println!("                      TEST RESULTS");

    println!("  Input     │ Expected │ Predicted │ Rounded │ Status");
    println!("  ──────────┼──────────┼───────────┼─────────┼────────");

    let mut all_correct = true;
    for (inputs, targets) in &batch {
        let input_nodes: Vec<Node> = inputs.iter()
            .map(|&x| Node::from(x))
            .collect();

        let output = mlp.forward(&input_nodes)[0].get_value();
        let rounded = if output > 0.5 { 1.0 } else { 0.0 };
        let expected = targets[0];
        let correct = (rounded - expected).abs() < 0.01;
        
        if !correct {
            all_correct = false;
        }

        let status = if correct { "✓" } else { "✗" };

        println!(
            "  [{:.0}, {:.0}]    │   {:.0}      │  {:.4}   │   {:.0}     │   {}",
            inputs[0], inputs[1], expected, output, rounded, status
        );
    }

    if all_correct {
        println!("   All predictions correct! XOR problem solved.");
    } else {
        println!("   Some predictions incorrect. Try more epochs or different learning rate.");
    }


    println!(" Try running with different thread counts:");
    println!("   RAYON_NUM_THREADS=1 cargo run --release --example parallel_xor");
    println!("   RAYON_NUM_THREADS=4 cargo run --release --example parallel_xor\n");
}

