//! Parallel Regression Benchmark (California Housing)
//!
//! This benchmark uses ParallelTrainer for data-parallel batch processing.
//! Compare with bench_regression_housing_sgd.rs to see parallelization benefits.
//!
//! Run with: cargo run --release --example bench_regression_housing_parallel

use csv::{Reader, Writer};
use rusty_axon::engine::Node;
use rusty_axon::nn::mlp::Mlp;
use rusty_axon::nn::activations::Activations;
use rusty_axon::nn::parallel::ParallelTrainer;
use rusty_axon::loss::Loss;
use rusty_axon::loss::MeanSquaredError;
use rusty_axon::loss::RootMeanSquaredError;
use std::time::Instant;
use sysinfo::System;

// Sigmoid normalization
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        x.exp() / (1.0 + x.exp())
    }
}

// Load California Housing dataset
fn load_csv(path: &str, limit: Option<usize>) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut reader = Reader::from_path(path).unwrap();
    let mut data = Vec::new();
    
    for (i, result) in reader.records().enumerate() {
        if let Some(lim) = limit {
            if i >= lim {
                break;
            }
        }
        
        let record = result.unwrap();
        let mut row = Vec::new();
        let mut valid = true;
        
        // Parse 8 features
        for i in 0..8 {
            match record[i].parse::<f64>() {
                Ok(val) => row.push(val),
                Err(_) => {
                    valid = false;
                    break;
                }
            }
        }
        
        if !valid {
            continue;
        }
        
        // Parse target
        match record[8].parse::<f64>() {
            Ok(val) => row.push(val),
            Err(_) => continue,
        }
        
        data.push(row);
    }
    
    // Handle NaN with column means
    let num_cols = 9;
    let mut col_means = vec![0.0; num_cols];
    let mut col_counts = vec![0; num_cols];
    
    for row in &data {
        for (i, &val) in row.iter().enumerate() {
            if !val.is_nan() {
                col_means[i] += val;
                col_counts[i] += 1;
            }
        }
    }
    
    for i in 0..num_cols {
        if col_counts[i] > 0 {
            col_means[i] /= col_counts[i] as f64;
        }
    }
    
    // Replace NaN with mean
    let mut cleaned_data = data;
    for row in &mut cleaned_data {
        for (i, val) in row.iter_mut().enumerate() {
            if val.is_nan() {
                *val = col_means[i];
            }
        }
    }
    
    // Split into X and y, apply sigmoid normalization
    let mut x = Vec::new();
    let mut y = Vec::new();
    
    for row in cleaned_data {
        let features: Vec<f64> = row[..8].iter().map(|&v| sigmoid(v)).collect();
        let target = sigmoid(row[8]);
        x.push(features);
        y.push(target);
    }
    
    (x, y)
}

// Compute metrics using framework's loss functions
fn compute_metrics(model: &Mlp, x: &[Vec<f64>], y: &[f64]) -> (f64, f64) {
    let mse_loss = MeanSquaredError;
    let rmse_loss = RootMeanSquaredError;
    
    let predictions: Vec<Node> = x.iter()
        .map(|x_i| {
            let inputs: Vec<Node> = x_i.iter().map(|&v| Node::from(v)).collect();
            let out = model.forward(&inputs);
            out[0].clone()
        })
        .collect();
    
    let targets: Vec<Node> = y.iter().map(|&v| Node::from(v)).collect();
    
    let mse = mse_loss.forward(&predictions, &targets).get_value();
    let rmse = rmse_loss.forward(&predictions, &targets).get_value();
    
    (mse, rmse)
}

fn main() {
    println!("=== Rusty-Axon: PARALLEL Regression Benchmark (California Housing) ===\n");
    
    // Show thread configuration
    let num_threads = rusty_axon::get_num_threads();
    println!("Using {} threads for parallel training\n", num_threads);
    
    let path = "python-tests/micrograd/regression-california-housing/dataset.csv";
    // Note: ParallelTrainer averages gradients
    // Use moderately higher lr than sequential (which sums gradients)
    let epochs = 10;
    let batch_size = 64;
    let lr = 0.1;  // Tuned for parallel training with averaged gradients
    let limit = Some(2000);
    
    // Load data
    let (x, y) = load_csv(path, limit);
    let n = x.len();
    
    println!("Loaded {} samples", n);
    println!("Features: {}, Target values: {}", x[0].len(), y.len());
    
    // Create model: 8 -> 16 -> 8 -> 1 (regression)
    let architecture = vec![8, 16, 8, 1];
    let activations = vec![Activations::Tanh, Activations::Tanh, Activations::None];
    
    let mut mlp = Mlp::new(&architecture, &activations);
    
    println!("Architecture: {:?}", mlp.get_architecture());
    println!("Parameters: {}", mlp.parameters().len());
    println!("Learning rate: {}, Epochs: {}, Batch size: {}\n", lr, epochs, batch_size);
    
    // Setup loss function and parallel trainer
    let loss_fn = MeanSquaredError;
    let trainer = ParallelTrainer::new(lr, architecture, activations);
    
    // Prepare CSV for metrics - write to rusty-axon-multi-thread folder
    let output_path = "python-tests/rusty-axon-multi-thread/regression-california-housing/rust_regression_metrics_parallel.csv";
    
    // Create directory if it doesn't exist
    std::fs::create_dir_all("python-tests/rusty-axon-multi-thread/regression-california-housing").ok();
    
    let mut wtr = Writer::from_path(output_path).unwrap();
    wtr.write_record(&["Epoch", "Loss", "RMSE", "CPU_Usage", "RAM_Usage", "Time_s"])
        .unwrap();
    
    let mut sys = System::new_all();
    
    // Training loop
    for epoch in 1..=epochs {
        let t0 = Instant::now();
        
        // Random permutation
        use rand::seq::SliceRandom;
        use rand::rng;
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng());
        
        let mut losses = Vec::new();
        
        // Batch training with parallel processing
        for batch_start in (0..n).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n);
            let batch_indices = &indices[batch_start..batch_end];
            
            // Prepare batch data for parallel trainer
            let batch: Vec<(Vec<f64>, Vec<f64>)> = batch_indices
                .iter()
                .map(|&idx| (x[idx].clone(), vec![y[idx]]))
                .collect();
            
            // Train batch in parallel
            let batch_loss = trainer.train_batch(&mut mlp, &batch, &loss_fn);
            losses.push(batch_loss);
        }
        
        // Compute metrics using framework's loss functions
        let mean_loss: f64 = losses.iter().sum::<f64>() / losses.len() as f64;
        let (_, rmse) = compute_metrics(&mlp, &x, &y);
        
        let epoch_time = t0.elapsed().as_secs_f64();
        
        // CPU and RAM usage
        sys.refresh_all();
        let cpu_usage: f64 = sys.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>() as f64
            / sys.cpus().len() as f64;
        let ram_usage = sys.used_memory() as f64 / sys.total_memory() as f64 * 100.0;
        
        println!(
            "Epoch {}/{} | Loss={:.6} | RMSE={:.6} | CPU={:.1}% | RAM={:.1}% | Time={:.1}s",
            epoch, epochs, mean_loss, rmse, cpu_usage, ram_usage, epoch_time
        );
        
        // Write to CSV
        wtr.write_record(&[
            epoch.to_string(),
            format!("{:.6}", mean_loss),
            format!("{:.6}", rmse),
            format!("{:.1}", cpu_usage),
            format!("{:.1}", ram_usage),
            format!("{:.1}", epoch_time),
        ])
        .unwrap();
    }
    
    wtr.flush().unwrap();
    println!("\nMetrics saved to: {}", output_path);
}
