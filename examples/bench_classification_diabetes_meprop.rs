use csv::Writer;
use rusty_axon::engine::Node;
use rusty_axon::nn::mlp::Mlp;
use rusty_axon::nn::activations::Activations;
use rusty_axon::optim::optimizer::Optimizer;
use rusty_axon::optim::meprop::MeProp;
use rusty_axon::loss::loss::Loss;
use rusty_axon::loss::cross_entropy::CrossEntropy;
use std::time::Instant;
use sysinfo::System;

// Load Pima diabetes dataset
fn load_pima(path: &str) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .comment(Some(b'#'))
        .from_path(path)
        .unwrap();
    
    let mut x = Vec::new();
    let mut y = Vec::new();
    
    for result in reader.records() {
        let record = result.unwrap();
        if record.len() != 9 {
            continue;
        }
        
        let mut features = Vec::new();
        let mut valid = true;
        
        for i in 0..8 {
            match record[i].parse::<f64>() {
                Ok(val) => features.push(val),
                Err(_) => {
                    valid = false;
                    break;
                }
            }
        }
        
        if !valid {
            continue;
        }
        
        match record[8].parse::<f64>() {
            Ok(val) => {
                x.push(features);
                y.push(val as usize);
            }
            Err(_) => continue,
        }
    }
    
    (x, y)
}

// Normalize features
fn normalize_features(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let max_vals = vec![17.0, 200.0, 122.0, 99.0, 846.0, 67.1, 2.42, 100.0];
    x.iter()
        .map(|row| {
            row.iter()
                .zip(max_vals.iter())
                .map(|(val, max)| val / max)
                .collect()
        })
        .collect()
}

// Prediction using argmax
fn predict(model: &Mlp, x: &[f64]) -> usize {
    let inputs: Vec<Node> = x.iter().map(|&v| Node::from(v)).collect();
    let logits = model.forward(&inputs);
    
    // Find class with highest logit value
    logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.get_value().total_cmp(&b.get_value()))
        .map(|(idx, _)| idx)
        .unwrap()
}

// F1 Score
fn f1_score(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let tp: usize = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(yt, yp)| **yt == 1 && **yp == 1)
        .count();
    
    let fp: usize = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(yt, yp)| **yt == 0 && **yp == 1)
        .count();
    
    let fn_: usize = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(yt, yp)| **yt == 1 && **yp == 0)
        .count();
    
    if tp + fp == 0 || tp + fn_ == 0 {
        return 0.0;
    }
    
    let precision = tp as f64 / (tp + fp) as f64;
    let recall = tp as f64 / (tp + fn_) as f64;
    
    if precision + recall == 0.0 {
        return 0.0;
    }
    
    2.0 * precision * recall / (precision + recall)
}

// Train/Test Split
fn train_test_split(
    x: Vec<Vec<f64>>,
    y: Vec<usize>,
    test_ratio: f64,
) -> (Vec<Vec<f64>>, Vec<usize>, Vec<Vec<f64>>, Vec<usize>) {
    use rand::seq::SliceRandom;
    use rand::rng;
    
    let mut data: Vec<(Vec<f64>, usize)> = x.into_iter().zip(y.into_iter()).collect();
    data.shuffle(&mut rng());
    
    let split = ((data.len() as f64) * (1.0 - test_ratio)) as usize;
    
    let (train, test) = data.split_at(split);
    
    let x_train: Vec<Vec<f64>> = train.iter().map(|(x, _)| x.clone()).collect();
    let y_train: Vec<usize> = train.iter().map(|(_, y)| *y).collect();
    let x_test: Vec<Vec<f64>> = test.iter().map(|(x, _)| x.clone()).collect();
    let y_test: Vec<usize> = test.iter().map(|(_, y)| *y).collect();
    
    (x_train, y_train, x_test, y_test)
}

// Convert label to one-hot encoding
fn one_hot(label: usize, num_classes: usize) -> Vec<Node> {
    (0..num_classes)
        .map(|i| Node::from(if i == label { 1.0 } else { 0.0 }))
        .collect()
}

// Compute accuracy on dataset
fn compute_accuracy(model: &Mlp, x: &[Vec<f64>], y: &[usize]) -> f64 {
    let correct = x.iter()
        .zip(y.iter())
        .filter(|(x_i, &y_i)| predict(model, x_i) == y_i)
        .count();
    correct as f64 / x.len() as f64
}

// Compute loss on dataset
fn compute_loss(model: &Mlp, x: &[Vec<f64>], y: &[usize], loss_fn: &CrossEntropy) -> f64 {
    let total_loss: f64 = x.iter()
        .zip(y.iter())
        .map(|(x_i, &y_i)| {
            let inputs: Vec<Node> = x_i.iter().map(|&v| Node::from(v)).collect();
            let logits = model.forward(&inputs);
            let targets = one_hot(y_i, 2);
            loss_fn.forward(&logits, &targets).get_value()
        })
        .sum();
    total_loss / x.len() as f64
}

fn main() {
    println!("=== Rusty-Axon: Classification Benchmark (Diabetes/Pima Indians) ===\n");
    
    // Load and prepare dataset
    let (x, y) = load_pima("python-tests/micrograd/classification-diabetes/dataset.csv");
    let x = normalize_features(&x);
    let (x_train, y_train, x_test, y_test) = train_test_split(x, y, 0.2);
    
    println!("Train samples: {}, Test samples: {}", x_train.len(), x_test.len());
    
    // Create network: 8 inputs -> 8 hidden -> 4 hidden -> 2 outputs (binary classification)
    let mlp = Mlp::new(
        &[8, 8, 4, 2],
        &[Activations::Tanh, Activations::Tanh, Activations::None],
    );
    
    // Hyperparameters
    let lr = 0.01;
    let epochs = 50;
    let batch_size = 32;
    
    println!("Architecture: {:?}", mlp.get_architecture());
    println!("Parameters: {}", mlp.parameters().len());
    println!("Learning rate: {}, Epochs: {}, Batch size: {}\n", lr, epochs, batch_size);
    
    // Setup loss function and optimizer
    let loss_fn = CrossEntropy::new(0.0); // No label smoothing
    let top_k = 0.5;
    let mut optimizer = MeProp::new(lr, mlp.parameters(), top_k);
    
    // Prepare CSV for metrics
    let mut wtr = Writer::from_path("python-tests/rusty-axon-rpi/classification-diabetes/rust_classification_metrics_meprop.csv").unwrap();
    wtr.write_record(&["Epoch", "Train_Loss", "Train_Acc", "Test_Loss", "Test_Acc", "F1", "Epoch_Time", "CPU_Usage", "RAM_Usage"])
        .unwrap();
    
    let start_total = Instant::now();
    let mut sys = System::new_all();
    
    // Training loop
    for epoch in 1..=epochs {
        let epoch_start = Instant::now();
        
        // Shuffle training data
        use rand::seq::SliceRandom;
        use rand::rng;
        let mut indices: Vec<usize> = (0..x_train.len()).collect();
        indices.shuffle(&mut rng());
        
        // Batch training
        for batch_start in (0..x_train.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(x_train.len());
            let batch_indices = &indices[batch_start..batch_end];
            
            // Zero gradients
            optimizer.zero_state();
            
            // Accumulate loss over batch
            let mut batch_loss = Node::from(0.0);
            for &idx in batch_indices {
                let inputs: Vec<Node> = x_train[idx].iter().map(|&v| Node::from(v)).collect();
                let logits = mlp.forward(&inputs);
                let targets = one_hot(y_train[idx], 2);
                let loss = loss_fn.forward(&logits, &targets);
                batch_loss = batch_loss + loss;
            }
            
            // Backward pass
            batch_loss.backward();
            
            // Update weights
            optimizer.step();
        }
        
        // Compute metrics
        let train_loss = compute_loss(&mlp, &x_train, &y_train, &loss_fn);
        let train_acc = compute_accuracy(&mlp, &x_train, &y_train);
        let test_loss = compute_loss(&mlp, &x_test, &y_test, &loss_fn);
        let test_acc = compute_accuracy(&mlp, &x_test, &y_test);
        
        let y_pred: Vec<usize> = x_test.iter().map(|x| predict(&mlp, x)).collect();
        let f1 = f1_score(&y_test, &y_pred);
        
        let epoch_time = epoch_start.elapsed().as_secs_f64();
        
        // CPU and RAM usage
        sys.refresh_all();
        let cpu_usage: f64 = sys.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>() as f64
            / sys.cpus().len() as f64;
        let ram_usage = sys.used_memory() as f64 / sys.total_memory() as f64 * 100.0;
        
        println!(
            "Epoch {:2} | Train Loss: {:.4} | Train Acc: {:.4} | Test Loss: {:.4} | Test Acc: {:.4} | F1: {:.4} | Time: {:.2}s | CPU: {:.1}% | RAM: {:.1}%",
            epoch, train_loss, train_acc, test_loss, test_acc, f1, epoch_time, cpu_usage, ram_usage
        );
        
        wtr.write_record(&[
            epoch.to_string(),
            format!("{:.6}", train_loss),
            format!("{:.6}", train_acc),
            format!("{:.6}", test_loss),
            format!("{:.6}", test_acc),
            format!("{:.6}", f1),
            format!("{:.6}", epoch_time),
            format!("{:.1}", cpu_usage),
            format!("{:.1}", ram_usage),
        ])
        .unwrap();
    }
    
    wtr.flush().unwrap();
    
    let total_time = start_total.elapsed().as_secs_f64();
    println!("\nTotal training time: {:.2}s", total_time);
    println!("Metrics saved to: python-tests/rusty-axon-rpi/classification-diabetes/rust_classification_metrics_meprop.csv");
}
