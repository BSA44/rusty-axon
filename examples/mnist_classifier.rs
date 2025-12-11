//! MNIST Handwritten Digits Classifier Demo (Parallel Version)
//! 
//! Architecture: 784 -> 64 -> 32 -> 10
//! Uses ParallelTrainer for multi-threaded batch processing.
//! 
//! Before running, prepare the data:
//!   cd python-tests && python prepare_mnist.py
//! 
//! Then run:
//!   cargo run --release --example mnist_classifier

use rusty_axon::engine::value::Node;
use rusty_axon::nn::mlp::Mlp;
use rusty_axon::nn::activations::Activations;
use rusty_axon::nn::parallel::ParallelTrainer;
use rusty_axon::loss::cross_entropy::CrossEntropy;
use rusty_axon::loss::loss::Loss;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

const NUM_CLASSES: usize = 10;

/// Load MNIST data from CSV file
/// Returns: (images as Vec<Vec<f64>>, labels as Vec<usize>)
fn load_mnist_csv(path: &str) -> Result<(Vec<Vec<f64>>, Vec<usize>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    let mut images = Vec::new();
    let mut labels = Vec::new();
    
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        
        // Skip header
        if i == 0 && line.starts_with("label") {
            continue;
        }
        
        let values: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
            .collect();
        
        if values.len() == 785 {  // 1 label + 784 pixels
            labels.push(values[0] as usize);
            images.push(values[1..].to_vec());
        }
    }
    
    Ok((images, labels))
}

/// Convert label to one-hot encoding (f64 version for parallel trainer)
fn one_hot(label: usize, num_classes: usize) -> Vec<f64> {
    (0..num_classes)
        .map(|i| if i == label { 1.0 } else { 0.0 })
        .collect()
}

/// Convert label to one-hot encoding (Node version for evaluation)
fn one_hot_node(label: usize, num_classes: usize) -> Vec<Node> {
    (0..num_classes)
        .map(|i| Node::from(if i == label { 1.0 } else { 0.0 }))
        .collect()
}

/// Get predicted class from output nodes
fn predict(outputs: &[Node]) -> usize {
    outputs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.get_value().partial_cmp(&b.get_value()).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Calculate accuracy on a dataset
fn evaluate(mlp: &Mlp, images: &[Vec<f64>], labels: &[usize]) -> f64 {
    let mut correct = 0;
    
    for (image, &label) in images.iter().zip(labels.iter()) {
        let inputs: Vec<Node> = image.iter().map(|&x| Node::from(x)).collect();
        let outputs = mlp.forward(&inputs);
        let predicted = predict(&outputs);
        
        if predicted == label {
            correct += 1;
        }
    }
    
    correct as f64 / labels.len() as f64 * 100.0
}

/// Confusion matrix for multi-class classification
struct ConfusionMatrix {
    matrix: [[usize; NUM_CLASSES]; NUM_CLASSES],  // [true_label][predicted_label]
}

impl ConfusionMatrix {
    fn new() -> Self {
        Self {
            matrix: [[0; NUM_CLASSES]; NUM_CLASSES],
        }
    }

    fn add(&mut self, true_label: usize, predicted: usize) {
        if true_label < NUM_CLASSES && predicted < NUM_CLASSES {
            self.matrix[true_label][predicted] += 1;
        }
    }

    /// Calculate per-class precision, recall, F1 and macro-averaged F1
    fn compute_f1_scores(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
        let mut precisions = vec![0.0; NUM_CLASSES];
        let mut recalls = vec![0.0; NUM_CLASSES];
        let mut f1_scores = vec![0.0; NUM_CLASSES];

        for class in 0..NUM_CLASSES {
            // True positives: correctly predicted as this class
            let tp = self.matrix[class][class] as f64;

            // False positives: predicted as this class but actually other classes
            let fp: f64 = (0..NUM_CLASSES)
                .filter(|&i| i != class)
                .map(|i| self.matrix[i][class] as f64)
                .sum();

            // False negatives: actually this class but predicted as other classes
            let fn_: f64 = (0..NUM_CLASSES)
                .filter(|&i| i != class)
                .map(|i| self.matrix[class][i] as f64)
                .sum();

            // Precision = TP / (TP + FP)
            precisions[class] = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };

            // Recall = TP / (TP + FN)
            recalls[class] = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };

            // F1 = 2 * (Precision * Recall) / (Precision + Recall)
            let p = precisions[class];
            let r = recalls[class];
            f1_scores[class] = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
        }

        // Macro F1: unweighted mean of F1 scores
        let macro_f1 = f1_scores.iter().sum::<f64>() / NUM_CLASSES as f64;

        (precisions, recalls, f1_scores, macro_f1)
    }
}

/// Evaluate model and compute F1 scores
fn evaluate_with_f1(mlp: &Mlp, images: &[Vec<f64>], labels: &[usize]) -> (f64, ConfusionMatrix) {
    let mut correct = 0;
    let mut confusion = ConfusionMatrix::new();
    
    for (image, &label) in images.iter().zip(labels.iter()) {
        let inputs: Vec<Node> = image.iter().map(|&x| Node::from(x)).collect();
        let outputs = mlp.forward(&inputs);
        let predicted = predict(&outputs);
        
        confusion.add(label, predicted);
        
        if predicted == label {
            correct += 1;
        }
    }
    
    let accuracy = correct as f64 / labels.len() as f64 * 100.0;
    (accuracy, confusion)
}

/// Compute average loss on a dataset
fn compute_loss(mlp: &Mlp, images: &[Vec<f64>], labels: &[usize], loss_fn: &CrossEntropy) -> f64 {
    let total_loss: f64 = images.iter()
        .zip(labels.iter())
        .map(|(image, &label)| {
            let inputs: Vec<Node> = image.iter().map(|&x| Node::from(x)).collect();
            let outputs = mlp.forward(&inputs);
            let targets = one_hot_node(label, NUM_CLASSES);
            loss_fn.forward(&outputs, &targets).get_value()
        })
        .sum();
    total_loss / images.len() as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("     MNIST Classifier - rusty-axon Parallel Demo            ");
    println!("     Architecture: 784 -> 64 -> 32 -> 10                    ");
    println!("============================================================");
    println!();

    // Show thread configuration
    let num_threads = rusty_axon::get_num_threads();
    println!("Using {} threads for parallel training", num_threads);
    println!();

    // Load data
    println!("[1/5] Loading MNIST data...");
    let train_path = "python-tests/mnist/mnist_train.csv";
    let test_path = "python-tests/mnist/mnist_test.csv";
    
    let (train_images, train_labels) = load_mnist_csv(train_path)
        .expect("Failed to load training data. Run: cd python-tests && python prepare_mnist.py");
    let (test_images, test_labels) = load_mnist_csv(test_path)
        .expect("Failed to load test data");
    
    println!("       Training samples: {}", train_images.len());
    println!("       Test samples: {}", test_images.len());

    // Create network: 784 -> 64 -> 32 -> 10
    println!();
    println!("[2/5] Creating neural network...");
    let architecture = vec![784, 64, 32, 10];
    let activations = vec![Activations::ReLU, Activations::ReLU, Activations::None];
    
    let mut mlp = Mlp::new(&architecture, &activations);
    
    let num_params: usize = mlp.parameters().len();
    println!("       Parameters: {}", num_params);
    println!("       Architecture: 784 -> 64 -> 32 -> 10");

    // Training setup
    let learning_rate = 0.1;  // Higher lr for averaged gradients in parallel training
    let epochs = 10;
    let batch_size =32;
    
    // Create parallel trainer
    let trainer = ParallelTrainer::new(learning_rate, architecture.clone(), activations.clone());
    let loss_fn = CrossEntropy::new(0.1);  // 10% label smoothing
    
    println!();
    println!("[3/5] Training (Parallel)...");
    println!("       Learning rate: {}", learning_rate);
    println!("       Epochs: {}", epochs);
    println!("       Batch size: {}", batch_size);
    println!("       Loss: CrossEntropy (label smoothing: 0.1)");
    println!("       Mode: Parallel ({} threads)", num_threads);
    println!();
    println!("       {:>5} | {:>10} | {:>10} | {:>10} | {:>8}", 
             "Epoch", "Train Loss", "Train Acc", "Test Acc", "Time");
    println!("       {}", "-".repeat(58));

    let total_start = Instant::now();
    
    // Prepare training indices for shuffling
    let n_train = train_images.len();
    
    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        
        // Shuffle training data indices
        use rand::seq::SliceRandom;
        use rand::rng;
        let mut indices: Vec<usize> = (0..n_train).collect();
        indices.shuffle(&mut rng());
        
        let mut epoch_losses = Vec::new();
        
        // Process batches in parallel
        for batch_start in (0..n_train).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_train);
            let batch_indices = &indices[batch_start..batch_end];
            
            // Prepare batch data: (inputs, one-hot targets)
            let batch: Vec<(Vec<f64>, Vec<f64>)> = batch_indices
                .iter()
                .map(|&idx| {
                    let inputs = train_images[idx].clone();
                    let targets = one_hot(train_labels[idx], NUM_CLASSES);
                    (inputs, targets)
                })
                .collect();
            
            // Train batch in parallel - all samples processed simultaneously!
            let batch_loss = trainer.train_batch(&mut mlp, &batch, &loss_fn);
            epoch_losses.push(batch_loss);
        }
        
        // Calculate metrics
        let avg_loss = epoch_losses.iter().sum::<f64>() / epoch_losses.len() as f64;
        let train_acc = evaluate(&mlp, &train_images, &train_labels);
        let test_acc = evaluate(&mlp, &test_images, &test_labels);
        let epoch_time = epoch_start.elapsed();
        
        println!("       {:>5} | {:>10.4} | {:>9.2}% | {:>9.2}% | {:>6.2}s",
                 epoch + 1, avg_loss, train_acc, test_acc, epoch_time.as_secs_f64());
    }
    
    let total_time = total_start.elapsed();
    
    // Final evaluation with F1 score
    println!();
    println!("[4/5] Final Evaluation on Test Set");
    let (final_test_acc, confusion) = evaluate_with_f1(&mlp, &test_images, &test_labels);
    let final_train_acc = evaluate(&mlp, &train_images, &train_labels);
    let final_loss = compute_loss(&mlp, &test_images, &test_labels, &loss_fn);
    let (precisions, recalls, f1_scores, macro_f1) = confusion.compute_f1_scores();
    
    println!();
    println!("       +-----------------------------------+");
    println!("       | Final Training Accuracy: {:6.2}% |", final_train_acc);
    println!("       | Final Test Accuracy:     {:6.2}% |", final_test_acc);
    println!("       | Final Test Loss:         {:6.4}  |", final_loss);
    println!("       | Macro F1 Score:          {:6.4}  |", macro_f1);
    println!("       | Total Training Time:     {:5.2}s  |", total_time.as_secs_f64());
    println!("       +-----------------------------------+");

    // Per-class metrics
    println!();
    println!("       Per-class metrics on test set:");
    println!("       {:>5} | {:>9} | {:>9} | {:>9}", "Class", "Precision", "Recall", "F1 Score");
    println!("       {}", "-".repeat(45));
    for class in 0..NUM_CLASSES {
        println!("       {:>5} | {:>9.4} | {:>9.4} | {:>9.4}",
                 class, precisions[class], recalls[class], f1_scores[class]);
    }
    println!("       {}", "-".repeat(45));
    println!("       {:>5} | {:>9} | {:>9} | {:>9.4}", "Macro", "-", "-", macro_f1);

    // Skip visualization - network is too large (784 inputs would create huge graph)
    println!();
    println!("[5/5] Sample Predictions:");
    
    // Show some predictions
    println!();
    println!("       Sample Predictions (first 10 test images):");
    println!("       {:>5} | {:>4} | {:>9} | {:>7}", "Image", "True", "Predicted", "Correct");
    println!("       {}", "-".repeat(40));
    
    for i in 0..10.min(test_images.len()) {
        let inputs: Vec<Node> = test_images[i].iter().map(|&x| Node::from(x)).collect();
        let outputs = mlp.forward(&inputs);
        let predicted = predict(&outputs);
        let correct = if predicted == test_labels[i] { "Yes" } else { "No" };
        
        println!("       {:>5} | {:>4} | {:>9} | {:>7}", 
                 i + 1, test_labels[i], predicted, correct);
    }

    println!();
    println!("Demo complete!");
    Ok(())
}
