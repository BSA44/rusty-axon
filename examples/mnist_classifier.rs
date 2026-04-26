//! MNIST Handwritten Digits Classifier Demo
//!
//! Architecture: 784 -> 64 -> 32 -> 10
//!
//! Before running, prepare the data:
//!   cd python-tests && python prepare_mnist.py
//!
//! Then run:
//!   cargo run --release --example mnist_classifier

use rusty_axon::engine::value::Node;
use rusty_axon::loss::cross_entropy::CrossEntropy;
use rusty_axon::loss::loss::Loss;
use rusty_axon::nn::activations::Activations;
use rusty_axon::nn::mlp::Mlp;
use rusty_axon::optim::optimizer::Optimizer;
use rusty_axon::optim::sgd::Sgd;

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

        if values.len() == 785 {
            // 1 label + 784 pixels
            labels.push(values[0] as usize);
            images.push(values[1..].to_vec());
        }
    }

    Ok((images, labels))
}

/// Convert label to one-hot encoding
fn one_hot(label: usize, num_classes: usize) -> Vec<Node> {
    (0..num_classes)
        .map(|i| Node::from(if i == label { 1.0 } else { 0.0 }))
        .collect()
}

/// Get predicted class from output nodes
fn predict(outputs: &[Node]) -> usize {
    outputs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.get_value().partial_cmp(&b.get_value()).unwrap())
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
    matrix: [[usize; NUM_CLASSES]; NUM_CLASSES], // [true_label][predicted_label]
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
            f1_scores[class] = if p + r > 0.0 {
                2.0 * p * r / (p + r)
            } else {
                0.0
            };
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("     MNIST Classifier - rusty-axon Demo                     ");
    println!("     Architecture: 784 -> 100 -> 50 -> 10                   ");
    println!("============================================================");
    println!();

    // Load data
    println!("[1/5] Loading MNIST data...");
    let train_path = "python-tests/mnist/mnist_train.csv";
    let test_path = "python-tests/mnist/mnist_test.csv";

    let (train_images, train_labels) = load_mnist_csv(train_path)
        .expect("Failed to load training data. Run: cd python-tests && python prepare_mnist.py");
    let (test_images, test_labels) = load_mnist_csv(test_path).expect("Failed to load test data");

    println!("       Training samples: {}", train_images.len());
    println!("       Test samples: {}", test_images.len());

    // Create network: 784 -> 64 -> 32 -> 10
    println!();
    println!("[2/5] Creating neural network...");
    let mlp = Mlp::new(
        &[784, 64, 32, 10],
        &[Activations::ReLU, Activations::ReLU, Activations::None], // No activation before softmax
    );

    let num_params: usize = mlp.parameters().len();
    println!("       Parameters: {}", num_params);
    println!("       Architecture: 784 -> 64 -> 32 -> 10");

    // Training setup
    let learning_rate = 0.01;
    let epochs = 10;
    let batch_size = 32;

    let mut optimizer = Sgd::new(learning_rate, mlp.parameters());
    let loss_fn = CrossEntropy::new(0.1); // 10% label smoothing

    println!();
    println!("[3/5] Training...");
    println!("       Learning rate: {}", learning_rate);
    println!("       Epochs: {}", epochs);
    println!("       Batch size: {}", batch_size);
    println!("       Loss: CrossEntropy (label smoothing: 0.1)");
    println!();
    println!(
        "       {:>5} | {:>10} | {:>10} | {:>10} | {:>8}",
        "Epoch", "Train Loss", "Train Acc", "Test Acc", "Time"
    );
    println!("       {}", "-".repeat(58));

    let total_start = Instant::now();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut epoch_correct = 0;
        let num_batches = (train_images.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(train_images.len());

            // Accumulate gradients over batch
            optimizer.zero_state();
            let mut batch_loss = Node::from(0.0);

            for i in start..end {
                let inputs: Vec<Node> = train_images[i].iter().map(|&x| Node::from(x)).collect();

                let outputs = mlp.forward(&inputs);
                let target = one_hot(train_labels[i], 10);

                // Track accuracy
                if predict(&outputs) == train_labels[i] {
                    epoch_correct += 1;
                }

                // Accumulate loss
                let sample_loss = loss_fn.forward(&outputs, &target);
                batch_loss = batch_loss + sample_loss;
            }

            // Average loss over batch
            let actual_batch_size = (end - start) as f64;
            batch_loss = batch_loss / actual_batch_size;
            epoch_loss += batch_loss.get_value();

            // Backward and update
            batch_loss.backward();
            optimizer.step();
        }

        // Calculate metrics
        let avg_loss = epoch_loss / num_batches as f64;
        let train_acc = epoch_correct as f64 / train_images.len() as f64 * 100.0;
        let test_acc = evaluate(&mlp, &test_images, &test_labels);
        let epoch_time = epoch_start.elapsed();

        println!(
            "       {:>5} | {:>10.4} | {:>9.2}% | {:>9.2}% | {:>6.2}s",
            epoch + 1,
            avg_loss,
            train_acc,
            test_acc,
            epoch_time.as_secs_f64()
        );
    }

    let total_time = total_start.elapsed();

    // Final evaluation with F1 score
    println!();
    println!("[4/5] Final Evaluation on Test Set");
    let (final_test_acc, confusion) = evaluate_with_f1(&mlp, &test_images, &test_labels);
    let final_train_acc = evaluate(&mlp, &train_images, &train_labels);
    let (precisions, recalls, f1_scores, macro_f1) = confusion.compute_f1_scores();

    println!();
    println!("       +-----------------------------------+");
    println!(
        "       | Final Training Accuracy: {:6.2}% |",
        final_train_acc
    );
    println!(
        "       | Final Test Accuracy:     {:6.2}% |",
        final_test_acc
    );
    println!("       | Macro F1 Score:          {:6.4}  |", macro_f1);
    println!(
        "       | Total Training Time:     {:5.2}s  |",
        total_time.as_secs_f64()
    );
    println!("       +-----------------------------------+");

    // Per-class metrics
    println!();
    println!("       Per-class metrics on test set:");
    println!(
        "       {:>5} | {:>9} | {:>9} | {:>9}",
        "Class", "Precision", "Recall", "F1 Score"
    );
    println!("       {}", "-".repeat(45));
    for class in 0..NUM_CLASSES {
        println!(
            "       {:>5} | {:>9.4} | {:>9.4} | {:>9.4}",
            class, precisions[class], recalls[class], f1_scores[class]
        );
    }
    println!("       {}", "-".repeat(45));
    println!(
        "       {:>5} | {:>9} | {:>9} | {:>9.4}",
        "Macro", "-", "-", macro_f1
    );

    // Visualize network architecture
    println!();
    println!("[5/5] Saving network architecture visualization...");
    mlp.visualize_network("mnist_network", "png").ok();

    // Show some predictions
    println!();
    println!("       Sample Predictions (first 10 test images):");
    println!(
        "       {:>5} | {:>4} | {:>9} | {:>7}",
        "Image", "True", "Predicted", "Correct"
    );
    println!("       {}", "-".repeat(40));

    for i in 0..10.min(test_images.len()) {
        let inputs: Vec<Node> = test_images[i].iter().map(|&x| Node::from(x)).collect();
        let outputs = mlp.forward(&inputs);
        let predicted = predict(&outputs);
        let correct = if predicted == test_labels[i] {
            "Yes"
        } else {
            "No"
        };

        println!(
            "       {:>5} | {:>4} | {:>9} | {:>7}",
            i + 1,
            test_labels[i],
            predicted,
            correct
        );
    }

    println!();
    println!("Demo complete! Network visualization saved to mnist_network.png");

    Ok(())
}
