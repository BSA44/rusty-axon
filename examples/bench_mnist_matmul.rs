//! MNIST matmul-kernel benchmark.
//!
//! Trains a tiny 784 -> 640 -> 320 -> 100 -> 10MLP for a few epochs on the MNIST
//! subset produced by `python-tests/prepare_mnist.py` and prints
//! machine-parseable timing/accuracy lines so an outer harness can
//! aggregate the matrixmultiply-vs-naive comparison.
//!
//! The active kernel is reported via the `matrixmultiply` / `naive-matmul`
//! cfg gates that drive [`engine::matmul::kernel`].
//!
//! Run paired comparisons with:
//!   cargo run --release --example bench_mnist_matmul                  # matrixmultiply
//!   cargo run --release --features naive-matmul --example bench_mnist_matmul  # naive

use rusty_axon::engine::value::Node;
use rusty_axon::loss::cross_entropy::CrossEntropy;
use rusty_axon::loss::loss::Loss;
use rusty_axon::nn::activations::Activations;
use rusty_axon::nn::mlp::Mlp;
use rusty_axon::optim::optimizer::Optimizer;
use rusty_axon::optim::sgd::Sgd;

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

#[cfg(all(feature = "matrixmultiply", not(feature = "naive-matmul")))]
const KERNEL: &str = "matrixmultiply";
#[cfg(any(not(feature = "matrixmultiply"), feature = "naive-matmul"))]
const KERNEL: &str = "naive";

fn load_mnist_csv(path: &str) -> Result<(Vec<Vec<f64>>, Vec<usize>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut images = Vec::new();
    let mut labels = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 && line.starts_with("label") {
            continue;
        }
        let values: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
            .collect();
        if values.len() == 785 {
            labels.push(values[0] as usize);
            images.push(values[1..].to_vec());
        }
    }
    Ok((images, labels))
}

fn one_hot(label: usize, num_classes: usize) -> Vec<Node> {
    (0..num_classes)
        .map(|i| Node::from(if i == label { 1.0 } else { 0.0 }))
        .collect()
}

fn predict(outputs: &[Node]) -> usize {
    outputs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.get_value().partial_cmp(&b.get_value()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn evaluate(mlp: &Mlp, images: &[Vec<f64>], labels: &[usize]) -> f64 {
    let mut correct = 0;
    for (image, &label) in images.iter().zip(labels.iter()) {
        let inputs: Vec<Node> = image.iter().map(|&x| Node::from(x)).collect();
        if predict(&mlp.forward(&inputs)) == label {
            correct += 1;
        }
    }
    correct as f64 / labels.len() as f64 * 100.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let epochs: usize = env::var("BENCH_EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let batch_size: usize = env::var("BENCH_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let train_limit: usize = env::var("BENCH_TRAIN_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let train_path = "python-tests/mnist/mnist_train.csv";
    let test_path = "python-tests/mnist/mnist_test.csv";

    let (mut train_images, mut train_labels) = load_mnist_csv(train_path)
        .expect("Failed to load training data. Run: python python-tests/prepare_mnist.py");
    let (test_images, test_labels) = load_mnist_csv(test_path).expect("Failed to load test data");

    if train_limit < train_images.len() {
        train_images.truncate(train_limit);
        train_labels.truncate(train_limit);
    }

    println!("[BENCH] kernel={}", KERNEL);
    println!("[BENCH] arch=784,640,320,100,10");
    println!("[BENCH] epochs={} batch={}", epochs, batch_size);
    println!(
        "[BENCH] train_samples={} test_samples={}",
        train_images.len(),
        test_images.len()
    );

    let mlp = Mlp::new(
        &[784, 640, 320, 100, 10],
        &[Activations::ReLU, Activations::ReLU, Activations::ReLU, Activations::None],
    );
    let mut optimizer = Sgd::new(0.01, mlp.parameters());
    let loss_fn = CrossEntropy::new(0.1);

    let total_start = Instant::now();

    for epoch in 1..=epochs {
        let epoch_start = Instant::now();
        let num_batches = (train_images.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(train_images.len());

            optimizer.zero_state();
            let mut batch_loss = Node::from(0.0);

            for i in start..end {
                let inputs: Vec<Node> = train_images[i].iter().map(|&x| Node::from(x)).collect();
                let outputs = mlp.forward(&inputs);
                let target = one_hot(train_labels[i], 10);
                batch_loss = batch_loss + loss_fn.forward(&outputs, &target);
            }

            let actual = (end - start) as f32;
            batch_loss = batch_loss / actual;
            batch_loss.backward();
            optimizer.step();
        }

        let epoch_time = epoch_start.elapsed().as_secs_f64();
        println!("[BENCH] epoch={} time_s={:.6}", epoch, epoch_time);
    }

    let total_time = total_start.elapsed().as_secs_f64();
    let test_acc = evaluate(&mlp, &test_images, &test_labels);

    println!("[BENCH] total_time_s={:.6}", total_time);
    println!("[BENCH] final_test_acc={:.4}", test_acc);

    Ok(())
}
