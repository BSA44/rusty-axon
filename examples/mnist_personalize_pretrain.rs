//! Pretrain the base classifier for the MNIST personalization demo (Phase 11).
//!
//! Trains an
//! `Mlp::new(&[784, 640, 320, 100, 10], &[ReLU, ReLU, ReLU, None])` on
//! the full MNIST training set produced by
//! `python python-tests/prepare_mnist.py` and writes `mnist_pretrained.axn`
//! next to this binary.  Architecture matches `benches/common/mod.rs`,
//! `examples/mnist_classifier.rs`, and the Phase K Burn / TFLite Micro
//! comparison harnesses so on-device fine-tune wall-clock numbers compare
//! directly against the bench data.
//!
//! Run:
//!   cargo run --release --example mnist_personalize_pretrain
//!
//! Tunables (env vars):
//!   PRETRAIN_EPOCHS  default 8
//!   PRETRAIN_BATCH   default 32
//!   PRETRAIN_LR      default 0.01
//!   PRETRAIN_OUT     default mnist_pretrained.axn

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use rusty_axon::engine::value::Node;
use rusty_axon::loss::cross_entropy::CrossEntropy;
use rusty_axon::loss::loss::Loss;
use rusty_axon::nn::activations::Activations;
use rusty_axon::nn::mlp::Mlp;
use rusty_axon::optim::optimizer::Optimizer;
use rusty_axon::optim::sgd::Sgd;

fn load_mnist_csv(path: &str) -> Result<(Vec<Vec<f32>>, Vec<usize>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut images = Vec::new();
    let mut labels = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 && line.starts_with("label") {
            continue;
        }
        let values: Vec<f32> = line
            .split(',')
            .map(|s| s.trim().parse::<f32>().unwrap_or(0.0))
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
        .map(|i| Node::from(if i == label { 1.0_f32 } else { 0.0_f32 }))
        .collect()
}

fn evaluate(mlp: &Mlp, images: &[Vec<f32>], labels: &[usize]) -> f64 {
    let mut correct = 0usize;
    for (image, &label) in images.iter().zip(labels.iter()) {
        // Pure-f32 path: same arithmetic, no graph build.
        let out = mlp.infer(image);
        let pred = out
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        if pred == label {
            correct += 1;
        }
    }
    correct as f64 / labels.len() as f64 * 100.0
}

fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> T {
    env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let epochs: usize = parse_env("PRETRAIN_EPOCHS", 8usize);
    let batch_size: usize = parse_env("PRETRAIN_BATCH", 32usize);
    let lr: f32 = parse_env("PRETRAIN_LR", 0.01_f32);
    let out_path = PathBuf::from(parse_env("PRETRAIN_OUT", "mnist_pretrained.axn".to_string()));

    println!("[pretrain] arch=784,640,320,100,10  epochs={}  batch={}  lr={}", epochs, batch_size, lr);

    let (train_images, train_labels) = load_mnist_csv("python-tests/mnist/mnist_train.csv")
        .expect("load mnist_train.csv (run python python-tests/prepare_mnist.py first)");
    let (test_images, test_labels) =
        load_mnist_csv("python-tests/mnist/mnist_test.csv").expect("load mnist_test.csv");
    println!(
        "[pretrain] train={} test={}",
        train_images.len(),
        test_images.len()
    );

    let mlp = Mlp::new(
        &[784, 640, 320, 100, 10],
        &[
            Activations::ReLU,
            Activations::ReLU,
            Activations::ReLU,
            Activations::None,
        ],
    );
    let mut optimizer = Sgd::new(lr, mlp.parameters());
    let loss_fn = CrossEntropy::new(0.1);

    let total = Instant::now();
    for epoch in 1..=epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0_f32;
        let num_batches = (train_images.len() + batch_size - 1) / batch_size;

        for b in 0..num_batches {
            let start = b * batch_size;
            let end = (start + batch_size).min(train_images.len());

            optimizer.zero_state();
            let mut batch_loss = Node::from(0.0_f32);

            for i in start..end {
                let inputs: Vec<Node> = train_images[i].iter().map(|&x| Node::from(x)).collect();
                let outputs = mlp.forward(&inputs);
                let target = one_hot(train_labels[i], 10);
                batch_loss = batch_loss + loss_fn.forward(&outputs, &target);
            }
            let actual = (end - start) as f32;
            batch_loss = batch_loss / actual;
            epoch_loss += batch_loss.get_value();
            batch_loss.backward();
            optimizer.step();
        }

        let test_acc = evaluate(&mlp, &test_images, &test_labels);
        println!(
            "[pretrain] epoch={} loss={:.4} test_acc={:.2}% time_s={:.2}",
            epoch,
            epoch_loss / num_batches as f32,
            test_acc,
            epoch_start.elapsed().as_secs_f64()
        );
    }
    let total_s = total.elapsed().as_secs_f64();

    let final_acc = evaluate(&mlp, &test_images, &test_labels);
    println!("[pretrain] final test_acc={:.2}% total_s={:.2}", final_acc, total_s);

    mlp.save(&out_path)?;
    let bytes = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    println!("[pretrain] wrote {} ({} bytes)", out_path.display(), bytes);

    if final_acc < 95.0 {
        eprintln!(
            "[pretrain] WARNING: final test accuracy {:.2}% below the 97% paper target. \
             Consider raising PRETRAIN_EPOCHS.",
            final_acc
        );
    }
    Ok(())
}
