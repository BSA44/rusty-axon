//! MNIST personalization fine-tune demo (Phase 11).
//!
//! Loads `mnist_pretrained.axn` (produced by
//! `examples/mnist_personalize_pretrain.rs`), evaluates the pretrained
//! 784->640->320->100->10 model on a clean held-out subset and on the same
//! subset after a fixed user-persona augmentation, then fine-tunes **only
//! the final 100->10 Linear** for a small number of epochs over a 200-sample
//! augmented training set.  Re-evaluates and writes `mnist_finetuned.axn`.
//!
//! The CSV trio is produced by `python-tests/generate_personalize_data.py`:
//!   mnist_personalize_clean.csv  -- 500 un-augmented samples
//!   mnist_personalize_test.csv   -- 500 augmented samples (eval)
//!   mnist_personalize_train.csv  -- 200 augmented samples (fine-tune set)
//!
//! Usage:
//!   rpi_finetune_mnist [model.axn] [train.csv] [test.csv] [clean.csv]
//!
//! Defaults map to the layout produced by the host-side helpers above.
//!
//! Fine-tune isolation (Phase 11 design note):
//! Rather than build a full Node graph through every layer, we run the
//! frozen prefix (layers 0..N-1) in pure f32 (`Linear::infer_into_f32`),
//! wrap the penultimate activations as fresh `Node` *leaves*, and call
//! `forward` on the last layer only.  Because the inputs to the head's
//! `MatMulTape` are leaves (`Operation::None`), `run_backward` skips the
//! `dx = Wt . d_out` propagation, so gradients accumulate exclusively in
//! the head layer's tape.  No `Node::detach` primitive needed.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use rusty_axon::engine::value::Node;
use rusty_axon::loss::cross_entropy::CrossEntropy;
use rusty_axon::loss::loss::Loss;
use rusty_axon::nn::activations::Activations;
use rusty_axon::nn::mlp::Mlp;
use rusty_axon::optim::optimizer::Optimizer;
use rusty_axon::optim::sgd::Sgd;
use sysinfo::{Pid, System};

const NUM_CLASSES: usize = 10;

fn rss_kib() -> u64 {
    let mut sys = System::new();
    let pid = Pid::from_u32(std::process::id());
    sys.refresh_process(pid);
    sys.process(pid).map(|p| p.memory()).unwrap_or(0) / 1024
}

fn load_mnist_csv(path: &str) -> std::io::Result<(Vec<Vec<f32>>, Vec<usize>)> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
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

fn argmax(out: &[f32]) -> usize {
    out.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn evaluate_f32(mlp: &Mlp, images: &[Vec<f32>], labels: &[usize]) -> f64 {
    let mut correct = 0usize;
    for (image, &label) in images.iter().zip(labels.iter()) {
        let out = mlp.infer(image);
        if argmax(&out) == label {
            correct += 1;
        }
    }
    correct as f64 / labels.len() as f64 * 100.0
}

/// Run layers `0..end` (exclusive) in pure f32, returning the output of
/// layer `end - 1`.  Mirrors `Mlp::infer` but stops before the head layer
/// so the fine-tune path can wrap the result as fresh `Node` leaves.
fn forward_prefix_f32(mlp: &Mlp, input: &[f32], end: usize) -> Vec<f32> {
    assert!(end >= 1 && end <= mlp.num_linear_layers());
    let mut current = vec![0.0_f32; mlp.layer(0).out_dim()];
    mlp.layer(0).infer_into_f32(input, &mut current);
    for i in 1..end {
        let mut next = vec![0.0_f32; mlp.layer(i).out_dim()];
        mlp.layer(i).infer_into_f32(&current, &mut next);
        current = next;
    }
    current
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let model_path = PathBuf::from(args.next().unwrap_or_else(|| "mnist_pretrained.axn".into()));
    let train_csv = args
        .next()
        .unwrap_or_else(|| "python-tests/mnist/mnist_personalize_train.csv".into());
    let test_csv = args
        .next()
        .unwrap_or_else(|| "python-tests/mnist/mnist_personalize_test.csv".into());
    let clean_csv = args
        .next()
        .unwrap_or_else(|| "python-tests/mnist/mnist_personalize_clean.csv".into());

    // Activation list must match what `mnist_personalize_pretrain.rs` saved.
    let activations = vec![
        Activations::ReLU,
        Activations::ReLU,
        Activations::ReLU,
        Activations::None,
    ];

    println!("[finetune] loading model {}", model_path.display());
    let mlp = match Mlp::load(&model_path, &activations) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[finetune] failed to load `{}`: {}", model_path.display(), e);
            return ExitCode::from(1);
        }
    };
    println!("[finetune] arch={:?}", mlp.get_architecture());
    let n_layers = mlp.num_linear_layers();
    let head_idx = n_layers - 1;
    println!(
        "[finetune] fine-tuning layer {} only ({} -> {}, {} params)",
        head_idx,
        mlp.layer(head_idx).in_dim(),
        mlp.layer(head_idx).out_dim(),
        mlp.layer(head_idx).in_dim() * mlp.layer(head_idx).out_dim()
            + mlp.layer(head_idx).out_dim()
    );

    let (train_images, train_labels) = match load_mnist_csv(&train_csv) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("[finetune] failed to load `{}`: {}", train_csv, e);
            return ExitCode::from(1);
        }
    };
    let (test_images, test_labels) = match load_mnist_csv(&test_csv) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("[finetune] failed to load `{}`: {}", test_csv, e);
            return ExitCode::from(1);
        }
    };
    let (clean_images, clean_labels) = match load_mnist_csv(&clean_csv) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("[finetune] failed to load `{}`: {}", clean_csv, e);
            return ExitCode::from(1);
        }
    };
    println!(
        "[finetune] samples: train={} test={} clean={}",
        train_images.len(),
        test_images.len(),
        clean_images.len()
    );

    let rss_load = rss_kib();

    // ---- Baseline ------------------------------------------------------
    let acc_clean_before = evaluate_f32(&mlp, &clean_images, &clean_labels);
    let acc_aug_before = evaluate_f32(&mlp, &test_images, &test_labels);
    println!(
        "[finetune] baseline clean={:.2}%  augmented={:.2}%  drop={:.2}pp",
        acc_clean_before,
        acc_aug_before,
        acc_clean_before - acc_aug_before
    );

    // ---- Fine-tune the head --------------------------------------------
    let lr: f32 = env::var("FT_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(0.01_f32);
    let epochs: usize = env::var("FT_EPOCHS").ok().and_then(|s| s.parse().ok()).unwrap_or(50usize);
    let batch_size: usize = env::var("FT_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(4usize);

    let head_params = mlp.parameters_for_layers(head_idx..n_layers);
    let mut optimizer = Sgd::new(lr, head_params);
    let loss_fn = CrossEntropy::new(0.0); // no label smoothing for fine-tune

    println!(
        "[finetune] head-only SGD: lr={} epochs={} batch={}",
        lr, epochs, batch_size
    );

    let total = Instant::now();
    let mut step_times_us: Vec<u64> = Vec::new();
    let mut last_loss = f32::NAN;

    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0_f32;
        let num_batches = (train_images.len() + batch_size - 1) / batch_size;

        for b in 0..num_batches {
            let t_step = Instant::now();
            let start = b * batch_size;
            let end = (start + batch_size).min(train_images.len());

            optimizer.zero_state();
            let mut batch_loss = Node::from(0.0_f32);

            for i in start..end {
                // Frozen prefix: pure-f32 forward through layers 0..head_idx.
                let hidden = forward_prefix_f32(&mlp, &train_images[i], head_idx);
                // Wrap as fresh leaves so the head's MatMulTape sees no upstream
                // and skips dx propagation.
                let leaves: Vec<Node> = hidden.iter().map(|&v| Node::from(v)).collect();
                // Train-path forward of the head layer only.
                let outputs = mlp.layer(head_idx).forward(&leaves);
                let target = one_hot(train_labels[i], NUM_CLASSES);
                batch_loss = batch_loss + loss_fn.forward(&outputs, &target);
            }
            let actual = (end - start) as f32;
            batch_loss = batch_loss / actual;
            epoch_loss += batch_loss.get_value();
            batch_loss.backward();
            optimizer.step();
            step_times_us.push(t_step.elapsed().as_micros() as u64);
        }

        last_loss = epoch_loss / num_batches as f32;
        if epoch == 1 || epoch == epochs || epoch % 10 == 0 {
            println!("[finetune] epoch={} loss={:.4}", epoch, last_loss);
        }
    }
    let total_s = total.elapsed().as_secs_f64();

    // Step-time summary (median + p95) across every batch step.
    step_times_us.sort_unstable();
    let median_us = step_times_us[step_times_us.len() / 2];
    let p95_us = step_times_us[(step_times_us.len() * 95) / 100];

    // ---- Re-evaluate ---------------------------------------------------
    let acc_clean_after = evaluate_f32(&mlp, &clean_images, &clean_labels);
    let acc_aug_after = evaluate_f32(&mlp, &test_images, &test_labels);

    println!(
        "[finetune] adapted  clean={:.2}%  augmented={:.2}%  delta_aug={:+.2}pp  delta_clean={:+.2}pp",
        acc_clean_after,
        acc_aug_after,
        acc_aug_after - acc_aug_before,
        acc_clean_after - acc_clean_before
    );
    println!(
        "[finetune] total_s={:.2}  step_median={}us  step_p95={}us  loss_last={:.4}",
        total_s, median_us, p95_us, last_loss
    );

    // ---- Save ----------------------------------------------------------
    let out_path = PathBuf::from(
        env::var("FT_OUT").unwrap_or_else(|_| "mnist_finetuned.axn".to_string()),
    );
    if let Err(e) = mlp.save(&out_path) {
        eprintln!("[finetune] save `{}` failed: {}", out_path.display(), e);
        return ExitCode::from(1);
    }
    let bytes = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    println!("[finetune] wrote {} ({} bytes)", out_path.display(), bytes);

    let rss_end = rss_kib();
    println!("[finetune] rss_load={}KiB rss_end={}KiB", rss_load, rss_end);

    ExitCode::SUCCESS
}
