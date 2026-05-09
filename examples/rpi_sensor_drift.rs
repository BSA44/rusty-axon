//! Synthetic sensor-drift on-device adaptation demo (Phase 11).
//!
//! Tiny `Mlp::new(&[1, 8, 8, 1], &[ReLU, ReLU, None])` (~50 params) trained
//! on a clean sensor distribution at t=0, then progressively re-fitted on
//! the Pi against drifted samples at t1, t2, t3.  At each stage the demo:
//!   1. evaluates MSE on the drifted set BEFORE adapting,
//!   2. fine-tunes the full network for `ADAPT_STEPS` steps over an
//!      `ADAPT_BUFFER`-sample window of drifted readings,
//!   3. re-evaluates MSE on the same drifted set,
//!   4. continues to the next stage with the just-adapted weights.
//!
//! Run on host or after cross-compiling for aarch64.  Generates the
//! `sensor_train.csv` / `sensor_drift_t{1,2,3}.csv` files first via
//! `python python-tests/generate_sensor_drift.py`.
//!
//! Tunables (env vars):
//!   PRETRAIN_EPOCHS  default 200    (host-side initial training, only
//!                                    runs if the .axn doesn't yet exist)
//!   PRETRAIN_LR      default 0.05
//!   ADAPT_STEPS      default 200
//!   ADAPT_BUFFER     default 100
//!   ADAPT_LR         default 0.05
//!
//! Usage:
//!   rpi_sensor_drift [model.axn] [train.csv] [t1.csv] [t2.csv] [t3.csv]

use std::env;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use rusty_axon::engine::value::Node;
use rusty_axon::loss::loss::Loss;
use rusty_axon::loss::mse::MeanSquaredError;
use rusty_axon::nn::activations::Activations;
use rusty_axon::nn::mlp::Mlp;
use rusty_axon::optim::optimizer::Optimizer;
use rusty_axon::optim::sgd::Sgd;

const ARCH: [usize; 4] = [1, 8, 8, 1];

fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> T {
    env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn load_csv_pairs(path: &str) -> std::io::Result<Vec<(f32, f32)>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 && line.starts_with("raw_reading") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() == 2 {
            let a: f32 = parts[0].parse().unwrap_or(0.0);
            let b: f32 = parts[1].parse().unwrap_or(0.0);
            out.push((a, b));
        }
    }
    Ok(out)
}

fn mse_eval(mlp: &Mlp, data: &[(f32, f32)]) -> f64 {
    let mut sum = 0.0_f64;
    let mut input = [0.0_f32; 1];
    for &(raw, cal) in data {
        input[0] = raw;
        let pred = mlp.infer(&input)[0];
        let d = (pred - cal) as f64;
        sum += d * d;
    }
    sum / data.len() as f64
}

fn pretrain(train: &[(f32, f32)]) -> Mlp {
    let epochs: usize = parse_env("PRETRAIN_EPOCHS", 200usize);
    let lr: f32 = parse_env("PRETRAIN_LR", 0.05_f32);
    println!(
        "[sensor] pretraining {:?} on {} samples (epochs={}, lr={})",
        ARCH,
        train.len(),
        epochs,
        lr
    );
    let mlp = Mlp::new(&ARCH, &[Activations::ReLU, Activations::ReLU, Activations::None]);
    let mut optimizer = Sgd::new(lr, mlp.parameters());
    let loss_fn = MeanSquaredError;

    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0_f32;
        for &(raw, cal) in train {
            optimizer.zero_state();
            let outputs = mlp.forward(&[Node::from(raw)]);
            let target = vec![Node::from(cal)];
            let mut sample_loss = loss_fn.forward(&outputs, &target);
            epoch_loss += sample_loss.get_value();
            sample_loss.backward();
            optimizer.step();
        }
        if epoch == 1 || epoch == epochs || epoch % 50 == 0 {
            println!(
                "[sensor] pretrain epoch={} avg_loss={:.6}",
                epoch,
                epoch_loss / train.len() as f32
            );
        }
    }
    mlp
}

/// One adapt cycle: `steps` SGD steps, each step drawing one (raw, cal)
/// pair from the most recent `buffer` samples of `data` (round-robin so the
/// model sees every fresh reading before any repeats).
fn adapt(mlp: &Mlp, data: &[(f32, f32)], steps: usize, buffer: usize, lr: f32) {
    let buffer = buffer.min(data.len());
    let window = &data[data.len() - buffer..];
    let mut optimizer = Sgd::new(lr, mlp.parameters());
    let loss_fn = MeanSquaredError;

    for step in 0..steps {
        let (raw, cal) = window[step % buffer];
        optimizer.zero_state();
        let outputs = mlp.forward(&[Node::from(raw)]);
        let target = vec![Node::from(cal)];
        let mut sample_loss = loss_fn.forward(&outputs, &target);
        sample_loss.backward();
        optimizer.step();
    }
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let model_path =
        args.next().unwrap_or_else(|| "sensor_initial.axn".to_string());
    let train_csv = args
        .next()
        .unwrap_or_else(|| "python-tests/sensor/sensor_train.csv".into());
    let t1 = args
        .next()
        .unwrap_or_else(|| "python-tests/sensor/sensor_drift_t1.csv".into());
    let t2 = args
        .next()
        .unwrap_or_else(|| "python-tests/sensor/sensor_drift_t2.csv".into());
    let t3 = args
        .next()
        .unwrap_or_else(|| "python-tests/sensor/sensor_drift_t3.csv".into());

    let activations = vec![Activations::ReLU, Activations::ReLU, Activations::None];

    // Load or pretrain.
    let mlp = if Path::new(&model_path).exists() {
        println!("[sensor] loading {}", model_path);
        match Mlp::load(Path::new(&model_path), &activations) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[sensor] failed to load `{}`: {}", model_path, e);
                return ExitCode::from(1);
            }
        }
    } else {
        let train = match load_csv_pairs(&train_csv) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[sensor] failed to load `{}`: {}", train_csv, e);
                return ExitCode::from(1);
            }
        };
        let mlp = pretrain(&train);
        if let Err(e) = mlp.save(Path::new(&model_path)) {
            eprintln!("[sensor] save initial `{}` failed: {}", model_path, e);
            return ExitCode::from(1);
        }
        println!("[sensor] saved {}", model_path);
        mlp
    };

    let train = match load_csv_pairs(&train_csv) {
        Ok(d) => d,
        Err(_) => Vec::new(),
    };
    if !train.is_empty() {
        let mse_train = mse_eval(&mlp, &train);
        println!("[sensor] mse on training distribution: {:.6}", mse_train);
    }

    let adapt_steps: usize = parse_env("ADAPT_STEPS", 200usize);
    let adapt_buffer: usize = parse_env("ADAPT_BUFFER", 100usize);
    let adapt_lr: f32 = parse_env("ADAPT_LR", 0.05_f32);
    println!(
        "[sensor] adapt config: steps={} buffer={} lr={}",
        adapt_steps, adapt_buffer, adapt_lr
    );

    for (label, path) in [("t1", t1.as_str()), ("t2", t2.as_str()), ("t3", t3.as_str())] {
        let data = match load_csv_pairs(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[sensor] failed to load `{}`: {}", path, e);
                return ExitCode::from(1);
            }
        };
        let mse_before = mse_eval(&mlp, &data);
        let t0 = Instant::now();
        adapt(&mlp, &data, adapt_steps, adapt_buffer, adapt_lr);
        let elapsed = t0.elapsed().as_secs_f64();
        let mse_after = mse_eval(&mlp, &data);
        let drop_pct = if mse_before > 0.0 {
            100.0 * (mse_before - mse_after) / mse_before
        } else {
            0.0
        };
        println!(
            "[sensor] {} N={} mse_before={:.6} mse_after={:.6} drop={:.1}% adapt_s={:.3}",
            label,
            data.len(),
            mse_before,
            mse_after,
            drop_pct,
            elapsed
        );
    }

    let final_path = env::var("ADAPTED_OUT").unwrap_or_else(|_| "sensor_adapted_t3.axn".to_string());
    if let Err(e) = mlp.save(Path::new(&final_path)) {
        eprintln!("[sensor] save adapted `{}` failed: {}", final_path, e);
        return ExitCode::from(1);
    }
    let bytes = std::fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);
    println!("[sensor] wrote {} ({} bytes)", final_path, bytes);

    ExitCode::SUCCESS
}
