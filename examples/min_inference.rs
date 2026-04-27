//! Minimal inference-only example.
//!
//! Builds under `cargo build --no-default-features --features inference`
//! (and under the default `train` build — `Mlp::infer` is always-on).  This
//! is the smallest realistic binary the Phase 10 binary-size matrix measures.
//!
//! Usage: `min_inference <model.axn> <num_layers>`
//!
//! Reads a single 0..1-normalised feature vector from stdin (one f32 per
//! whitespace-separated token), runs `Mlp::infer`, and prints the output.
//!
//! Activation choice is hard-coded: `ReLU` on every hidden layer, `None`
//! on the output.  v1 of the `.axn` format does not serialize activations
//! (Phase 5 design choice — kept minimal); callers that need a different
//! activation pattern should adapt this file or build their own loader.

use std::env;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::ExitCode;

use rusty_axon::nn::activations::Activations;
use rusty_axon::nn::mlp::Mlp;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let model_path = match args.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: min_inference <model.axn> <num_layers>");
            return ExitCode::from(2);
        }
    };
    let num_layers: usize = match args.next().and_then(|s| s.parse().ok()) {
        Some(n) if n >= 1 => n,
        _ => {
            eprintln!("usage: min_inference <model.axn> <num_layers>");
            return ExitCode::from(2);
        }
    };

    // ReLU on hidden layers, None on the output layer.
    let mut activations: Vec<Activations> = vec![Activations::ReLU; num_layers];
    if let Some(last) = activations.last_mut() {
        *last = Activations::None;
    }

    let mlp = match Mlp::load(&model_path, &activations) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to load `{}`: {}", model_path.display(), e);
            return ExitCode::from(1);
        }
    };

    let mut buf = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut buf) {
        eprintln!("error reading stdin: {}", e);
        return ExitCode::from(1);
    }
    let input: Vec<f32> = match buf
        .split_ascii_whitespace()
        .map(|s| s.parse::<f32>())
        .collect::<Result<_, _>>()
    {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error parsing stdin as f32: {}", e);
            return ExitCode::from(1);
        }
    };

    let output = mlp.infer(&input);
    for v in output {
        println!("{}", v);
    }
    ExitCode::SUCCESS
}
