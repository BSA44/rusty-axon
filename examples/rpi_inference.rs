//! Pure-inference companion demo for the Raspberry Pi Zero 2 W (Phase 9).
//!
//! Builds under `cargo build --no-default-features --features inference`
//! (and the default `train` build).  Used by the Phase 10 binary-size
//! matrix and as the smoke-test artifact for the cross-compile pipeline.
//!
//! Behaviour:
//! 1. Load an `.axn` model.
//! 2. Build an `InferArena` (zero per-call allocation; Phase 8).
//! 3. Generate a deterministic synthetic input matching the model's first
//!    layer.  We avoid parsing stdin so the binary stays minimal — that
//!    code path lives in `examples/min_inference.rs`.
//! 4. Run `iters` warm inferences, time them, print median + p95 latency
//!    and resident-set size.
//!
//! Activation choice: `ReLU` on every hidden layer, `None` on the output
//! (`.axn` v1 does not serialize activations — Phase 5 design choice).
//!
//! Usage: `rpi_inference <model.axn> [num_layers] [iters]`
//!   - `num_layers` defaults to 4 (the paper's 784→640→320→100→10 MLP).
//!   - `iters`      defaults to 1000.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use rusty_axon::nn::activations::Activations;
use rusty_axon::nn::arena::InferArena;
use rusty_axon::nn::mlp::Mlp;
use sysinfo::{Pid, System};

fn parse_or<T: std::str::FromStr>(s: Option<String>, default: T) -> T {
    s.and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn rss_kib() -> u64 {
    let mut sys = System::new();
    let pid = Pid::from_u32(std::process::id());
    sys.refresh_process(pid);
    sys.process(pid).map(|p| p.memory()).unwrap_or(0) / 1024
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let model_path = match args.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: rpi_inference <model.axn> [num_layers] [iters]");
            return ExitCode::from(2);
        }
    };
    let num_layers: usize = parse_or(args.next(), 4_usize);
    let iters: usize = parse_or(args.next(), 1000_usize);

    let mut activations = vec![Activations::ReLU; num_layers];
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

    let in_dim = mlp.layer(0).in_dim();
    let out_dim = mlp.layer(mlp.num_linear_layers() - 1).out_dim();
    let mut arena = InferArena::for_mlp(&mlp);

    // Deterministic synthetic input: a low-amplitude sinusoid across `in_dim`.
    // Avoids `rand` (already a dep but pulls extra code under `inference`).
    let mut input = vec![0.0_f32; in_dim];
    for (i, x) in input.iter_mut().enumerate() {
        *x = ((i as f32) * 0.017).sin() * 0.5 + 0.5;
    }
    let mut output = vec![0.0_f32; out_dim];

    let rss_before = rss_kib();

    // Warm-up: page-in code, prime any one-shot allocator state.
    for _ in 0..10 {
        mlp.infer_into_arena(&input, &mut output, &mut arena);
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        mlp.infer_into_arena(&input, &mut output, &mut arena);
        samples.push(t0.elapsed().as_nanos() as u64);
    }
    samples.sort_unstable();
    let median_ns = samples[samples.len() / 2];
    let p95_ns = samples[(samples.len() * 95) / 100];

    let rss_after = rss_kib();

    println!("model:        {}", model_path.display());
    println!("arch:         {:?}", mlp.get_architecture());
    println!("arena bytes:  {}", arena.buffer_bytes());
    println!("iters:        {}", iters);
    println!(
        "median:       {:.3} ms ({} ns)",
        (median_ns as f64) / 1.0e6,
        median_ns
    );
    println!(
        "p95:          {:.3} ms ({} ns)",
        (p95_ns as f64) / 1.0e6,
        p95_ns
    );
    println!(
        "rss:          {} KiB before, {} KiB after",
        rss_before, rss_after
    );

    // First few outputs as a quick sanity print so smoke-tests (qemu / Pi)
    // can confirm the binary actually ran the network.
    let preview = output.iter().take(4).copied().collect::<Vec<_>>();
    println!("output[0..4]: {:?}", preview);

    ExitCode::SUCCESS
}
