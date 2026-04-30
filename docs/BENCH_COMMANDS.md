# Phase 8 — Bench Commands

The Phase 8 benchmark suite uses [criterion](https://bheisler.github.io/criterion.rs/book/index.html) to measure forward latency, training-step cost, fine-tune-step cost, and the matmul-kernel speedup ratio across feature combinations. Every bench targets the same MLP shape — **`784 → 640 → 320 → 100 → 10`** with `[ReLU, ReLU, ReLU, None]` — so numbers across runs are directly comparable.

This document is the **command reference**. There is no automation; running the suite is a sequence of `cargo bench` invocations, each tagged with a `--save-baseline` so successive runs of different feature combos sit side-by-side under `target/criterion/` instead of clobbering each other.

---

## Prerequisites

1. **Toolchain** — stable Rust pinned by `rust-toolchain.toml`. No nightly required.
2. **MNIST data** — the bench files at `benches/training_step.rs` and `benches/finetune_step.rs` load the first N MNIST samples. Run the dataset prep script once if `python-tests/mnist/mnist_train.csv` is missing:

   ```bash
   python python-tests/prepare_mnist.py
   ```

3. **Disk space** — criterion writes per-bench JSON, raw samples, and HTML reports to `target/criterion/`. A full run of all four feature combos lands at ~30–80 MB.

---

## Sanity check

Before kicking off the full suite, confirm the criterion wiring works on the cheapest bench:

```bash
cargo bench --bench matmul_kernel -- --quick
```

`--quick` reduces criterion's measurement budget for a fast smoke test. Expect three sub-benches (`64x64x64`, `256x256x256`, `784x640 matvec`) to print timings within ~30 seconds total.

---

## Full suite

Run the four blocks below in order. Each block uses a distinct `--save-baseline` name so all four runs leave their `estimates.json` on disk for downstream parsing.

### 1. `matrixmultiply` (auto-NEON on aarch64) — train + fused-matmul benches

```bash
cargo bench --no-default-features --features train,matrixmultiply -- --save-baseline mm
```

**Compiles:** `forward_train`, `forward_train_legacy`, `forward_infer_f32`, `training_step`, `finetune_step`, `matmul_kernel`.
**Skips:** `forward_infer_i8` (no `quant-i8`).

**Cost (host x86_64, indicative):**
- `forward_train` ≈ tens of ms
- `forward_train_legacy` ≈ ~5 minutes total (10 samples × ~30 s each)
- `training_step` ≈ ~2 minutes total (20 samples)
- `finetune_step` ≈ ~1 minute total
- the rest are sub-second

### 2. Naive scalar kernel — same train benches for the speedup ratio

```bash
cargo bench --no-default-features --features train,naive-matmul -- --save-baseline naive
```

**Compiles:** same as block 1, but `sgemm_rm` resolves to the naive `for i for j for k` triple loop.
**Comparison key:** divide block-1's `matmul_kernel` numbers by block-2's to get the matrixmultiply-vs-naive speedup table that anchors the paper.

### 3. Inference-only `f32` — pure-`&[f32]` arena path

```bash
cargo bench --no-default-features --features inference,matrixmultiply -- --save-baseline infer_f32
```

**Compiles:** `forward_infer_f32`, `matmul_kernel`.
**Skips:** every train-gated bench (`forward_train`, `forward_train_legacy`, `training_step`, `finetune_step`) and the `quant-i8`-gated `forward_infer_i8`.

This is the **headline edge-inference latency** number for the paper.

> **Important:** `matrixmultiply` *must* be in the feature list. With
> `--no-default-features`, omitting it makes the kernel selector fall back
> to the naive triple loop, and the resulting numbers will mirror the
> `naive` block above instead of the matrixmultiply-accelerated path.

### 4. Inference + INT8 PTQ

```bash
cargo bench --no-default-features --features inference,quant-i8,matrixmultiply -- --save-baseline infer_i8
```

**Compiles:** `forward_infer_i8`, `forward_infer_f32`, `matmul_kernel`.
Compare `forward_infer_i8` against `forward_infer_f32` from block 3 to quantify the latency cost (or savings) of the dequant-fused path.

> Same caveat as block 3: include `matrixmultiply` in the feature list so
> the f32 portions of the workload still go through the optimized kernel.
> The dequant-fused INT8 path is scalar by design (matrixmultiply has no
> int8 GEMM) and stays slower than the f32 path — PTQ's win is binary
> size + RAM, not latency.

---

## Where the output lands

Per bench id, criterion writes:

```
target/criterion/<group>/<bench_id>/
├── new/                    ← latest run (clobbered each time)
├── mm/                     ← saved by `--save-baseline mm`
│   ├── estimates.json      ← mean, median, std_dev, CI — primary parse target
│   ├── benchmark.json      ← group_id, function_id, full_id metadata
│   ├── sample.json         ← raw [(iters, total_time_ns)] pairs
│   ├── tukey.json          ← outlier fences
│   └── raw.csv             ← same as sample.json, CSV form
├── naive/
├── infer_f32/
├── infer_i8/
└── report/                 ← HTML report
```

All times in `estimates.json` are **f64 nanoseconds**. The `mean.point_estimate` field is the headline number; `mean.confidence_interval.{lower_bound, upper_bound}` brackets the 95 % CI. Schema is stable across criterion 0.5.x (and back to 0.3.x).

The aggregated HTML report is at `target/criterion/report/index.html` — open it in a browser for distribution plots and per-bench drilldowns.

---

## Re-running and cleanup

- **Re-run a single bench:** `cargo bench --bench <name>` (e.g. `cargo bench --bench matmul_kernel`). Combine with the same `--save-baseline` name to update an existing baseline.
- **Drop one bench's history:** `rm -rf target/criterion/<group_or_id>` — criterion will rebuild on the next run.
- **Drop everything criterion knows:** `rm -rf target/criterion`.
- **Force a rebuild of the bench binaries** (e.g. after toggling features the bench code itself reads): `cargo clean -p rusty-axon` then re-run.

`cargo bench` always rebuilds with optimizations (`--release` is implicit), so a clean run is dominated by compile time on the first invocation per feature combo and by the `measurement_time` budget thereafter.
