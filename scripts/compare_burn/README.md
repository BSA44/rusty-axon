# compare-burn

Burn (NdArray backend) baseline harness for the rusty-axon paper.

## What this crate is

A standalone Cargo project (deliberately *outside* the rusty-axon
workspace) that pulls in the official [`burn`](https://burn.dev/) crate
as a normal dependency and rebuilds the same MLP architecture rusty-axon's
Phase 8 criterion suite uses (matching `benches/common/mod.rs`):

```
784 -> 640 -> 320 -> 100 -> 10
ReLU, ReLU, ReLU, None
```

It exposes three driver functions — `forward_one`, `infer_into_buf`,
`train_step_batch32` — each backing a criterion benchmark. The numbers
populate the Burn column of Tables 1, 2, and 3 in [`docs/PAPER.md`](../../docs/PAPER.md).

## Why a separate Cargo project

* **Apples-to-apples build profile.** Burn is built under the same
  `release-edge` profile as rusty-axon (LTO fat, opt-level z, strip,
  panic abort) so the binary-size cell of Table 4 is meaningful.
* **No workspace contamination.** Burn pulls in ~200 transitive crates;
  living outside the workspace keeps `cargo check`/`cargo test` in the
  main repo fast.
* **Pinned versions.** `burn = "=0.16.0"`, `criterion = "=0.5.1"` —
  paper-table reproducibility.

## Why the NdArray backend

The headline rusty-axon claim is *pure-CPU Rust without BLAS*. The fair
comparison is Burn under the same constraint, which means the NdArray
backend with no `blas-*` feature. A "Burn at full speed" reference
(NdArray + OpenBLAS, or LibTorch backend) can be added as an extra
column with a footnote — but the headline comparison stays NdArray-only.

## Running the benches

```sh
# All three benches (host).
cargo bench --manifest-path scripts/compare_burn/Cargo.toml

# Single bench.
cargo bench --manifest-path scripts/compare_burn/Cargo.toml --bench infer_into_buf

# Cross-compile for Pi Zero 2 W aarch64 (matches rusty-axon's RPi build).
cargo bench --manifest-path scripts/compare_burn/Cargo.toml \
  --target aarch64-unknown-linux-gnu --no-run
```

Criterion writes `target/criterion/**/estimates.json`; the existing
`scripts/parse_criterion.py` (Phase 8) consumes the same layout.

## Binary-size measurement

```sh
cargo build --manifest-path scripts/compare_burn/Cargo.toml \
  --profile release-edge --bin min_inference
ls -l scripts/compare_burn/target/release-edge/min_inference
```

That number drops into Table 4 alongside the rusty-axon
`min_inference` sizes from `binary_sizes.csv`.

## Caveats

* First build downloads + compiles Burn and its deps — expect ~5 minutes
  on a warm cache, longer cold. **Do not try to build this on the Pi.**
  Cross-compile from host.
* Burn's `Sgd` requires its own optimizer config; the harness wires the
  minimum (`SgdConfig::new()`) so the bench measures *step cost*, not
  config plumbing.
* Some Burn 0.16 APIs are still being stabilized; if a future bump
  breaks compilation, pin a newer version in `Cargo.toml` and update
  the imports — the public surface used here (Module, Linear,
  CrossEntropyLoss, SgdConfig, Autodiff) is the long-term API.
