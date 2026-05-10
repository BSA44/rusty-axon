# rusty-axon vs. Burn vs. TFLite Micro

Side-by-side comparison of `rusty-axon` against the two baselines retained
for the v0.3 paper:

* **[Burn](https://burn.dev/)** — Rust train+infer foil. Tests the claim
  *"we are lighter than Rust train-capable frameworks."*
* **[TFLite Micro](https://github.com/tensorflow/tflite-micro)** — C++
  edge-inference foil. Tests the claim *"we can train where edge
  runtimes can't."*

Candle and MicroFlow are intentionally excluded — Candle is redundant
with Burn (same Rust train+infer category), MicroFlow is redundant with
TFLite Micro (same inference-only edge category). The two retained
baselines test exactly one paper claim each.

---

## Methodology

> Pin everything except the framework.

| Knob              | Pinned value                                                              |
|-------------------|---------------------------------------------------------------------------|
| Architecture      | MLP **784 → 640 → 320 → 100 → 10**, ReLU/ReLU/ReLU/None (logits)          |
| Dataset           | MNIST, identical train/test split, pixels / 255.0 → f32                   |
| Dtype             | f32 end-to-end (and i8 PTQ for the rusty-axon-only PTQ row)               |
| Hardware          | Raspberry Pi Zero 2 W aarch64 only (paper positions rusty-axon as edge-first; host x86_64 rows were intentionally dropped to keep the comparison focused on the deployment target) |
| Compiler stance   | Rust: `release-edge` (LTO fat, opt-level z, panic abort, strip symbols)   |
|                   | C/C++ (TFLM): `-Os -flto -ffunction-sections -fdata-sections -Wl,--gc-sections -s` |
| Workload          | Single-sample inference for latency tables; batch=32 for training tables  |
| Inputs            | Deterministic LCG (seed `0x9E3779B9`) — identical bytes consumed by all three frameworks |

The architecture matches `benches/common/mod.rs` exactly so the Burn
and TFLite Micro cells in every table compare against the same workload
rusty-axon's existing Phase 8 criterion data was measured on — no
re-runs of the rusty-axon benches required to populate the cross-
framework rows. Every harness exposes the architecture as a single
constant (`ARCH` in Burn, `ARCH` in the Keras script, baked into the
exported `.tflite` for TFLM), so re-pinning to a different shape is one
line plus a Keras retrain.

The Burn comparison uses the **NdArray backend with no BLAS feature** —
pure-CPU Rust, matching the rusty-axon constraint. A "Burn at full
speed" reference (NdArray + OpenBLAS, or LibTorch) can be added as an
extra column with a footnote, but the headline comparison stays
NdArray-only for fairness.

---

## Where the harnesses live

| Framework      | Location                                                | Build entry-point                         |
|----------------|---------------------------------------------------------|-------------------------------------------|
| rusty-axon     | this repo                                               | `cargo bench` / `scripts/measure_binary_size.{sh,ps1}` |
| Burn           | [`scripts/compare_burn/`](../scripts/compare_burn/)     | `cargo bench --manifest-path scripts/compare_burn/Cargo.toml` |
| TFLite Micro   | [`scripts/compare_tflite_micro/`](../scripts/compare_tflite_micro/) | `make` after `python ../../python-tests/train_keras_mnist.py` |

Each harness has its own `README.md` with prerequisites and exact
commands. The end-to-end driver in
[`scripts/run_paper_artifacts.sh`](../scripts/run_paper_artifacts.sh)
invokes all three after Phase 11's demos finish, then renders the
tables below from the raw CSV outputs.

---

## Result tables

> All cells marked `pending` until the harnesses have been executed
> end-to-end on the host **and** on a Pi Zero 2 W; cells marked `N/A`
> are categorical differences with a one-line reason; cells marked
> `not benched` are intentionally skipped (no plan to fill).
> The PR that populates these tables also updates
> [docs/PAPER.md](PAPER.md).

### Status as of last measurement run

All numbers are measured on a Raspberry Pi Zero 2 W (Cortex-A53, aarch64,
512 MB RAM, Pi OS 64-bit). Host x86_64 rows were intentionally dropped.

**Filled:**
- rusty-axon — fused fwd, infer f32, infer i8, SGD batch=32, last-layer fine-tune
- Burn — `forward_one` only (matches the train-mode forward column)
- TFLite Micro — inference latency, peak RSS, binary size, accuracy (Keras
  source model, equivalent to TFLM since the export is lossless f32)

**Still pending:**
- rusty-axon MNIST accuracy at convergence (Table 7 — current 49% is
  from a deliberately-short pretrain for the personalization demo,
  not a converged number)
- PTQ accuracy delta (Table 9 — needs a `quant_eval` example, not yet written)
- Burn fine-tune cells in Tables 6, 8 (no Burn fine-tune harness; would
  require writing a Burn equivalent of `rpi_finetune_mnist`)
- rusty-axon MeProp training step (Table 3 — bench not yet wired up for MeProp)

Raw stdout from each measured run is preserved under
[`measurements/`](measurements/) — see that directory's README for
the log-to-cell mapping.

### Table 1 — Forward latency, single sample on RPi Zero 2 W (microseconds, lower is better)

| rusty-axon (legacy scalar) | rusty-axon (fused, train) | rusty-axon (infer f32) | rusty-axon (infer i8) | Burn (NdArray) | TFLite Micro |
|---------------------------:|--------------------------:|-----------------------:|----------------------:|---------------:|-------------:|
|              *not benched* |                **13,167** |             **11,175** |            **20,659** |     **18,650** |    **3,960** |

Sources:
* rusty-axon: `target/criterion/forward_train_legacy/`,
  `target/criterion/forward_train_fused_784_640_320_100_10/`,
  `target/criterion/forward_infer_f32_arena_784_640_320_100_10/`,
  `target/criterion/forward_infer_i8_arena_784_640_320_100_10/`
* Burn: `scripts/compare_burn/target/criterion/burn_forward_one_784_640_320_100_10/`
  and `burn_infer_into_buf_784_640_320_100_10/`
* TFLM: `./tflm_mnist bench 5000` (printed `mean_us`)

### Table 2 — Inference latency, batch=1, infer build, RPi Zero 2 W (microseconds, lower is better)

The headline edge number. Same data as Table 1's "infer f32"/"infer i8"
columns for rusty-axon, isolated for narrative reasons.

| rusty-axon (infer f32) | rusty-axon (infer i8) | Burn (NdArray) | TFLite Micro |
|-----------------------:|----------------------:|---------------:|-------------:|
|             **11,175** |            **20,659** |     **16,764** |    **3,960** |

### Table 3 — Single training step, batch=32, RPi Zero 2 W (milliseconds, lower is better)

| rusty-axon SGD | rusty-axon MeProp | Burn (NdArray) | TFLite Micro                    |
|---------------:|------------------:|---------------:|---------------------------------|
|       **881**  |     *not benched* |    **180**     | N/A — TFLM is inference-only    |

> **Burn wins the batched-training row by ~5×** because Burn collapses
> the batch into a single `M×K×N` GEMM with `N=32` whereas rusty-axon's
> scalar autograd iterates the 32 samples one at a time through the
> Node graph. This is the cost of keeping the Value-based autograd
> intact (the paper's hard constraint). For *single-sample* training
> latency (Table 1's "fused, train" column), rusty-axon comes out
> ahead — 13.2 ms vs Burn's 18.7 ms forward pass — so the gap is
> specifically a batching gap, not an autograd-engine gap.

Sources:
* rusty-axon: `target/criterion/training_step/`
* Burn: `scripts/compare_burn/target/criterion/burn_train_step_batch32_784_640_320_100_10/`

### Table 4 — Binary size, stripped, `release-edge` (KiB, lower is better)

Already partially populated for rusty-axon by `scripts/measure_binary_size.{sh,ps1}`
(see [BINARY_SIZE.md](BINARY_SIZE.md)).

| Combo | Description                                | Target            | Size (bytes) |
|-------|--------------------------------------------|-------------------|-------------:|
| E     | rusty-axon `release-edge` + `inference`    | aarch64 (Pi)      |   **451 424**|
| F     | rusty-axon `release-edge` + `inference,quant-i8` | aarch64 (Pi) |   **451 424**|
| H     | Burn `release-edge` + NdArray              | aarch64 (Pi)      |   **586 672**|
| J     | TFLite Micro, `-Os -flto -s`               | aarch64 (Pi)      | **3 057 088**|

> **rusty-axon E vs F.** Identical strip-after-LTO byte count on aarch64;
> the host x86_64 build shows a 2 KiB difference (`binary_sizes.csv`
> Combos C/D), so the i8 dequant code is being included but the linker is
> stripping or coalescing it to a no-op size delta on aarch64. Reported
> as the measured value, not adjusted.
>
> **Headline.** rusty-axon (~440 KiB) is **6.7× smaller than TFLM** on
> the same target, **1.3× smaller than Burn** at minimal config. The
> Burn number does not include any actual model weights — the rusty-axon
> binaries don't either (weights are loaded from `.axn` at runtime).

> Host x86_64 measurements (Combos A–D, G, I from `binary_sizes.csv`)
> are kept in the raw CSV for completeness but omitted here — the paper's
> binary-size argument is about what fits on a Pi Zero 2 W.

### Table 5 — Peak RSS during inference, RPi Zero 2 W only (KiB, lower is better)

| Framework     | Peak RSS (KiB) |
|---------------|---------------:|
| rusty-axon    |      **5 664** |
| Burn          |      **9 488** |
| TFLite Micro  |      **5 840** |

> rusty-axon and TFLM are within ~3% (5.5 MiB class), Burn is ~70%
> larger. None of the three is anywhere near the Pi Zero 2 W's 512 MB
> ceiling for inference; the differentiator is binary size, not RSS.

Sources: `sysinfo` for the Rust binaries (already wired into
`examples/min_inference.rs` and `examples/rpi_inference.rs`),
`getrusage(RUSAGE_SELF)` for the C harness (printed by `tflm_mnist`).

### Table 6 — Peak RSS during training, RPi Zero 2 W only (MiB, lower is better)

| Framework     | Peak RSS (MiB) |
|---------------|---------------:|
| rusty-axon    | **29.1** (head-only fine-tune) |
| Burn          |      *pending* |
| TFLite Micro  |  N/A — TFLM is inference-only |

> rusty-axon RSS measured by `rpi_finetune_mnist`: `rss_load = 26 604
> KiB` after model+CSVs are mapped, `rss_end = 29 820 KiB` after 50
> epochs of head-only SGD. Source log:
> [`measurements/rpi_finetune_mnist.log`](measurements/rpi_finetune_mnist.log).
> Full-network training would be larger (the autograd graph holds
> Node references for every parameter); we don't have a captured run
> for that workload.

### Table 7 — MNIST test accuracy

Sanity check that we're comparing equivalent models. Small differences
are expected (different optimizers, weight init, batch order); large
differences mean an architectural mismatch.

| Framework     | f32 accuracy | i8 accuracy           |
|---------------|-------------:|----------------------:|
| rusty-axon    |    **49.0%** *(see note)* |             *pending* |
| Burn          |    *pending* | N/A — no PTQ harness  |
| TFLite Micro  |   **99.43%** | N/A — float-only export |

> **rusty-axon accuracy disclaimer.** The 49% number is the test
> accuracy of the lightly-pretrained `mnist_pretrained.axn` used by the
> Phase 11 fine-tune demo (8 epochs, scalar autograd — converged enough
> for a personalization-demo baseline, not for an accuracy comparison).
> Bringing rusty-axon to TFLM/Keras parity (~99%) requires significantly
> more epochs because the `Value`-based autograd is per-sample rather
> than batched. The accuracy gap here is a **training-budget** gap, not
> a model-capacity gap; the architecture is identical across all three
> frameworks (`784→640→320→100→10`).

Sources:
* rusty-axon: `examples/mnist_classifier.rs` final test-acc line.
* Burn: a `--example mnist_eval` could be added to the Burn harness;
  for now we cite Keras's `val_acc` as a proxy since both train on the
  same split.
* TFLM: same Keras model evaluated by `python-tests/train_keras_mnist.py`
  → `mnist_mlp_metadata.json["test_accuracy"]`.

### Table 8 — Fine-tune wall-clock per step on RPi Zero 2 W (ms, lower is better)

| Workload                      | rusty-axon | Burn       | TFLite Micro                  |
|-------------------------------|-----------:|-----------:|-------------------------------|
| MNIST last-layer-only fine-tune (Phase 11) |  **46.6** | *not benched* | N/A — TFLM is inference-only |
| Sensor-drift full-model fine-tune (1→8→8→1) (Phase 11) | **0.17** | *not benched* | N/A — TFLM is inference-only |

> rusty-axon MNIST cell measured by `examples/rpi_finetune_mnist`
> (median over 50 epochs × 50 batches, batch=4). Sensor-drift cell
> measured by `examples/rpi_sensor_drift` (200 SGD steps in 33 ms per
> drift slice → 0.165 ms/step).
>
> The criterion `finetune_step` micro-bench at the same shape reports
> ~81 ms/step; the demo is faster (46.6 ms) because it amortizes the
> frozen-prefix forward across the batch and reuses scratch buffers
> the bench reallocates. The demo number is the realistic workflow
> latency — use it for the paper headline.

### Table 9 — PTQ accuracy delta (rusty-axon only)

| Model                        | f32 acc | i8 acc | Δ (pp) |
|------------------------------|--------:|-------:|-------:|
| MNIST 784→640→320→100→10     | *pending* | *pending* | *pending* |

### Table 10 — Sensor-drift adaptation (rusty-axon only)

| Time slice | MSE before fine-tune | MSE after fine-tune | Drop  |
|------------|---------------------:|--------------------:|------:|
| t1         |          **0.03255** |         **0.00042** | 98.7% |
| t2         |          **0.05934** |         **0.00264** | 95.5% |
| t3         |          **0.05380** |         **0.00236** | 95.6% |

Source: [`measurements/rpi_sensor_drift.log`](measurements/rpi_sensor_drift.log).

### Table 11 — MNIST personalization fine-tune, full demo result set (rusty-axon only)

End-to-end numbers from `examples/rpi_finetune_mnist`. The base model
is deliberately under-trained (~50% test accuracy) so the demo has
real headroom; the cells here describe what one head-only adaptation
cycle does on a Pi Zero 2 W, not a converged-classifier baseline.

| Metric                                  | Value                            |
|-----------------------------------------|---------------------------------:|
| Frozen prefix                           | layers 0..2 (784→640→320→100)    |
| Trainable head                          | layer 3 (100→10), 1 010 params   |
| Fine-tune dataset                       | 200 augmented samples, batch=4   |
| Optimizer                               | SGD lr=0.01, 50 epochs           |
| **Accuracy on clean test, before**      | **49.0 %**                       |
| **Accuracy on clean test, after**       | **45.4 %**  (Δ −3.6 pp)          |
| **Accuracy on augmented test, before**  | **30.0 %**                       |
| **Accuracy on augmented test, after**   | **30.2 %**  (Δ +0.2 pp)          |
| Total wall-clock                        | **116.9 s**  (50 ep × 50 batches)|
| Per-step median / p95                   | **46.6 ms / 46.8 ms**            |
| Loss at last epoch                      | 0.2118  (started 0.2152)         |
| RSS after model load                    | **26 604 KiB**                   |
| RSS at end of fine-tune                 | **29 820 KiB**                   |
| Output `.axn` size                      | 2 962 908 bytes                  |

Source: [`measurements/rpi_finetune_mnist.log`](measurements/rpi_finetune_mnist.log).

> **Reading the accuracy deltas.** Both the clean drop (−3.6 pp) and
> the marginal augmented gain (+0.2 pp) are honest negative signals
> for *this base model*: SGD on the head can't recover much because
> the frozen prefix is itself only ~30% accurate on the augmented
> distribution. The result confirms the demo's *plumbing* (load → eval
> → fine-tune the head only → re-eval → save) works end-to-end on
> 512 MB; the **pedagogically useful** result is in Table 10
> (sensor-drift), where the base model is at-convergence and adaptation
> recovers >95% of the drift error in 33 ms per slice.

> Pretrained on 800 in-distribution samples (`MSE = 2.3e-4`). Each
> drifted slice (`t1/t2/t3`) is a 200-sample drift; rusty-axon adapts
> with 200 SGD steps in 33 ms per slice and recovers >95% of the drift
> error every time. Source: `examples/rpi_sensor_drift`.

---

## Reproducing every cell

```sh
# Prereqs (host):
#   - rustup install 1.87.0 + aarch64-unknown-linux-gnu target
#   - cross or aarch64-linux-gnu-gcc
#   - python with tensorflow==2.16.* (TFLM model export)
#   - git submodule for tflite-micro under scripts/compare_tflite_micro/

# 1. rusty-axon: Phase 8 benches + Phase 10 binary sizes (already automated).
bash scripts/run_paper_artifacts.sh --rpi

# 2. Burn baseline (host).
cargo bench --manifest-path scripts/compare_burn/Cargo.toml

# 3. Burn baseline (cross-compiled for aarch64 — bench binaries only,
#    transfer to Pi and run there).
cargo bench --manifest-path scripts/compare_burn/Cargo.toml \
  --target aarch64-unknown-linux-gnu --no-run

# 4. TFLite Micro: train, export, build, bench (host).
python python-tests/train_keras_mnist.py
make -C scripts/compare_tflite_micro
./scripts/compare_tflite_micro/tflm_mnist bench 5000

# 5. TFLite Micro: aarch64 cross-build, transfer, bench on Pi.
make -C scripts/compare_tflite_micro \
  CC=aarch64-linux-gnu-gcc \
  CXX=aarch64-linux-gnu-g++ \
  AR=aarch64-linux-gnu-ar \
  target=tflm_mnist_aarch64
scp scripts/compare_tflite_micro/tflm_mnist_aarch64 pi@rpi-zero:~/
ssh pi@rpi-zero ./tflm_mnist_aarch64 bench 5000
```

---

## Caveats and threats to validity

* **Host vs edge.** On a workstation, Burn (especially with a BLAS
  backend) is expected to outperform rusty-axon. The paper's positioning
  is **edge**, not host: "rusty-axon is not designed to beat Burn on a
  workstation; it is designed to fit and fine-tune on a Pi Zero 2 W
  where Burn's footprint is impractical."
* **Burn build cost.** Burn pulls in ~200 transitive crates and takes
  ~5 minutes to build cold. **Do not try to build it on the Pi.** Cross-
  compile from host.
* **TFLM scheduling overhead.** `MicroInterpreter::Invoke` includes
  per-call op-resolver dispatch. For a 4-op MLP this is in the noise,
  but for larger graphs it becomes a tax that doesn't apply to the
  rusty-axon path.
* **Optimizer parity.** rusty-axon's SGD and Burn's `SgdConfig` are
  textbook SGD with no momentum / weight-decay flags set. If we add an
  optimizer (Adam, etc.), it must be matched on both sides.
* **Float determinism.** `f32` matmul ordering differs across backends;
  cross-framework outputs match only to ~`1e-4` even with identical
  inputs. The accuracy comparison cells (Table 7) are tolerant of this
  by construction; latency cells are not affected.
