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
| Hardware          | host x86_64 **and** Pi Zero 2 W aarch64 — both rows for every cell        |
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
> are categorical differences with a one-line reason. The PR that
> populates these tables also updates [docs/PAPER.md](PAPER.md).

### Table 1 — Forward latency, single sample (microseconds, lower is better)

| Target            | rusty-axon (legacy scalar) | rusty-axon (fused, train) | rusty-axon (infer f32) | rusty-axon (infer i8) | Burn (NdArray) | TFLite Micro |
|-------------------|---------------------------:|--------------------------:|-----------------------:|----------------------:|---------------:|-------------:|
| host x86_64       |                  *pending* |                 *pending* |              *pending* |             *pending* |      *pending* |    *pending* |
| RPi Zero 2 W aarch64 |               *pending* |                 *pending* |              *pending* |             *pending* |      *pending* |    *pending* |

Sources:
* rusty-axon: `target/criterion/forward_train_legacy/`,
  `target/criterion/forward_train_fused_784_640_320_100_10/`,
  `target/criterion/forward_infer_f32_arena_784_640_320_100_10/`,
  `target/criterion/forward_infer_i8_arena_784_640_320_100_10/`
* Burn: `scripts/compare_burn/target/criterion/burn_forward_one_784_640_320_100_10/`
  and `burn_infer_into_buf_784_640_320_100_10/`
* TFLM: `./tflm_mnist bench 5000` (printed `mean_us`)

### Table 2 — Inference latency, batch=1, infer build (microseconds, lower is better)

The headline edge number. Same data as Table 1's "infer f32"/"infer i8"
columns for rusty-axon, isolated for narrative reasons.

| Target            | rusty-axon (infer f32) | rusty-axon (infer i8) | Burn (NdArray) | TFLite Micro |
|-------------------|-----------------------:|----------------------:|---------------:|-------------:|
| host x86_64       |              *pending* |             *pending* |      *pending* |    *pending* |
| RPi Zero 2 W      |              *pending* |             *pending* |      *pending* |    *pending* |

### Table 3 — Single training step, batch=32 (milliseconds, lower is better)

| Target            | rusty-axon SGD | rusty-axon MeProp | Burn (NdArray) | TFLite Micro                    |
|-------------------|---------------:|------------------:|---------------:|---------------------------------|
| host x86_64       |      *pending* |         *pending* |      *pending* | N/A — TFLM is inference-only    |
| RPi Zero 2 W      |      *pending* |         *pending* |      *pending* | N/A — TFLM is inference-only    |

Sources:
* rusty-axon: `target/criterion/training_step/`
* Burn: `scripts/compare_burn/target/criterion/burn_train_step_batch32_784_640_320_100_10/`

### Table 4 — Binary size, stripped, `release-edge` (KiB, lower is better)

Already partially populated for rusty-axon by `scripts/measure_binary_size.{sh,ps1}`
(see [BINARY_SIZE.md](BINARY_SIZE.md)).

| Combo | Description                                | Target            | Size (bytes) |
|-------|--------------------------------------------|-------------------|-------------:|
| C     | rusty-axon `release-edge` + `inference`    | host x86_64       |     198 144  |
| D     | rusty-axon `release-edge` + `inference,quant-i8` | host x86_64 |     200 192  |
| E     | rusty-axon `release-edge` + `inference`    | aarch64 (Pi)      |    *pending* |
| F     | rusty-axon `release-edge` + `inference,quant-i8` | aarch64 (Pi) |    *pending* |
| G     | Burn `release-edge` + NdArray              | host x86_64       |    *pending* |
| H     | Burn `release-edge` + NdArray              | aarch64 (Pi)      |    *pending* |
| I     | TFLite Micro, `-Os -flto -s`               | host x86_64       |    *pending* |
| J     | TFLite Micro, `-Os -flto -s`               | aarch64 (Pi)      |    *pending* |

### Table 5 — Peak RSS during inference, RPi Zero 2 W only (KiB, lower is better)

| Framework     | Peak RSS (KiB) |
|---------------|---------------:|
| rusty-axon    |      *pending* |
| Burn          |      *pending* |
| TFLite Micro  |      *pending* |

Sources: `sysinfo` for the Rust binaries (already wired into
`examples/min_inference.rs` and `examples/rpi_inference.rs`),
`getrusage(RUSAGE_SELF)` for the C harness (printed by `tflm_mnist`).

### Table 6 — Peak RSS during training, RPi Zero 2 W only (MiB, lower is better)

| Framework     | Peak RSS (MiB) |
|---------------|---------------:|
| rusty-axon    |      *pending* |
| Burn          |      *pending* |
| TFLite Micro  |  N/A — TFLM is inference-only |

### Table 7 — MNIST test accuracy

Sanity check that we're comparing equivalent models. Small differences
are expected (different optimizers, weight init, batch order); large
differences mean an architectural mismatch.

| Framework     | f32 accuracy | i8 accuracy           |
|---------------|-------------:|----------------------:|
| rusty-axon    |    *pending* |             *pending* |
| Burn          |    *pending* | N/A — no PTQ harness  |
| TFLite Micro  |    *pending* | N/A — float-only export |

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
| MNIST last-layer-only fine-tune (Phase 11) |  *pending* |  *pending* | N/A — TFLM is inference-only  |
| Sensor-drift full-model fine-tune (Phase 11) | *pending* |  *pending* | N/A — TFLM is inference-only  |

### Table 9 — PTQ accuracy delta (rusty-axon only)

| Model                        | f32 acc | i8 acc | Δ (pp) |
|------------------------------|--------:|-------:|-------:|
| MNIST 784→640→320→100→10     | *pending* | *pending* | *pending* |

### Table 10 — Sensor-drift adaptation (rusty-axon only)

| Time slice | MSE before fine-tune | MSE after fine-tune |
|------------|---------------------:|--------------------:|
| t1         |            *pending* |           *pending* |
| t2         |            *pending* |           *pending* |
| t3         |            *pending* |           *pending* |

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
