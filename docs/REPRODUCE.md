# Reproducing the v0.3 paper numbers

**Goal.** Starting from a clean checkout, end up with every numeric cell
in [`COMPARISON.md`](COMPARISON.md) backed by a captured stdout log under
[`measurements/`](measurements/).

This is the single entry point for reproducibility. For background on
*why* a particular knob is set, see:

- [`PAPER_REWORK_PLAN.md`](PAPER_REWORK_PLAN.md) — the 13-phase plan that
  shaped what gets measured.
- [`COMPARISON.md`](COMPARISON.md) — the result tables themselves.
- [`RPI_DEPLOY.md`](RPI_DEPLOY.md) — cross-compile toolchain details.
- [`AXN_FORMAT.md`](AXN_FORMAT.md) — the `.axn` model wire format.
- [`BENCH_COMMANDS.md`](BENCH_COMMANDS.md) — feature-flag matrix per bench.

---

## 0. TL;DR

```
host  : prepare data (Python) + cross-compile binaries (cross + WSL)
Pi    : run binaries, capture stdout
host  : extract numbers, paste into COMPARISON.md, drop logs into docs/measurements/
```

Allow ~2 hours wall-clock if everything works first try, ~half a day if
you hit one of the known gotchas in §8.

---

## 1. Hardware required

| Item                 | Notes                                                 |
|----------------------|-------------------------------------------------------|
| Raspberry Pi Zero 2 W | Cortex-A53 aarch64, 512 MB RAM, Pi OS 64-bit Lite     |
| Dev host             | Windows 10/11 + WSL2 Ubuntu 20.04 (or Linux directly) |
| Network              | Pi reachable from host over SSH (LAN or USB-Ethernet) |
| Free disk            | ~5 GB on host (TFLM tree + cross artifacts)           |

The Pi must be **64-bit** Pi OS. 32-bit images will refuse to load
`aarch64` ELFs.

---

## 2. One-time host setup

### 2.1 Rust toolchain

```bash
# Pin to the version recorded in rust-toolchain.toml
rustup show                       # accept the prompt to install if needed
rustup target add aarch64-unknown-linux-gnu

# `cross` is the recommended cross-compile driver
cargo install cross --locked
```

### 2.2 WSL packages (cross-compile + TFLM build)

Inside WSL Ubuntu 20.04:

```bash
sudo apt update
sudo apt install -y \
    build-essential git wget unzip xxd \
    gcc-aarch64-linux-gnu g++-aarch64-linux-gnu binutils-aarch64-linux-gnu \
    python3 python3-pip python3-venv \
    python3-numpy python3-pil python3-six python3-wheel
```

The `python3-numpy` / `python3-pil` packages are needed at TFLM **build**
time (its codegen scripts use them); the actual TFLM runtime has no
Python dependency.

### 2.3 Python venv for Keras export

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install tensorflow              # 2.16+ tested; 2.21.0 used in our run
```

### 2.4 Pi: SSH + GNU `time`

On the Pi (over SSH or direct console):

```bash
sudo apt install -y time            # /usr/bin/time -v for RSS measurement
mkdir -p ~/axon_demo ~/axon_bench ~/burn_bench
```

---

## 3. Phase A — prepare data on host

### A1. MNIST CSV for the rusty-axon benches and demos

```bash
cd <repo-root>
python python-tests/prepare_mnist.py
# Produces: python-tests/mnist/mnist_train.csv  (50 000 rows)
#           python-tests/mnist/mnist_test.csv   (10 000 rows)
```

### A2. Personalization-demo CSVs

```bash
python python-tests/generate_personalize_data.py
# Produces:
#   python-tests/mnist/mnist_personalize_train.csv  (200 augmented)
#   python-tests/mnist/mnist_personalize_test.csv   (500 augmented)
#   python-tests/mnist/mnist_personalize_clean.csv  (500 clean)
```

### A3. Train the Keras MNIST model and export TFLite

```bash
python python-tests/train_keras_mnist.py
# Produces:
#   python-tests/mnist_mlp.tflite                       (~3 MB float32)
#   scripts/compare_tflite_micro/mnist_mlp_tflite.h     (xxd-style C header)
#   python-tests/mnist_mlp_metadata.json                (test_accuracy, etc.)
```

The `.h` is what TFLM links against — it's the model bytes embedded as
a C array. `mnist_mlp_metadata.json["test_accuracy"]` is the value that
fills Table 7's TFLite Micro row.

### A4. Pretrain the rusty-axon base model for the personalization demo

```bash
cargo run --release --example mnist_personalize_pretrain
# Produces: mnist_pretrained.axn (in repo root)
# Prints final test_acc -> Table 7 rusty-axon f32 cell
```

Defaults: 8 epochs, batch=32, lr=0.01. Bump `PRETRAIN_EPOCHS` if you
want a converged accuracy (the default is deliberately short for the
personalization demo — see Table 11 disclaimer in COMPARISON.md).

---

## 4. Phase B — cross-compile everything

All commands assume you're in the repo root (WSL or Linux host).

### B1. rusty-axon criterion benches

```bash
cross bench --target aarch64-unknown-linux-gnu --profile release-edge \
            --bench forward_train --no-run
cross bench --target aarch64-unknown-linux-gnu --profile release-edge \
            --bench forward_infer_f32 --no-run \
            --no-default-features --features inference,matrixmultiply
cross bench --target aarch64-unknown-linux-gnu --profile release-edge \
            --bench forward_infer_i8 --no-run \
            --no-default-features --features inference,matrixmultiply,quant-i8
cross bench --target aarch64-unknown-linux-gnu --profile release-edge \
            --bench training_step --no-run
cross bench --target aarch64-unknown-linux-gnu --profile release-edge \
            --bench finetune_step --no-run
cross bench --target aarch64-unknown-linux-gnu --profile release-edge \
            --bench matmul_kernel --no-run
```

Bench binaries land at `target/aarch64-unknown-linux-gnu/release-edge/deps/<bench>-<hash>`.
There is **always** a `.d` dependency manifest next to each ELF — strip
it from any glob with `grep -v '\.d$'` (see §8.5).

### B2. rusty-axon inference example (Combos E and F)

```bash
# f32 build -> Combo E
cross build --target aarch64-unknown-linux-gnu --profile release-edge \
            --example rpi_inference \
            --no-default-features --features inference,matrixmultiply
cp target/aarch64-unknown-linux-gnu/release-edge/examples/rpi_inference \
   target/aarch64-unknown-linux-gnu/release-edge/examples/rpi_inference_f32

# i8 build -> Combo F (force re-link, see §8.4)
rm target/aarch64-unknown-linux-gnu/release-edge/examples/rpi_inference
cross build --target aarch64-unknown-linux-gnu --profile release-edge \
            --example rpi_inference \
            --no-default-features --features inference,matrixmultiply,quant-i8
cp target/aarch64-unknown-linux-gnu/release-edge/examples/rpi_inference \
   target/aarch64-unknown-linux-gnu/release-edge/examples/rpi_inference_i8
```

### B3. Phase 11 demos

```bash
cross build --target aarch64-unknown-linux-gnu --profile release-edge \
            --example rpi_finetune_mnist
cross build --target aarch64-unknown-linux-gnu --profile release-edge \
            --example rpi_sensor_drift
```

### B4. Burn benches and minimal inference binary

```bash
# Bench binaries (forward_one, infer_into_buf, train_step_batch32)
cross bench --manifest-path scripts/compare_burn/Cargo.toml \
            --target aarch64-unknown-linux-gnu --profile release-edge \
            --bench forward_one --no-run
cross bench --manifest-path scripts/compare_burn/Cargo.toml \
            --target aarch64-unknown-linux-gnu --profile release-edge \
            --bench infer_into_buf --no-run
cross bench --manifest-path scripts/compare_burn/Cargo.toml \
            --target aarch64-unknown-linux-gnu --profile release-edge \
            --bench train_step_batch32 --no-run

# Minimum-footprint inference binary (Combo H + Burn RSS)
cross build --manifest-path scripts/compare_burn/Cargo.toml \
            --target aarch64-unknown-linux-gnu --profile release-edge \
            --bin min_inference
```

### B5. TFLite Micro harness (cross-compile in WSL — see §8.1)

```bash
cd scripts/compare_tflite_micro

# Clone TFLM if not already present (do this once)
git clone --depth 1 https://github.com/tensorflow/tflite-micro.git

# Build microlite for aarch64 — CRITICAL: pass TARGET_TOOLCHAIN_PREFIX
# AND CC_TOOL/CXX_TOOL/AR_TOOL so the lib is genuinely aarch64
make -j$(nproc) -C tflite-micro \
     -f tensorflow/lite/micro/tools/make/Makefile \
     TARGET=linux TARGET_ARCH=aarch64 \
     TARGET_TOOLCHAIN_PREFIX=aarch64-linux-gnu- \
     CC_TOOL=aarch64-linux-gnu-gcc \
     CXX_TOOL=aarch64-linux-gnu-g++ \
     AR_TOOL=aarch64-linux-gnu-ar \
     microlite

# Verify the lib is actually aarch64 (see §8.2 for what a wrong-arch lib
# looks like in `file` output)
aarch64-linux-gnu-objdump -a tflite-micro/gen/linux_aarch64_default_gcc/lib/libtensorflow-microlite.a \
    | grep 'file format' | head -3
# expect: file format elf64-littleaarch64

# Build the harness (the harness Makefile must pass -DTF_LITE_STATIC_MEMORY,
# see §8.3 for the ABI-skew bug if you remove it)
make TFLM_DIR=tflite-micro \
     CC=aarch64-linux-gnu-gcc \
     CXX=aarch64-linux-gnu-g++ \
     AR=aarch64-linux-gnu-ar \
     target=tflm_mnist_aarch64

file tflm_mnist_aarch64
# expect: ELF 64-bit LSB executable, ARM aarch64
```

---

## 5. Phase C — transfer artifacts to the Pi

```bash
ssh sarvar@rpizero.local "mkdir -p ~/axon_demo ~/axon_bench ~/burn_bench ~/axon_demo/python-tests/mnist"

# rusty-axon: bench binaries (pick the freshest non-.d ELF for each name)
for b in forward_train forward_infer_f32 forward_infer_i8 \
         training_step finetune_step matmul_kernel; do
  BIN=$(ls -1t target/aarch64-unknown-linux-gnu/release-edge/deps/${b}-* \
        | grep -v '\.d$' | head -1)
  scp "$BIN" sarvar@rpizero.local:~/axon_bench/
done

# rusty-axon: inference example, demos, pretrained model, MNIST CSVs
scp target/aarch64-unknown-linux-gnu/release-edge/examples/rpi_inference_f32 \
    target/aarch64-unknown-linux-gnu/release-edge/examples/rpi_inference_i8 \
    target/aarch64-unknown-linux-gnu/release-edge/examples/rpi_finetune_mnist \
    target/aarch64-unknown-linux-gnu/release-edge/examples/rpi_sensor_drift \
    mnist_pretrained.axn \
    sarvar@rpizero.local:~/axon_demo/

scp python-tests/mnist/mnist_train.csv \
    python-tests/mnist/mnist_test.csv \
    python-tests/mnist/mnist_personalize_*.csv \
    sarvar@rpizero.local:~/axon_demo/python-tests/mnist/

# Bench harness needs MNIST too (training_step / finetune_step load it)
ssh sarvar@rpizero.local "ln -sf ~/axon_demo/python-tests ~/axon_bench/python-tests"

# Burn: bench ELFs + min_inference
for b in forward_one infer_into_buf train_step_batch32; do
  BIN=$(ls -1t scripts/compare_burn/target/aarch64-unknown-linux-gnu/release-edge/deps/${b}-* \
        | grep -v '\.d$' | head -1)
  scp "$BIN" sarvar@rpizero.local:~/burn_bench/
done
scp scripts/compare_burn/target/aarch64-unknown-linux-gnu/release-edge/min_inference \
    sarvar@rpizero.local:~/burn_bench/

# TFLM: single self-contained binary
scp scripts/compare_tflite_micro/tflm_mnist_aarch64 sarvar@rpizero.local:~/
```

---

## 6. Phase D — run on the Pi

SSH to the Pi for everything below. Capture every stdout into a log file
under `~/results/`; you'll archive these as `docs/measurements/*.log`
afterward.

```bash
ssh sarvar@rpizero.local
mkdir -p ~/results
```

### D1. rusty-axon criterion benches

Each criterion bench takes 30 s – 3 min on the Pi. Total ~15 min if
nothing surprises you.

```bash
cd ~/axon_bench
for b in forward_train forward_infer_f32 forward_infer_i8 \
         training_step finetune_step matmul_kernel; do
  BIN=$(ls -1t ${b}-* 2>/dev/null | grep -v '\.d$' | head -1)
  [ -z "$BIN" ] && { echo "!! no ELF for $b"; continue; }
  echo "================ $b ================"
  ./$BIN --bench 2>&1 | tee ~/results/${b}_full.log
done
```

The `time:` line in each log is the cell value. Median is the middle of
the three numbers in `[lo med hi]`.

### D2. Burn criterion benches

```bash
cd ~/burn_bench
for b in forward_one infer_into_buf train_step_batch32; do
  BIN=$(ls -1t ${b}-* 2>/dev/null | grep -v '\.d$' | head -1)
  echo "================ $b ================"
  ./$BIN --bench 2>&1 | tee ~/results/burn_${b}_full.log
done
```

### D3. TFLite Micro inference benchmark

```bash
~/tflm_mnist_aarch64 bench 5000 2>&1 | tee ~/results/tflm_bench.log
# Reports: tflm_mnist bench iters=5000 total_us=... mean_us=... peak_rss_kb=...
```

`mean_us` → Table 1/2 TFLM cell. `peak_rss_kb` → Table 5 TFLM row.

### D4. RSS measurements

```bash
# rusty-axon (uses sysinfo internally; pass any .axn model)
~/axon_demo/rpi_inference_f32 ~/axon_demo/mnist_pretrained.axn \
    2>&1 | tee ~/results/rss_axon.log

# Burn (uses /usr/bin/time -v for peak RSS)
/usr/bin/time -v ~/burn_bench/min_inference \
    2>&1 | tee ~/results/rss_burn.log
```

### D5. Phase 11 demos

```bash
cd ~/axon_demo
./rpi_finetune_mnist 2>&1 | tee ~/results/rpi_finetune_mnist.log
./rpi_sensor_drift  2>&1 | tee ~/results/rpi_sensor_drift.log
```

The fine-tune demo takes ~2 min; sensor-drift takes <1 min.

### D6. Binary sizes (Table 4)

```bash
ls -l ~/axon_demo/rpi_inference_f32 \
       ~/axon_demo/rpi_inference_i8 \
       ~/burn_bench/min_inference \
       ~/tflm_mnist_aarch64 \
    | tee ~/results/binary_sizes.log
```

### D7. Bundle and ship the logs back

```bash
tar czf ~/all_results.tgz -C ~ results/
exit                                # back to host
scp sarvar@rpizero.local:~/all_results.tgz .
tar xzf all_results.tgz
```

---

## 7. Phase E — populate the doc

### E1. Save raw logs

Move/copy each `~/results/*.log` into [`docs/measurements/`](measurements/):

| Log on Pi                 | Lands at                                             |
|---------------------------|------------------------------------------------------|
| `rss_axon.log`            | `docs/measurements/rpi_inference_axon_rss.log`       |
| `rss_burn.log`            | `docs/measurements/rpi_inference_burn_rss.log`       |
| `rpi_finetune_mnist.log`  | `docs/measurements/rpi_finetune_mnist.log`           |
| `rpi_sensor_drift.log`    | `docs/measurements/rpi_sensor_drift.log`             |
| `tflm_bench.log`          | `docs/measurements/tflm_bench.log`                   |
| `binary_sizes.log`        | `docs/measurements/binary_sizes.log`                 |

Bench logs (`forward_train_full.log`, etc.) don't need to be re-checked
in — criterion already preserves them as JSON under each harness's
`target/criterion/<id>/new/estimates.json` (the `Median.point_estimate`
field, in nanoseconds). If you want stdout backups, drop them next to
the others.

### E2. Update each cell

Open [`COMPARISON.md`](COMPARISON.md) and walk the per-cell map below.
Bold the new value to mark it as measured (vs `*pending*`).

---

## 8. Per-cell reproduction map

| Cell                                            | Source                                                               |
|-------------------------------------------------|----------------------------------------------------------------------|
| Table 1 / Table 2 — rusty-axon fused fwd        | `forward_train_full.log` `time:` median                              |
| Table 1 / Table 2 — rusty-axon infer f32        | `forward_infer_f32_full.log` `time:` median                          |
| Table 1 / Table 2 — rusty-axon infer i8         | `forward_infer_i8_full.log` `time:` median                           |
| Table 1 — Burn (NdArray)                        | `burn_forward_one_full.log` `time:` median                           |
| Table 2 — Burn (NdArray)                        | `burn_infer_into_buf_full.log` `time:` median                        |
| Table 1 / Table 2 — TFLite Micro                | `tflm_bench.log` `mean_us`                                           |
| Table 3 — rusty-axon SGD                        | `training_step_full.log` `time:` median (ms)                         |
| Table 3 — Burn (NdArray)                        | `burn_train_step_batch32_full.log` `time:` median (ms)               |
| Table 4 — Combo E (axon inference, aarch64)     | `binary_sizes.log` line for `rpi_inference_f32`                      |
| Table 4 — Combo F (axon inference+i8, aarch64)  | `binary_sizes.log` line for `rpi_inference_i8`                       |
| Table 4 — Combo H (Burn aarch64)                | `binary_sizes.log` line for `min_inference`                          |
| Table 4 — Combo J (TFLM aarch64)                | `binary_sizes.log` line for `tflm_mnist_aarch64`                     |
| Table 5 — rusty-axon RSS                        | `rpi_inference_axon_rss.log` `rss: ... after`                        |
| Table 5 — Burn RSS                              | `rpi_inference_burn_rss.log` `Maximum resident set size (kbytes)`    |
| Table 5 — TFLite Micro RSS                      | `tflm_bench.log` `peak_rss_kb`                                       |
| Table 6 — rusty-axon (training RSS)             | `rpi_finetune_mnist.log` `rss_end`                                   |
| Table 7 — TFLite Micro f32 accuracy             | `python-tests/mnist_mlp_metadata.json` `test_accuracy`               |
| Table 7 — rusty-axon f32 accuracy               | `mnist_personalize_pretrain` final `test_acc` line (host run)        |
| Table 8 — MNIST last-layer fine-tune (axon)     | `rpi_finetune_mnist.log` `step_median`                               |
| Table 8 — Sensor-drift fine-tune (axon)         | `rpi_sensor_drift.log` `adapt_s` per slice ÷ 200 steps               |
| Table 10 — t1/t2/t3 MSE before/after            | `rpi_sensor_drift.log` `mse_before` / `mse_after`                    |
| Table 11 — full personalization demo            | `rpi_finetune_mnist.log` (every line; one cell per metric)           |

---

## 9. Known gotchas

### 9.1 Don't build TFLM natively on the Pi

TFLM's microlite plus the harness pulls in ~150 MB of C++ that `cc1plus`
will OOM-kill on a 512 MB Pi Zero 2 W (even with `-j1` it's ~45 min
under heavy swap thrash). Cross-compile from WSL instead — instructions
in §4 step B5.

### 9.2 Verify the `microlite` archive is genuinely aarch64

`file libtensorflow-microlite.a` only reports "current ar archive" — it
doesn't tell you what arch the **objects inside** were built for. Check
with:

```bash
aarch64-linux-gnu-objdump -a tflite-micro/gen/linux_aarch64_default_gcc/lib/libtensorflow-microlite.a \
    | grep 'file format' | head -3
```

If you see `elf64-x86-64`, the static lib was built with the host
compiler. `rm -rf gen/` and rerun the `make microlite` command from §4
B5 with `CC_TOOL` / `CXX_TOOL` / `AR_TOOL` set explicitly (just
`TARGET_TOOLCHAIN_PREFIX` is not always honored).

### 9.3 TFLM `TF_LITE_STATIC_MEMORY` ABI skew

If the TFLM harness reports

```
[diag] input  type=0 bytes=3136 dims=-1
[diag] output type=0 bytes=40 dims=-1
expected f32 IO; got input=0 output=0
```

…then segfaults on `Invoke`, the harness's `TfLiteTensor` struct
layout doesn't match the lib's. TFLM's microlite is built with
`-DTF_LITE_STATIC_MEMORY` which **reorders** `TfLiteTensor` fields
("largest-to-smallest" for the slim variant). The harness Makefile
must set the same define. The repo's
`scripts/compare_tflite_micro/Makefile` already does:

```
TFLM_DEFINES := -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -DTF_LITE_USE_CTIME
```

If you fork this build, preserve those defines and the
`-fno-unwind-tables -fno-asynchronous-unwind-tables -fmessage-length=0`
flags in `TFLM_COMMON`. See
[`tensorflow/lite/core/c/common.h`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/core/c/common.h)
for the conditional struct layout.

### 9.4 `cross` caches across feature-flag changes

If you cross-build `rpi_inference` with `--features inference,matrixmultiply`,
then immediately rebuild with `--features inference,matrixmultiply,quant-i8`
and the binary is **byte-identical**, the second build was cached. Force
a re-link:

```bash
rm target/aarch64-unknown-linux-gnu/release-edge/examples/rpi_inference
cross build ...                     # this time it actually re-runs the linker
```

(The host-side `binary_sizes.csv` shows a 2 KiB delta between f32 and
i8 builds; on aarch64 the strip-after-LTO can bring it to zero — which
is fine to record as long as you've verified the rebuild ran.)

### 9.5 `./bench-*` glob matches multiple ELFs and `.d` files

```bash
./forward_train-* --bench --quick
# bash expands the glob: ELF1 ELF2 forward_train-XYZ.d --bench --quick
# the first ELF sees its sibling as argv[1] -> "unexpected argument found"
```

Always pick exactly one ELF first:

```bash
BIN=$(ls -1t forward_train-* | grep -v '\.d$' | head -1)
./$BIN --bench
```

### 9.6 `tee` over SSH appears to swallow criterion output

`tee` is line-buffered when stdout is a tty, fully-buffered when piped.
Over SSH the latter kicks in and you see "Gnuplot not found" then nothing
for 30 s, then a flood at the end. Either accept the silence (the data
arrives), or force line buffering:

```bash
./$BIN --bench 2>&1 | stdbuf -oL -eL tee out.log
```

### 9.7 `bash` multi-line paste leaves residual state

If a previous multi-line paste fails to parse (e.g. a `(` inside an
unquoted string), the shell can hold half-tokens that corrupt the next
command. If you see `syntax error near unexpected token \`done\``,
type `exec bash` to get a fresh shell.

### 9.8 `/usr/bin/time` is not on Pi OS by default

```bash
sudo apt install -y time            # not the bash builtin; this is GNU time
/usr/bin/time -v ~/burn_bench/min_inference
```

The bash builtin `time` doesn't print RSS; you need GNU `time -v` for
the `Maximum resident set size` line.

---

## 10. Pinned versions (for byte-stable rebuilds)

| Component         | Version        | Where pinned                                    |
|-------------------|----------------|-------------------------------------------------|
| Rust toolchain    | per `rust-toolchain.toml`        | repo root                            |
| `matrixmultiply`  | `=0.3.9`       | `Cargo.toml`                                    |
| `criterion`       | `=0.5.1`       | `Cargo.toml`, `scripts/compare_burn/Cargo.toml` |
| `burn`            | `=0.16.0`      | `scripts/compare_burn/Cargo.toml`               |
| `burn-ndarray`    | `=0.16.0`      | `scripts/compare_burn/Cargo.toml`               |
| TensorFlow (host) | 2.21.0 tested  | not pinned — any 2.16+ should export the same `.tflite` |
| TFLite Micro      | `main` HEAD as of clone | shallow clone in `scripts/compare_tflite_micro/tflite-micro/` |
| `cross`           | latest         | install with `cargo install cross --locked`     |

The TFLM tree is intentionally **not** vendored — it's a 100 MB+ dep
that refreshes frequently. Pin the commit yourself if you need
absolute byte-stability across rebuilds (`cd tflite-micro && git rev-parse HEAD`).

---

## 11. What's intentionally not reproducible here

- **Host x86_64 latency rows.** Dropped from `COMPARISON.md` — the paper
  positions rusty-axon as edge-first and the workstation row was a
  distraction. `binary_sizes.csv` retains host sizes as a side artifact.
- **Burn fine-tune cells in Tables 6 and 8.** No Burn fine-tune harness
  exists yet; would require a Burn equivalent of `rpi_finetune_mnist`.
- **PTQ accuracy delta (Table 9).** No `quant_eval` example exists. A
  follow-up release would add one that loads `mnist_pretrained.axn`,
  evaluates f32 accuracy, quantizes, and re-evaluates.
- **rusty-axon MNIST accuracy at convergence.** The current 49% Table 7
  cell is a deliberately-short pretrain for the personalization demo's
  sake. To reproduce a converged rusty-axon accuracy cell, set
  `PRETRAIN_EPOCHS=80` (or higher) and re-run §3 step A4 — this is
  a multi-hour run on host.
