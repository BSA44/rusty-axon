# compare-tflite-micro

TFLite Micro inference baseline for the rusty-axon paper comparison.

## What this is

A small C/C++ harness that:

1. Embeds the `.tflite` model produced by
   [`python-tests/train_keras_mnist.py`](../../python-tests/train_keras_mnist.py)
   (architecture **784 → 640 → 320 → 100 → 10**, ReLU/ReLU/ReLU/None —
   same as `benches/common/mod.rs`, `examples/mnist_classifier.rs`, and
   the Burn harness).
2. Links against the official **TFLite Micro** runtime (statically) and
   runs single-sample inference under the deterministic LCG-generated
   input that the rusty-axon and Burn benches also use.
3. Reports either one `argmax` (smoke mode) or a timed loop
   (`bench <iters>`) plus peak RSS via `getrusage`.

It deliberately avoids any framework-specific niceties so the comparison
isolates the runtime cost.

## Why C, not Rust

TFLite Micro is C++; the official build flow is `make`. Wrapping it in a
Rust binding would add bindgen overhead and introduce a confound (the
binding's overhead vs the runtime's). The C harness is the same shape as
what an embedded developer would actually ship.

## Prerequisites

* A C/C++ toolchain (`gcc`/`clang` for host; `aarch64-linux-gnu-g++` for Pi).
* Python 3 with `tensorflow==2.16.*` for the model export step.
* The TFLite Micro source tree, checked out as a git submodule:

  ```sh
  cd scripts/compare_tflite_micro
  git clone --depth 1 https://github.com/tensorflow/tflite-micro.git
  make -C tflite-micro -f tensorflow/lite/micro/tools/make/Makefile microlite
  ```

  That last command builds the `libtensorflow-microlite.a` we link against.

## Build (host)

```sh
# 1. Train + export the .tflite (run once; produces mnist_mlp_tflite.h).
python ../../python-tests/train_keras_mnist.py

# 2. Build the harness, linking the prebuilt microlite static library.
make TFLM_DIR=tflite-micro
```

## Build (cross-compile for Pi Zero 2 W)

```sh
# Rebuild microlite for aarch64 first (separate gen/ subdir).
make -C tflite-micro -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=cortex_m_generic TARGET_ARCH=cortex-a53 microlite \
  || true   # the canonical cross flow uses the host gcc with -march=armv8

# Or, simpler: use the aarch64 cross-gcc directly, same Makefile:
make TFLM_DIR=tflite-micro \
     CC=aarch64-linux-gnu-gcc \
     CXX=aarch64-linux-gnu-g++ \
     AR=aarch64-linux-gnu-ar \
     target=tflm_mnist_aarch64
```

## Running

```sh
./tflm_mnist                      # smoke test: prints argmax + peak RSS
./tflm_mnist bench 5000           # mean us per Invoke over 5000 iterations
```

The two numbers — mean us/inference and peak RSS — drop into Tables 2
and 5 of [`docs/COMPARISON.md`](../../docs/COMPARISON.md).

## Compiler flags

`-Os -flto -ffunction-sections -fdata-sections -Wl,--gc-sections -s` —
chosen to mirror rusty-axon's `release-edge` Cargo profile (`opt-level =
"z"`, `lto = "fat"`, `strip = "symbols"`). C++ exceptions and RTTI are
disabled, matching TFLM's own build defaults. Document any deviations
in COMPARISON.md so reviewers can reproduce.

## Why training is "N/A" for TFLite Micro

TFLite Micro is inference-only. There is no on-device training path in
the upstream runtime. The "Single training step" and "Fine-tune
wall-clock" cells in the paper tables are explicitly `N/A — inference
only` for the TFLM column. This is the categorical difference the paper
is built around.
