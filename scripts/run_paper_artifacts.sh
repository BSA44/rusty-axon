#!/usr/bin/env bash
# End-to-end driver for the Phase 11 paper-artifact bundle.
#
# Walks the full reproduction sequence on the host:
#   1. Prepare MNIST + persona + sensor datasets (Python).
#   2. Pretrain the personalization base model and the sensor calibration
#      model.
#   3. Run the on-device demos (host build, then optionally cross-compile
#      for the Pi if `--rpi` is passed and the cross-toolchain is available).
#   4. Refresh the binary-size and benchmark tables.
#
# Each step's stdout is mirrored to docs/artifacts/<step>.log so the paper's
# tables can cite specific log lines.  Failures abort with a non-zero exit
# so CI can pick them up.
#
# Usage:
#   bash scripts/run_paper_artifacts.sh [--rpi] [--skip-pretrain] [--skip-bench]

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
ART="$ROOT/docs/artifacts"
mkdir -p "$ART"

RPI=0
SKIP_PRETRAIN=0
SKIP_BENCH=0
for arg in "$@"; do
    case "$arg" in
        --rpi)            RPI=1 ;;
        --skip-pretrain)  SKIP_PRETRAIN=1 ;;
        --skip-bench)     SKIP_BENCH=1 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

log_step() { echo; echo "=== $1 ==="; }

# 1. Datasets ----------------------------------------------------------------
log_step "1. prepare datasets"
if [ ! -f python-tests/mnist/mnist_train.csv ]; then
    python python-tests/prepare_mnist.py | tee "$ART/01_prepare_mnist.log"
else
    echo "  mnist_train.csv already present, skipping prepare_mnist.py"
fi
if [ ! -f python-tests/mnist/mnist_personalize_train.csv ]; then
    python python-tests/generate_personalize_data.py | tee "$ART/02_personalize.log"
else
    echo "  personalize CSVs already present, skipping generate_personalize_data.py"
fi
if [ ! -f python-tests/sensor/sensor_train.csv ]; then
    python python-tests/generate_sensor_drift.py | tee "$ART/03_sensor.log"
else
    echo "  sensor CSVs already present, skipping generate_sensor_drift.py"
fi

# 2. Pretrain ----------------------------------------------------------------
log_step "2. pretrain base models"
if [ "$SKIP_PRETRAIN" -eq 0 ]; then
    if [ ! -f mnist_pretrained.axn ]; then
        cargo run --release --example mnist_personalize_pretrain \
            2>&1 | tee "$ART/04_pretrain_mnist.log"
    else
        echo "  mnist_pretrained.axn already present, skipping mnist pretrain"
    fi
    if [ ! -f sensor_initial.axn ]; then
        # The sensor demo pretrains the network on first invocation when the
        # .axn doesn't yet exist; pass ADAPT_STEPS=0 to suppress the adapt
        # cycle so this stage is purely the pretrain.
        ADAPT_STEPS=0 cargo run --release --example rpi_sensor_drift \
            2>&1 | tee "$ART/05_pretrain_sensor.log"
    else
        echo "  sensor_initial.axn already present, skipping sensor pretrain"
    fi
else
    echo "  --skip-pretrain set, skipping pretraining"
fi

# 3. On-device demos (host build) -------------------------------------------
log_step "3. run demos on host"
cargo run --release --example rpi_finetune_mnist 2>&1 \
    | tee "$ART/06_finetune_mnist.log"
cargo run --release --example rpi_sensor_drift 2>&1 \
    | tee "$ART/07_sensor_drift.log"

# 4. Cross-compile for Pi Zero 2 W ------------------------------------------
if [ "$RPI" -eq 1 ]; then
    log_step "4. cross-compile for aarch64"
    bash scripts/build_rpi.sh 2>&1 | tee "$ART/08_build_rpi.log"
fi

# 5. Binary-size + bench tables ---------------------------------------------
if [ "$SKIP_BENCH" -eq 0 ]; then
    log_step "5. refresh binary-size table"
    bash scripts/measure_binary_size.sh 2>&1 | tee "$ART/09_binary_size.log"
fi

log_step "done"
echo "  artifacts written to $ART"
