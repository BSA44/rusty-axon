# Phase 11: end-to-end driver for the paper-artifact bundle (Windows).
#
# Mirror of `scripts/run_paper_artifacts.sh`.  See that file for the full
# rationale.  Walks the same reproduction sequence on the host:
#   1. Prepare MNIST + persona + sensor datasets (Python).
#   2. Pretrain the personalization base model and the sensor calibration
#      model.
#   3. Run the on-device demos (host build, then optionally cross-compile
#      for the Pi if -Rpi is passed and the cross-toolchain is available).
#   4. Refresh the binary-size and benchmark tables.
#   5. (-Compare) Run the Phase-K Burn + TFLite Micro comparison harnesses
#      so docs/COMPARISON.md can be populated with cross-framework numbers.
#
# Each step's stdout is mirrored to docs/artifacts/<step>.log so the paper's
# tables can cite specific log lines.  Failures abort with a non-zero exit
# so CI can pick them up.
#
# Usage:
#   pwsh scripts/run_paper_artifacts.ps1
#   pwsh scripts/run_paper_artifacts.ps1 -Rpi
#   pwsh scripts/run_paper_artifacts.ps1 -SkipPretrain -SkipBench
#   pwsh scripts/run_paper_artifacts.ps1 -Compare

[CmdletBinding()]
param(
    [switch]$Rpi,
    [switch]$SkipPretrain,
    [switch]$SkipBench,
    [switch]$Compare
)

$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")
$Root = (Get-Location).Path
$Art  = Join-Path $Root "docs/artifacts"
New-Item -ItemType Directory -Force -Path $Art | Out-Null

function Log-Step([string]$msg) {
    Write-Host ""
    Write-Host "=== $msg ==="
}

function Invoke-Stage([string]$logPath, [scriptblock]$action) {
    & $action 2>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# 1. Datasets ----------------------------------------------------------------
Log-Step "1. prepare datasets"
if (-not (Test-Path "python-tests/mnist/mnist_train.csv")) {
    Invoke-Stage (Join-Path $Art "01_prepare_mnist.log") {
        python python-tests/prepare_mnist.py
    }
} else {
    Write-Host "  mnist_train.csv already present, skipping prepare_mnist.py"
}
if (-not (Test-Path "python-tests/mnist/mnist_personalize_train.csv")) {
    Invoke-Stage (Join-Path $Art "02_personalize.log") {
        python python-tests/generate_personalize_data.py
    }
} else {
    Write-Host "  personalize CSVs already present, skipping generate_personalize_data.py"
}
if (-not (Test-Path "python-tests/sensor/sensor_train.csv")) {
    Invoke-Stage (Join-Path $Art "03_sensor.log") {
        python python-tests/generate_sensor_drift.py
    }
} else {
    Write-Host "  sensor CSVs already present, skipping generate_sensor_drift.py"
}

# 2. Pretrain ----------------------------------------------------------------
Log-Step "2. pretrain base models"
if (-not $SkipPretrain) {
    if (-not (Test-Path "mnist_pretrained.axn")) {
        Invoke-Stage (Join-Path $Art "04_pretrain_mnist.log") {
            cargo run --release --example mnist_personalize_pretrain
        }
    } else {
        Write-Host "  mnist_pretrained.axn already present, skipping mnist pretrain"
    }
    if (-not (Test-Path "sensor_initial.axn")) {
        # The sensor demo pretrains the network on first invocation when the
        # .axn doesn't yet exist; pass ADAPT_STEPS=0 to suppress the adapt
        # cycle so this stage is purely the pretrain.
        $prevAdapt = $env:ADAPT_STEPS
        $env:ADAPT_STEPS = "0"
        try {
            Invoke-Stage (Join-Path $Art "05_pretrain_sensor.log") {
                cargo run --release --example rpi_sensor_drift
            }
        } finally {
            $env:ADAPT_STEPS = $prevAdapt
        }
    } else {
        Write-Host "  sensor_initial.axn already present, skipping sensor pretrain"
    }
} else {
    Write-Host "  -SkipPretrain set, skipping pretraining"
}

# 3. On-device demos (host build) -------------------------------------------
Log-Step "3. run demos on host"
Invoke-Stage (Join-Path $Art "06_finetune_mnist.log") {
    cargo run --release --example rpi_finetune_mnist
}
Invoke-Stage (Join-Path $Art "07_sensor_drift.log") {
    cargo run --release --example rpi_sensor_drift
}

# 4. Cross-compile for Pi Zero 2 W ------------------------------------------
if ($Rpi) {
    Log-Step "4. cross-compile for aarch64"
    Invoke-Stage (Join-Path $Art "08_build_rpi.log") {
        pwsh -File scripts/build_rpi.ps1
    }
}

# 5. Binary-size + bench tables ---------------------------------------------
if (-not $SkipBench) {
    Log-Step "5. refresh binary-size table"
    Invoke-Stage (Join-Path $Art "09_binary_size.log") {
        pwsh -File scripts/measure_binary_size.ps1
    }
}

# 6. Phase-K cross-framework comparison (Burn + TFLite Micro) ---------------
# Opt-in because both harnesses pull in heavy toolchains (Burn = 5+ min cold
# build, TFLM = git submodule + C compiler + tensorflow Python).  When the
# inputs are already prepared, this stage takes a few minutes total.
if ($Compare) {
    Log-Step "6a. Burn baseline (compare_burn)"
    Invoke-Stage (Join-Path $Art "10_compare_burn.log") {
        cargo bench --manifest-path scripts/compare_burn/Cargo.toml
    }

    Log-Step "6b. TFLite Micro: train + export + build + bench"
    if (-not (Test-Path "scripts/compare_tflite_micro/mnist_mlp_tflite.h")) {
        Invoke-Stage (Join-Path $Art "11_train_keras.log") {
            python python-tests/train_keras_mnist.py
        }
    } else {
        Write-Host "  mnist_mlp_tflite.h already present, skipping Keras export"
    }
    if (-not (Test-Path "scripts/compare_tflite_micro/tflite-micro")) {
        Write-Warning "  tflite-micro submodule missing under scripts/compare_tflite_micro/"
        Write-Warning "  see scripts/compare_tflite_micro/README.md for the clone+build steps"
    } else {
        Invoke-Stage (Join-Path $Art "12_build_tflm.log") {
            make -C scripts/compare_tflite_micro
        }
        Invoke-Stage (Join-Path $Art "13_bench_tflm.log") {
            ./scripts/compare_tflite_micro/tflm_mnist bench 5000
        }
    }
}

Log-Step "done"
Write-Host "  artifacts written to $Art"
