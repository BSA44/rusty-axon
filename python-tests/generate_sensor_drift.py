#!/usr/bin/env python3
"""
Generate a synthetic drifting-sensor regression dataset for the Phase 11
on-device adaptation demo.

Scenario: a 1-input -> 1-output regression that maps a "raw sensor reading"
to its "calibrated value". The ground-truth mapping is a smooth nonlinear
function; the *sensor* drifts over deployment time (offset + gain
distortion + small Gaussian noise), so the model trained at t=0 grows
progressively wrong unless re-calibrated against fresh samples.

Outputs at python-tests/sensor/ :
  sensor_train.csv     -- 800 (raw, calibrated) pairs from the t=0 sensor
  sensor_drift_t1.csv  -- 200 pairs after mild drift
  sensor_drift_t2.csv  -- 200 pairs after moderate drift
  sensor_drift_t3.csv  -- 200 pairs after heavy drift
  sensor_preview.png   -- raw vs calibrated for every drift stage

CSV format matches the rest of the project: one header row, then
  raw_reading, calibrated_value      (both float, no normalization)

Requires: numpy, matplotlib (only matplotlib is for the preview).
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# Latent ground-truth: calibrated = f(true_value).  This is what the model
# must learn from the *raw* sensor reading at t=0 and recover after drift.
def true_calibration(v):
    return 0.6 * v + 0.3 * np.sin(1.2 * v)


# Drift profile: at deployment time t the sensor reports
#   raw = (true_value + offset_t) * gain_t + noise
# so the calibration network's task is "raw -> calibrated".  As (offset, gain)
# drift, the same calibrated value comes from a different raw, breaking the
# initially-learned mapping.
DRIFT_STAGES = {
    "train":      {"offset": 0.00, "gain": 1.00, "noise": 0.02},
    "drift_t1":   {"offset": 0.30, "gain": 1.05, "noise": 0.03},
    "drift_t2":   {"offset": 0.65, "gain": 1.12, "noise": 0.04},
    "drift_t3":   {"offset": 1.00, "gain": 1.20, "noise": 0.05},
}

N_TRAIN = 800
N_DRIFT = 200
SEED    = 42
V_LO, V_HI = -3.0, 3.0   # range of latent true values


def sample_pairs(n, stage_cfg, rng):
    v = rng.uniform(V_LO, V_HI, size=n).astype(np.float32)
    raw = ((v + stage_cfg["offset"]) * stage_cfg["gain"]
           + rng.normal(0.0, stage_cfg["noise"], size=n).astype(np.float32))
    cal = true_calibration(v).astype(np.float32)
    return raw.astype(np.float32), cal.astype(np.float32)


def to_csv(raw, cal, path):
    data = np.column_stack([raw, cal])
    header = "raw_reading,calibrated_value"
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"  wrote {path} ({len(raw)} samples)")


def main():
    here    = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "sensor")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # Train set at t=0 (no drift yet).
    raw_tr, cal_tr = sample_pairs(N_TRAIN, DRIFT_STAGES["train"], rng)
    to_csv(raw_tr, cal_tr, os.path.join(out_dir, "sensor_train.csv"))

    # Drift stages.
    drift_data = {}
    for name in ("drift_t1", "drift_t2", "drift_t3"):
        raw_d, cal_d = sample_pairs(N_DRIFT, DRIFT_STAGES[name], rng)
        to_csv(raw_d, cal_d, os.path.join(out_dir, f"sensor_{name}.csv"))
        drift_data[name] = (raw_d, cal_d)

    # Preview: scatter raw vs calibrated for each stage so reviewers can
    # eyeball the drift curve.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(raw_tr, cal_tr, s=8, alpha=0.5, label="train (t=0)")
    for name, (raw_d, cal_d) in drift_data.items():
        ax.scatter(raw_d, cal_d, s=8, alpha=0.5, label=name)
    ax.set_xlabel("raw sensor reading")
    ax.set_ylabel("calibrated value")
    ax.set_title("sensor calibration target across drift stages")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    preview = os.path.join(out_dir, "sensor_preview.png")
    fig.savefig(preview, dpi=120)
    plt.close(fig)
    print(f"  wrote {preview}")

    print()
    print("Done.  Files for the Phase 11 sensor-drift demo:")
    for name in ("sensor_train.csv", "sensor_drift_t1.csv",
                 "sensor_drift_t2.csv", "sensor_drift_t3.csv"):
        print(f"  {os.path.join(out_dir, name)}")
    print(f"  {preview}")


if __name__ == "__main__":
    main()
