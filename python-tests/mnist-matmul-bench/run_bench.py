#!/usr/bin/env python3
"""
Compare matmul kernels (matrixmultiply vs naive) for the rusty-axon MNIST
training example.

Runs the `bench_mnist_matmul` example N times per kernel, captures
per-epoch and total wall-clock time from the binary's stdout, and samples
the subprocess CPU/RAM via psutil while it runs. Results land in
`mnist_matmul_bench.csv` next to this script.

Build is done once per feature set before timed runs so we never measure
compile time.

Usage:
  cd <repo root>
  python python-tests/mnist-matmul-bench/run_bench.py
  python python-tests/mnist-matmul-bench/run_bench.py --runs 5 --epochs 2

Requires: psutil (`pip install psutil`).
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
MNIST_TRAIN = REPO_ROOT / "python-tests" / "mnist" / "mnist_train.csv"
MNIST_TEST = REPO_ROOT / "python-tests" / "mnist" / "mnist_test.csv"
PREP_SCRIPT = REPO_ROOT / "python-tests" / "prepare_mnist.py"
OUT_CSV = SCRIPT_DIR / "mnist_matmul_bench.csv"

VARIANTS = [
    # (label, extra cargo args, expected KERNEL string in bench output)
    ("matrixmultiply", [], "matrixmultiply"),
    ("naive", ["--features", "naive-matmul"], "naive"),
]

EPOCH_RE = re.compile(r"\[BENCH\]\s+epoch=(\d+)\s+time_s=([0-9.eE+-]+)")
TOTAL_RE = re.compile(r"\[BENCH\]\s+total_time_s=([0-9.eE+-]+)")
ACC_RE = re.compile(r"\[BENCH\]\s+final_test_acc=([0-9.eE+-]+)")
KERNEL_RE = re.compile(r"\[BENCH\]\s+kernel=(\S+)")


def ensure_mnist():
    if MNIST_TRAIN.exists() and MNIST_TEST.exists():
        return
    print(f"[prep] MNIST CSVs not found, running {PREP_SCRIPT}")
    subprocess.run([sys.executable, str(PREP_SCRIPT)], check=True, cwd=REPO_ROOT)


def cargo_build(extra_args):
    cmd = ["cargo", "build", "--release"] + extra_args + ["--example", "bench_mnist_matmul"]
    print(f"[build] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def cargo_run_cmd(extra_args):
    return ["cargo", "run", "--release", "--quiet"] + extra_args + ["--example", "bench_mnist_matmul"]


class ResourceSampler(threading.Thread):
    """Sample subprocess + children CPU% and RSS at a fixed interval."""

    def __init__(self, pid, interval=0.1):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.stop_event = threading.Event()
        self.cpu_samples = []
        self.rss_samples = []
        self.num_cpus = psutil.cpu_count(logical=True) or 1

    def run(self):
        try:
            proc = psutil.Process(self.pid)
            # Prime cpu_percent so the next call returns a real delta.
            proc.cpu_percent(interval=None)
            for child in proc.children(recursive=True):
                try:
                    child.cpu_percent(interval=None)
                except psutil.Error:
                    pass
        except psutil.Error:
            return

        while not self.stop_event.is_set():
            time.sleep(self.interval)
            try:
                proc = psutil.Process(self.pid)
                cpu = proc.cpu_percent(interval=None)
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        cpu += child.cpu_percent(interval=None)
                        rss += child.memory_info().rss
                    except psutil.Error:
                        pass
                # Normalize to percent of one core (psutil already does that
                # per-process); divide by core count to get a system-wide
                # fraction comparable to other CPU% reports in this repo.
                self.cpu_samples.append(cpu / self.num_cpus)
                self.rss_samples.append(rss)
            except psutil.Error:
                break

    def stop(self):
        self.stop_event.set()


def parse_bench_output(stdout):
    epochs = []
    total = None
    acc = None
    kernel = None
    for line in stdout.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            epochs.append((int(m.group(1)), float(m.group(2))))
            continue
        m = TOTAL_RE.search(line)
        if m:
            total = float(m.group(1))
            continue
        m = ACC_RE.search(line)
        if m:
            acc = float(m.group(1))
            continue
        m = KERNEL_RE.search(line)
        if m:
            kernel = m.group(1)
    return kernel, epochs, total, acc


def run_once(extra_args, env_overrides):
    cmd = cargo_run_cmd(extra_args)
    env = os.environ.copy()
    env.update(env_overrides)

    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    sampler = ResourceSampler(proc.pid, interval=0.1)
    sampler.start()

    stdout, stderr = proc.communicate()

    sampler.stop()
    sampler.join(timeout=2.0)

    if proc.returncode != 0:
        sys.stderr.write(stdout)
        sys.stderr.write(stderr)
        raise RuntimeError(f"bench_mnist_matmul exited with code {proc.returncode}")

    return stdout, sampler.cpu_samples, sampler.rss_samples


def summarize(values, default=0.0):
    if not values:
        return default, default
    return sum(values) / len(values), max(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="runs per kernel (default 3)")
    parser.add_argument("--epochs", type=int, default=3, help="epochs per run (default 3)")
    parser.add_argument("--batch", type=int, default=32, help="batch size (default 32)")
    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="cap training samples (0 = use all)",
    )
    args = parser.parse_args()

    ensure_mnist()

    env_overrides = {
        "BENCH_EPOCHS": str(args.epochs),
        "BENCH_BATCH": str(args.batch),
    }
    if args.train_limit > 0:
        env_overrides["BENCH_TRAIN_LIMIT"] = str(args.train_limit)

    # Build once per variant before timed runs.
    for label, extra, _ in VARIANTS:
        cargo_build(extra)

    rows = []
    for label, extra, expected_kernel in VARIANTS:
        for run_idx in range(1, args.runs + 1):
            print(f"[run] kernel={label} run={run_idx}/{args.runs}")
            stdout, cpu_samples, rss_samples = run_once(extra, env_overrides)
            kernel, epochs, total, acc = parse_bench_output(stdout)

            if kernel != expected_kernel:
                sys.stderr.write(stdout)
                raise RuntimeError(
                    f"expected kernel={expected_kernel}, got kernel={kernel}"
                )
            if total is None or acc is None or not epochs:
                sys.stderr.write(stdout)
                raise RuntimeError("bench output missing total_time/final_acc/epochs")

            epoch_times = [t for _, t in epochs]
            avg_epoch = sum(epoch_times) / len(epoch_times)
            avg_cpu, peak_cpu = summarize(cpu_samples)
            avg_rss, peak_rss = summarize(rss_samples)

            row = {
                "kernel": label,
                "run": run_idx,
                "epochs": args.epochs,
                "batch": args.batch,
                "total_time_s": f"{total:.6f}",
                "avg_epoch_time_s": f"{avg_epoch:.6f}",
                "epoch_times_s": ";".join(f"{t:.6f}" for t in epoch_times),
                "final_test_acc_pct": f"{acc:.4f}",
                "avg_cpu_pct": f"{avg_cpu:.2f}",
                "peak_cpu_pct": f"{peak_cpu:.2f}",
                "avg_ram_mb": f"{avg_rss / (1024 * 1024):.2f}",
                "peak_ram_mb": f"{peak_rss / (1024 * 1024):.2f}",
                "samples": len(cpu_samples),
            }
            rows.append(row)
            print(
                f"      total={total:.2f}s avg_epoch={avg_epoch:.2f}s "
                f"acc={acc:.2f}% avg_cpu={avg_cpu:.1f}% peak_ram={row['peak_ram_mb']}MB"
            )

    fieldnames = [
        "kernel",
        "run",
        "epochs",
        "batch",
        "total_time_s",
        "avg_epoch_time_s",
        "epoch_times_s",
        "final_test_acc_pct",
        "avg_cpu_pct",
        "peak_cpu_pct",
        "avg_ram_mb",
        "peak_ram_mb",
        "samples",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[done] wrote {OUT_CSV}")

    # Quick aggregate at the end so the user sees the speedup.
    by_kernel = {}
    for r in rows:
        by_kernel.setdefault(r["kernel"], []).append(float(r["total_time_s"]))
    print("\nKernel summary (mean total time across runs):")
    means = {}
    for k, ts in by_kernel.items():
        m = sum(ts) / len(ts)
        means[k] = m
        print(f"  {k:<16} {m:.2f}s  (n={len(ts)})")
    if "naive" in means and "matrixmultiply" in means and means["matrixmultiply"] > 0:
        speedup = means["naive"] / means["matrixmultiply"]
        print(f"\nmatrixmultiply speedup vs naive: {speedup:.2f}x")


if __name__ == "__main__":
    main()
