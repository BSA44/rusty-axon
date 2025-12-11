#!/usr/bin/env python3
"""
Flexible benchmark comparison between Micrograd (Python) and Rusty-Axon (Rust).

Add or remove CSVs by editing the CONFIG_* lists below—no code changes needed.
All classification CSVs must share the same column schema; likewise for regression.
"""
import csv
import sys
import os
from typing import Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend


# ---- Configuration ---------------------------------------------------------
# Extend or trim these lists to change which CSVs are compared.
CLASSIFICATION_RUNS = [
    ("Python (Micrograd)", "python-tests/micrograd/classification-diabetes/training_metrics.csv"),
    ("Rust SGD", "python-tests/rusty-axon/classification-diabetes/rust_classification_metrics_sgd.csv"),
    ("Rust MeProp", "python-tests/rusty-axon/classification-diabetes/rust_classification_metrics_meprop.csv"),
    ("Rust SGD Rasp. Pi", "python-tests/rusty-axon-rpi/classification-diabetes/rust_classification_metrics_sgd.csv"),
    ("Rust MeProp Rasp. Pi", "python-tests/rusty-axon-rpi/classification-diabetes/rust_classification_metrics_meprop.csv"),
]

REGRESSION_RUNS = [
    ("Python (Micrograd)", "python-tests/micrograd/regression-california-housing/training_metrics.csv"),
    ("Rust SGD", "python-tests/rusty-axon/regression-california-housing/rust_regression_metrics_sgd.csv"),
    ("Rust MeProp", "python-tests/rusty-axon/regression-california-housing/rust_regression_metrics_meprop.csv"),
    ("Rust SGD Rasp. Pi", "python-tests/rusty-axon-rpi/regression-california-housing/rust_regression_metrics_sgd.csv"),
    ("Rust MeProp Rasp. Pi", "python-tests/rusty-axon-rpi/regression-california-housing/rust_regression_metrics_meprop.csv"),
]


# ---- Data helpers ----------------------------------------------------------
def load_csv(filepath: str) -> List[Dict[str, str]]:
    with open(filepath, "r") as f:
        return list(csv.DictReader(f))


def ensure_graphs_dir() -> None:
    os.makedirs("python-tests/graphs", exist_ok=True)


def load_runs(run_specs: Sequence[Tuple[str, str]]) -> Optional[List[Tuple[str, List[Dict[str, str]]]]]:
    runs = []
    for label, path in run_specs:
        if not os.path.exists(path):
            print(f"Error: missing file for '{label}': {path}")
            return None
        data = load_csv(path)
        if not data:
            print(f"Error: empty data for '{label}': {path}")
            return None
        runs.append((label, data))
    return runs


# ---- Reporting helpers -----------------------------------------------------
def print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def final_epoch_table(runs: Sequence[Tuple[str, List[Dict[str, str]]]], metrics: Sequence[Tuple[str, str]], epoch_label: str) -> None:
    headers = ["Metric"] + [label for label, _ in runs] + [f"Diff vs {runs[0][0]}"]
    print(f"\nFINAL EPOCH METRICS ({epoch_label})")
    print("-" * 100)
    print(" | ".join(f"{h:<20}" for h in headers))
    print("-" * 100)

    for name, key in metrics:
        baseline = float(runs[0][1][-1][key])
        row = [f"{name:<20}"]
        diffs = []
        for label, data in runs:
            val = float(data[-1][key])
            row.append(f"{val:<20.6f}")
            if label != runs[0][0]:
                diffs.append(val - baseline)
        if diffs:
            row.append(" | ".join(f"{d:+.6f}" for d in diffs))
        else:
            row.append("0.000000")
        print(" ".join(row))


def performance_table(
    runs: Sequence[Tuple[str, List[Dict[str, str]]]],
    time_key: str,
    cpu_key: str,
    ram_key: str,
) -> None:
    print("\nPERFORMANCE METRICS")
    print("-" * 100)

    totals = []
    avgs = []
    cpu = []
    ram = []
    for _, data in runs:
        total_time = sum(float(row[time_key]) for row in data)
        totals.append(total_time)
        avgs.append(total_time / len(data))
        cpu.append(sum(float(row[cpu_key]) for row in data) / len(data))
        ram.append(sum(float(row[ram_key]) for row in data) / len(data))

    baseline_total = totals[0]
    print("Total Training Time:")
    for (label, _), total in zip(runs, totals):
        speedup = baseline_total / total if total > 0 else 0
        print(f"  {label:<15} {total:.2f}s  (Speedup vs {runs[0][0]}: {speedup:.2f}x)")

    print("\nAverage Time per Epoch:")
    baseline_avg = avgs[0]
    for (label, _), avg in zip(runs, avgs):
        speedup = baseline_avg / avg if avg > 0 else 0
        print(f"  {label:<15} {avg:.2f}s  (Speedup vs {runs[0][0]}: {speedup:.2f}x)")

    print("\nAverage CPU Usage:")
    for (label, _), c in zip(runs, cpu):
        print(f"  {label:<15} {c:.1f}%")

    print("\nAverage RAM Usage:")
    for (label, _), r in zip(runs, ram):
        print(f"  {label:<15} {r:.1f}%")


# ---- Plotting --------------------------------------------------------------
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
COLORS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]


def plot_metric(
    runs: Sequence[Tuple[str, List[Dict[str, str]]]],
    metric_key: str,
    title: str,
    ylabel: str,
    outfile: str,
    x_key: str = "Epoch",
    scale: float = 1.0,
) -> None:
    plt.figure(figsize=(10, 6))
    for idx, (label, data) in enumerate(runs):
        epochs = [int(row[x_key]) for row in data]
        values = [float(row[metric_key]) * scale for row in data]
        marker = MARKERS[idx % len(MARKERS)]
        color = COLORS[idx % len(COLORS)]
        plt.plot(epochs, values, marker + "-", label=label, linewidth=2, markersize=5, color=color)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


# ---- Markdown generation ---------------------------------------------------
def markdown_table(runs: Sequence[Tuple[str, List[Dict[str, str]]]], metrics: Sequence[Tuple[str, str]], title: str) -> str:
    headers = ["Metric"] + [label for label, _ in runs] + [f"Diff vs {runs[0][0]}"]
    lines = [title, "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for name, key in metrics:
        baseline = float(runs[0][1][-1][key])
        row = [name]
        diffs = []
        for label, data in runs:
            val = float(data[-1][key])
            row.append(f"{val:.6f}")
            if label != runs[0][0]:
                diffs.append(val - baseline)
        row.append(" / ".join(f"{d:+.6f}" for d in diffs) if diffs else "0.000000")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def markdown_perf(runs: Sequence[Tuple[str, List[Dict[str, str]]]], time_key: str, cpu_key: str, ram_key: str) -> str:
    # Header layout: Baseline then each other run paired with its speedup column
    headers = ["Metric", runs[0][0]]
    for label, _ in runs[1:]:
        headers.extend([label, "Speedup"])
    lines = ["### Performance Metrics", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    totals = [sum(float(row[time_key]) for row in data) for _, data in runs]
    avgs = [t / len(data) for t, (_, data) in zip(totals, runs)]
    cpu = [sum(float(row[cpu_key]) for row in data) / len(data) for _, data in runs]
    ram = [sum(float(row[ram_key]) for row in data) / len(data) for _, data in runs]
    baseline = totals[0]
    baseline_avg = avgs[0]

    def row_with_speedups(values: Sequence[float], metric_fmt: str, baseline_value: float) -> str:
        cells = [metric_fmt.format(values[0])]
        for val in values[1:]:
            speed = baseline_value / val if val else 0
            cells.extend([metric_fmt.format(val), f"{speed:.2f}x"])
        return "| " + " | ".join(cells) + " |"

    lines.append("| Total Training Time (s) | " + row_with_speedups(totals, "{:.2f}", baseline).strip("| ") )
    lines.append("| Avg Time per Epoch (s) | " + row_with_speedups(avgs, "{:.2f}", baseline_avg).strip("| ") )

    # CPU/RAM: keep speedup cells but mark as "-"
    cpu_cells = [f"{cpu[0]:.1f}"] + [item for val in cpu[1:] for item in (f"{val:.1f}", "-")]
    ram_cells = [f"{ram[0]:.1f}"] + [item for val in ram[1:] for item in (f"{val:.1f}", "-")]
    lines.append("| Avg CPU Usage (%) | " + " | ".join(cpu_cells) + " |")
    lines.append("| Avg RAM Usage (%) | " + " | ".join(ram_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def save_summary_to_markdown(
    classification_runs: Optional[Sequence[Tuple[str, List[Dict[str, str]]]]] = None,
    regression_runs: Optional[Sequence[Tuple[str, List[Dict[str, str]]]]] = None,
) -> None:
    ensure_graphs_dir()
    with open("python-tests/graphs/benchmark_summary.md", "w") as f:
        f.write("# Benchmark Comparison: Micrograd (Python) vs Rusty-Axon (Rust)\n\n")
        f.write(f"*Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("---\n\n")

        if classification_runs:
            f.write("## Classification Benchmark (Diabetes/Pima Indians)\n\n")
            metrics = [
                ("Train Loss", "Train_Loss"),
                ("Train Accuracy", "Train_Acc"),
                ("Test Loss", "Test_Loss"),
                ("Test Accuracy", "Test_Acc"),
                ("F1 Score", "F1"),
            ]
            f.write(markdown_table(classification_runs, metrics, "### Final Epoch Metrics\n"))
            f.write(markdown_perf(classification_runs, "Epoch_Time", "CPU_Usage", "RAM_Usage"))
            f.write("\n### Generated Plots\n\n")
            f.write("- `classification_train_loss.png`\n")
            f.write("- `classification_test_loss.png`\n")
            f.write("- `classification_train_accuracy.png`\n")
            f.write("- `classification_test_accuracy.png`\n")
            f.write("- `classification_f1_score.png`\n")
            f.write("- `classification_cpu_usage.png`\n")
            f.write("- `classification_ram_usage.png`\n")
            f.write("- `classification_epoch_time.png`\n")
            f.write("\n---\n\n")

        if regression_runs:
            f.write("## Regression Benchmark (California Housing)\n\n")
            metrics = [
                ("Loss (MSE)", "Loss"),
                ("RMSE", "RMSE"),
            ]
            f.write(markdown_table(regression_runs, metrics, "### Final Epoch Metrics\n"))
            f.write(markdown_perf(regression_runs, "Time_s", "CPU_Usage", "RAM_Usage"))
            f.write("\n### Generated Plots\n\n")
            f.write("- `regression_loss.png`\n")
            f.write("- `regression_rmse.png`\n")
            f.write("- `regression_cpu_usage.png`\n")
            f.write("- `regression_ram_usage.png`\n")
            f.write("- `regression_epoch_time.png`\n")
            f.write("\n---\n\n")


# ---- Benchmark flows -------------------------------------------------------
def compare_classification() -> Optional[Sequence[Tuple[str, List[Dict[str, str]]]]]:
    print_header("CLASSIFICATION BENCHMARK COMPARISON (Diabetes/Pima Indians)")
    runs = load_runs(CLASSIFICATION_RUNS)
    if not runs:
        print("\nPlease ensure all classification CSVs exist. Add/remove files via CLASSIFICATION_RUNS.")
        return None

    metrics = [
        ("Train Loss", "Train_Loss"),
        ("Train Accuracy", "Train_Acc"),
        ("Test Loss", "Test_Loss"),
        ("Test Accuracy", "Test_Acc"),
        ("F1 Score", "F1"),
    ]
    final_epoch_table(runs, metrics, "Epoch 50")
    performance_table(runs, "Epoch_Time", "CPU_Usage", "RAM_Usage")

    print("\nGENERATING COMPARISON PLOTS...")
    ensure_graphs_dir()
    plot_metric(runs, "Train_Loss", "Classification: Train Loss Comparison", "Train Loss", "python-tests/graphs/classification_train_loss.png")
    plot_metric(runs, "Test_Loss", "Classification: Test Loss Comparison", "Test Loss", "python-tests/graphs/classification_test_loss.png")
    plot_metric(runs, "Train_Acc", "Classification: Train Accuracy Comparison", "Train Accuracy (%)", "python-tests/graphs/classification_train_accuracy.png", scale=100.0)
    plot_metric(runs, "Test_Acc", "Classification: Test Accuracy Comparison", "Test Accuracy (%)", "python-tests/graphs/classification_test_accuracy.png", scale=100.0)
    plot_metric(runs, "F1", "Classification: F1 Score Comparison", "F1 Score", "python-tests/graphs/classification_f1_score.png")
    plot_metric(runs, "CPU_Usage", "Classification: CPU Usage Comparison", "CPU Usage (%)", "python-tests/graphs/classification_cpu_usage.png")
    plot_metric(runs, "RAM_Usage", "Classification: RAM Usage Comparison", "RAM Usage (%)", "python-tests/graphs/classification_ram_usage.png")
    plot_metric(runs, "Epoch_Time", "Classification: Epoch Time Comparison", "Time (seconds)", "python-tests/graphs/classification_epoch_time.png")
    print()
    return runs


def compare_regression() -> Optional[Sequence[Tuple[str, List[Dict[str, str]]]]]:
    print_header("REGRESSION BENCHMARK COMPARISON (California Housing)")
    runs = load_runs(REGRESSION_RUNS)
    if not runs:
        print("\nPlease ensure all regression CSVs exist. Add/remove files via REGRESSION_RUNS.")
        return None

    metrics = [
        ("Loss (MSE)", "Loss"),
        ("RMSE", "RMSE"),
    ]
    final_epoch_table(runs, metrics, "Epoch 5")
    performance_table(runs, "Time_s", "CPU_Usage", "RAM_Usage")

    print("\nGENERATING COMPARISON PLOTS...")
    ensure_graphs_dir()
    plot_metric(runs, "Loss", "Regression: Loss (MSE) Comparison", "Loss (MSE)", "python-tests/graphs/regression_loss.png")
    plot_metric(runs, "RMSE", "Regression: RMSE Comparison", "RMSE", "python-tests/graphs/regression_rmse.png")
    plot_metric(runs, "CPU_Usage", "Regression: CPU Usage Comparison", "CPU Usage (%)", "python-tests/graphs/regression_cpu_usage.png")
    plot_metric(runs, "RAM_Usage", "Regression: RAM Usage Comparison", "RAM Usage (%)", "python-tests/graphs/regression_ram_usage.png")
    plot_metric(runs, "Time_s", "Regression: Epoch Time Comparison", "Time (seconds)", "python-tests/graphs/regression_epoch_time.png")
    print()
    return runs


# ---- Entry point -----------------------------------------------------------
def main() -> None:
    classification_runs = None
    regression_runs = None

    if len(sys.argv) > 1:
        benchmark = sys.argv[1].lower()
        if benchmark == "classification":
            classification_runs = compare_classification()
        elif benchmark == "regression":
            regression_runs = compare_regression()
        else:
            print(f"Unknown benchmark: {benchmark}")
            print("Usage: python compare_results.py [classification|regression]")
            return
    else:
        classification_runs = compare_classification()
        print()
        regression_runs = compare_regression()


    if classification_runs or regression_runs:
        print("\n📝 Saving summary to markdown...")
        save_summary_to_markdown(classification_runs, regression_runs)
        print("  Saved: python-tests/graphs/benchmark_summary.md")
        print("=" * 80)


if __name__ == "__main__":
    main()

