#!/usr/bin/env python3
"""
Comparison script for Micrograd (Python) vs Rusty-Axon (Rust) benchmarks
"""
import csv
import sys
import os
from typing import List, Dict, Tuple, Callable

# Define all possible sources; missing files are skipped automatically
source_specs = [
    ('Python (Micrograd)', 'python-tests/micrograd/classification-diabetes/training_metrics.csv'),
    ('Rust SGD', 'python-tests/rusty-axon/classification-diabetes/rust_classification_metrics_sgd.csv'),
    ('Rust MeProp', 'python-tests/rusty-axon/classification-diabetes/rust_classification_metrics_meprop.csv'),
    ('Rust SGD RPI', 'python-tests/rusty-axon/classification-diabetes/rust_classification_metrics_sgd_rpi.csv'),
    ('Rust MeProp RPI', 'python-tests/rusty-axon/classification-diabetes/rust_classification_metrics_meprop_rpi.csv'),
]

loaded = load_sources(source_specs)
if not loaded:
    print("Error: No data files found for classification. Please run the benchmarks first.")
    return None

# Convert to dict label->data for summary function
data_map = {label: data for label, data in loaded}

# Final epoch comparison
labels = list(data_map.keys())
print("\nFINAL EPOCH METRICS")
print("-" * 100)
header = f"{'Metric':<25} " + " ".join(f"{lbl:<20}" for lbl in labels)
print(header)
print("-" * 100)

metrics = [
    ('Train Loss', 'Train_Loss'),
    ('Train Accuracy', 'Train_Acc'),
    ('Test Loss', 'Test_Loss'),
    ('Test Accuracy', 'Test_Acc'),
    ('F1 Score', 'F1'),
]

baseline = labels[0]
for name, key in metrics:
    vals = []
    for lbl in labels:
        try:
            vals.append(float(data_map[lbl][-1].get(key, float('nan'))))
        except Exception:
            vals.append(float('nan'))
    line = f"{name:<25} " + " ".join(f"{v:<20.6f}" for v in vals)
    print(line)

# Performance comparison
print("\nPERFORMANCE METRICS")
print("-" * 100)
time_key = 'Epoch_Time'
cpu_key = 'CPU_Usage'
ram_key = 'RAM_Usage'

totals = {lbl: sum(float(r.get(time_key, 0)) for r in data_map[lbl]) for lbl in labels}
avgs = {lbl: {
    'avg_time': totals[lbl] / len(data_map[lbl]) if data_map[lbl] else 0,
    'avg_cpu': sum(float(r.get(cpu_key, 0)) for r in data_map[lbl]) / len(data_map[lbl]) if data_map[lbl] else 0,
    'avg_ram': sum(float(r.get(ram_key, 0)) for r in data_map[lbl]) / len(data_map[lbl]) if data_map[lbl] else 0,
} for lbl in labels}

for lbl in labels:
    print(f"  {lbl}: Total Time={totals[lbl]:.2f}s, Avg Epoch={avgs[lbl]['avg_time']:.2f}s, Avg CPU={avgs[lbl]['avg_cpu']:.1f}%, Avg RAM={avgs[lbl]['avg_ram']:.1f}%")

# Generate plots generically
print("\nGENERATING COMPARISON PLOTS...")
ensure_graphs_dir()

markers = ['o-', 's-', '^-', 'd-', 'x-']
def plot_metric(metric_key: str, ylabel: str, title: str, out_name: str, transform: Callable[[float], float] = lambda x: x):
    plt.figure(figsize=(10, 6))
    for i, lbl in enumerate(labels):
        data = data_map[lbl]
        epochs = [int(row.get('Epoch', idx)) for idx, row in enumerate(data, start=1)]
        values = [transform(float(row.get(metric_key, float('nan')))) for row in data]
        plt.plot(epochs, values, markers[i % len(markers)], label=lbl, linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f'python-tests/graphs/{out_name}'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    plot_metric('Train_Loss', 'Train Loss', 'Classification: Train Loss Comparison', 'classification_train_loss.png')
    plot_metric('Test_Loss', 'Test Loss', 'Classification: Test Loss Comparison', 'classification_test_loss.png')
    plot_metric('Train_Acc', 'Train Accuracy (%)', 'Classification: Train Accuracy Comparison', 'classification_train_accuracy.png', lambda x: x * 100)
    plot_metric('Test_Acc', 'Test Accuracy (%)', 'Classification: Test Accuracy Comparison', 'classification_test_accuracy.png', lambda x: x * 100)
    plot_metric('F1', 'F1 Score', 'Classification: F1 Score Comparison', 'classification_f1_score.png')
    plot_metric('CPU_Usage', 'CPU Usage (%)', 'Classification: CPU Usage Comparison', 'classification_cpu_usage.png')
    plot_metric('RAM_Usage', 'RAM Usage (%)', 'Classification: RAM Usage Comparison', 'classification_ram_usage.png')
    plot_metric('Epoch_Time', 'Time (seconds)', 'Classification: Epoch Time Comparison', 'classification_epoch_time.png')

    print("\n")
    return data_map

def compare_regression():
    """Compare regression benchmark results"""
    print("=" * 80)
    print("REGRESSION BENCHMARK COMPARISON (California Housing)")
    print("=" * 80)

    source_specs = [
        ('Python (Micrograd)', 'python-tests/micrograd/regression-california-housing/training_metrics.csv'),
        ('Rust SGD', 'python-tests/rusty-axon/regression-california-housing/rust_regression_metrics_sgd.csv'),
        ('Rust MeProp', 'python-tests/rusty-axon/regression-california-housing/rust_regression_metrics_meprop.csv'),
    ]

    loaded = load_sources(source_specs)
    if not loaded:
        print("Error: No data files found for regression. Please run the benchmarks first.")
        return None

    data_map = {label: data for label, data in loaded}
    labels = list(data_map.keys())

    # Final epoch comparison
    print("\nFINAL EPOCH METRICS")
    print("-" * 100)
    header = f"{'Metric':<25} " + " ".join(f"{lbl:<20}" for lbl in labels)
    print(header)
    print("-" * 100)

    metrics = [
        ('Loss (MSE)', 'Loss'),
        ('RMSE', 'RMSE'),
    ]
    for name, key in metrics:
        vals = []
        for lbl in labels:
            try:
                vals.append(float(data_map[lbl][-1].get(key, float('nan'))))
            except Exception:
                vals.append(float('nan'))
        line = f"{name:<25} " + " ".join(f"{v:<20.6f}" for v in vals)
        print(line)

    # Performance
    print("\nPERFORMANCE METRICS")
    time_key = 'Time_s'
    cpu_key = 'CPU_Usage'
    ram_key = 'RAM_Usage'

    totals = {lbl: sum(float(r.get(time_key, 0)) for r in data_map[lbl]) for lbl in labels}
    avgs = {lbl: {
        'avg_time': totals[lbl] / len(data_map[lbl]) if data_map[lbl] else 0,
        'avg_cpu': sum(float(r.get(cpu_key, 0)) for r in data_map[lbl]) / len(data_map[lbl]) if data_map[lbl] else 0,
        'avg_ram': sum(float(r.get(ram_key, 0)) for r in data_map[lbl]) / len(data_map[lbl]) if data_map[lbl] else 0,
    } for lbl in labels}

    for lbl in labels:
        print(f"  {lbl}: Total Time={totals[lbl]:.2f}s, Avg Epoch={avgs[lbl]['avg_time']:.2f}s, Avg CPU={avgs[lbl]['avg_cpu']:.1f}%, Avg RAM={avgs[lbl]['avg_ram']:.1f}%")

    # Plots
    print("\nGENERATING COMPARISON PLOTS...")
    ensure_graphs_dir()
    markers = ['o-', 's-', '^-', 'd-', 'x-']

def plot_metric(metric_key: str, ylabel: str, title: str, out_name: str, transform: Callable[[float], float] = lambda x: x, epoch_key: str = 'Epoch'):
    plt.figure(figsize=(10, 6))
    for i, lbl in enumerate(labels):
        data = data_map[lbl]
        epochs = [int(row.get(epoch_key, idx)) for idx, row in enumerate(data, start=1)]
        values = [transform(float(row.get(metric_key, float('nan')))) for row in data]
        plt.plot(epochs, values, markers[i % len(markers)], label=lbl, linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f'python-tests/graphs/{out_name}'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    plot_metric('Loss', 'Loss (MSE)', 'Regression: Loss (MSE) Comparison', 'regression_loss.png')
    plot_metric('RMSE', 'RMSE', 'Regression: RMSE Comparison', 'regression_rmse.png')
    plot_metric('CPU_Usage', 'CPU Usage (%)', 'Regression: CPU Usage Comparison', 'regression_cpu_usage.png')
    plot_metric('RAM_Usage', 'RAM Usage (%)', 'Regression: RAM Usage Comparison', 'regression_ram_usage.png')
    plot_metric('Time_s', 'Time (seconds)', 'Regression: Epoch Time Comparison', 'regression_epoch_time.png', epoch_key='Epoch')

    print("\n")
    return data_map

def main():
    """Main comparison function"""
    classification_data = None
    regression_data = None

    if len(sys.argv) > 1:
        benchmark = sys.argv[1].lower()
        if benchmark == 'classification':
            classification_data = compare_classification()
        elif benchmark == 'regression':
            regression_data = compare_regression()
        else:
            print(f"Unknown benchmark: {benchmark}")
            print("Usage: python compare_results.py [classification|regression]")
            return
    else:
        # Run both comparisons
        classification_data = compare_classification()
        print("\n")
        regression_data = compare_regression()

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("• Rust should be 5-10x faster due to compiled, optimized code")
    print("• Model quality (accuracy/loss) should be similar (±5%)")
    print("• Minor differences expected due to random initialization")
    print("• Rust provides type safety & memory safety guarantees")
    print("=" * 80)

    # Save summary to markdown
    if classification_data or regression_data:
        print("\n📝 Saving summary to markdown...")
        save_summary_to_markdown(classification_data, regression_data)
        print("  Saved: python-tests/graphs/benchmark_summary.md")
        print("=" * 80)

if __name__ == "__main__":
    main()

