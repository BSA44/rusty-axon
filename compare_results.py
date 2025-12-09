#!/usr/bin/env python3
"""
Comparison script for Micrograd (Python) vs Rusty-Axon (Rust) benchmarks
"""
import csv
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def load_csv(filepath):
    """Load CSV file and return as list of dictionaries"""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def ensure_graphs_dir():
    """Create graphs directory if it doesn't exist"""
    os.makedirs('python-tests/graphs', exist_ok=True)

def compare_classification():
    """Compare classification benchmark results"""
    print("=" * 80)
    print("CLASSIFICATION BENCHMARK COMPARISON (Diabetes/Pima Indians)")
    print("=" * 80)
    
    try:
        python_data = load_csv('python-tests/micrograd/classification-diabetes/training_metrics.csv')
        rust_data = load_csv('python-tests/rusty-axon/classification-diabetes/rust_classification_metrics.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run both benchmarks first:")
        print("  Python: cd python-tests/micrograd/classification-diabetes && python classification-diabetes.py")
        print("  Rust:   cargo run --release --example bench_classification_diabetes")
        return
    
    if not python_data or not rust_data:
        print("Error: Empty data files")
        return
    
    # Final epoch comparison
    py_final = python_data[-1]
    rust_final = rust_data[-1]
    
    print("\n📊 FINAL EPOCH METRICS (Epoch 50)")
    print("-" * 80)
    print(f"{'Metric':<20} {'Python (Micrograd)':<25} {'Rust (Rusty-Axon)':<25} {'Difference'}")
    print("-" * 80)
    
    metrics = [
        ('Train Loss', 'Train_Loss'),
        ('Train Accuracy', 'Train_Acc'),
        ('Test Loss', 'Test_Loss'),
        ('Test Accuracy', 'Test_Acc'),
        ('F1 Score', 'F1'),
    ]
    
    for name, key in metrics:
        py_val = float(py_final[key])
        rust_val = float(rust_final[key])
        diff = rust_val - py_val
        print(f"{name:<20} {py_val:<25.6f} {rust_val:<25.6f} {diff:+.6f}")
    
    # Performance comparison
    print("\n⚡ PERFORMANCE METRICS")
    print("-" * 80)
    
    py_total_time = sum(float(row['Epoch_Time']) for row in python_data)
    rust_total_time = sum(float(row['Epoch_Time']) for row in rust_data)
    speedup = py_total_time / rust_total_time if rust_total_time > 0 else 0
    
    py_avg_time = py_total_time / len(python_data)
    rust_avg_time = rust_total_time / len(rust_data)
    
    print(f"Total Training Time:")
    print(f"  Python:  {py_total_time:.2f}s")
    print(f"  Rust:    {rust_total_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x {'🚀' if speedup > 1 else ''}")
    
    print(f"\nAverage Time per Epoch:")
    print(f"  Python:  {py_avg_time:.2f}s")
    print(f"  Rust:    {rust_avg_time:.2f}s")
    
    py_avg_cpu = sum(float(row['CPU_Usage']) for row in python_data) / len(python_data)
    rust_avg_cpu = sum(float(row['CPU_Usage']) for row in rust_data) / len(rust_data)
    
    py_avg_ram = sum(float(row['RAM_Usage']) for row in python_data) / len(python_data)
    rust_avg_ram = sum(float(row['RAM_Usage']) for row in rust_data) / len(rust_data)
    
    print(f"\nAverage CPU Usage:")
    print(f"  Python:  {py_avg_cpu:.1f}%")
    print(f"  Rust:    {rust_avg_cpu:.1f}%")
    
    print(f"\nAverage RAM Usage:")
    print(f"  Python:  {py_avg_ram:.1f}%")
    print(f"  Rust:    {rust_avg_ram:.1f}%")
    
    # Generate plots
    print("\n📈 GENERATING COMPARISON PLOTS...")
    ensure_graphs_dir()
    
    epochs_py = [int(row['Epoch']) for row in python_data]
    epochs_rust = [int(row['Epoch']) for row in rust_data]
    
    # Plot 1: Train Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['Train_Loss']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust, [float(row['Train_Loss']) for row in rust_data], 's-', label='Rust (Rusty-Axon)', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Train Loss', fontsize=12)
    plt.title('Classification: Train Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_train_loss.png', dpi=150)
    plt.close()
    print("  ✓ Saved: python-tests/graphs/classification_train_loss.png")
    
    # Plot 2: Test Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['Test_Loss']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust, [float(row['Test_Loss']) for row in rust_data], 's-', label='Rust (Rusty-Axon)', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.title('Classification: Test Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_test_loss.png', dpi=150)
    plt.close()
    print("  ✓ Saved: python-tests/graphs/classification_test_loss.png")
    
    # Plot 3: Train Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['Train_Acc']) * 100 for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust, [float(row['Train_Acc']) * 100 for row in rust_data], 's-', label='Rust (Rusty-Axon)', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Train Accuracy (%)', fontsize=12)
    plt.title('Classification: Train Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_train_accuracy.png', dpi=150)
    plt.close()
    print("  ✓ Saved: python-tests/graphs/classification_train_accuracy.png")
    
    # Plot 4: Test Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['Test_Acc']) * 100 for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust, [float(row['Test_Acc']) * 100 for row in rust_data], 's-', label='Rust (Rusty-Axon)', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Classification: Test Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_test_accuracy.png', dpi=150)
    plt.close()
    print("  ✓ Saved: python-tests/graphs/classification_test_accuracy.png")
    
    # Plot 5: F1 Score
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['F1']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust, [float(row['F1']) for row in rust_data], 's-', label='Rust (Rusty-Axon)', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Classification: F1 Score Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_f1_score.png', dpi=150)
    plt.close()
    print("  ✓ Saved: python-tests/graphs/classification_f1_score.png")
    
    # Plot 6: CPU Usage
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['CPU_Usage']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust, [float(row['CPU_Usage']) for row in rust_data], 's-', label='Rust (Rusty-Axon)', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('CPU Usage (%)', fontsize=12)
    plt.title('Classification: CPU Usage Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_cpu_usage.png', dpi=150)
    plt.close()
    print("  ✓ Saved: python-tests/graphs/classification_cpu_usage.png")
    
    # Plot 7: RAM Usage
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['RAM_Usage']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust, [float(row['RAM_Usage']) for row in rust_data], 's-', label='Rust (Rusty-Axon)', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('RAM Usage (%)', fontsize=12)
    plt.title('Classification: RAM Usage Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_ram_usage.png', dpi=150)
    plt.close()
    print("  ✓ Saved: python-tests/graphs/classification_ram_usage.png")
    
    # Plot 8: Epoch Time
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['Epoch_Time']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust, [float(row['Epoch_Time']) for row in rust_data], 's-', label='Rust (Rusty-Axon)', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Classification: Epoch Time Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_epoch_time.png', dpi=150)
    plt.close()
    print("  ✓ Saved: python-tests/graphs/classification_epoch_time.png")
    
    print("\n")

def compare_regression():
    """Compare regression benchmark results"""
    print("=" * 80)
    print("REGRESSION BENCHMARK COMPARISON (California Housing)")
    print("=" * 80)
    
    try:
        python_data = load_csv('python-tests/micrograd/regression-california-housing/training_metrics.csv')
        rust_data = load_csv('python-tests/rusty-axon/regression-california-housing/rust_regression_metrics.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run both benchmarks first:")
        print("  Python: cd python-tests/micrograd/regression-california-housing && python regression-california-housing.py")
        print("  Rust:   cargo run --release --example bench_regression_housing")
        return
    
    if not python_data or not rust_data:
        print("Error: Empty data files")
        return
    
    # Final epoch comparison
    py_final = python_data[-1]
    rust_final = rust_data[-1]
    
    print("\n📊 FINAL EPOCH METRICS (Epoch 5)")
    print("-" * 80)
    print(f"{'Metric':<20} {'Python (Micrograd)':<25} {'Rust (Rusty-Axon)':<25} {'Difference'}")
    print("-" * 80)
    
    metrics = [
        ('Loss (MSE)', 'Loss'),
        ('RMSE', 'RMSE'),
    ]
    
    for name, key in metrics:
        py_val = float(py_final[key])
        rust_val = float(rust_final[key])
        diff = rust_val - py_val
        print(f"{name:<20} {py_val:<25.6f} {rust_val:<25.6f} {diff:+.6f}")
    
    # Performance comparison
    print("\n⚡ PERFORMANCE METRICS")
    print("-" * 80)
    
    py_total_time = sum(float(row['Time_s']) for row in python_data)
    rust_total_time = sum(float(row['Time_s']) for row in rust_data)
    speedup = py_total_time / rust_total_time if rust_total_time > 0 else 0
    
    py_avg_time = py_total_time / len(python_data)
    rust_avg_time = rust_total_time / len(rust_data)
    
    print(f"Total Training Time:")
    print(f"  Python:  {py_total_time:.2f}s")
    print(f"  Rust:    {rust_total_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x {'🚀' if speedup > 1 else ''}")
    
    print(f"\nAverage Time per Epoch:")
    print(f"  Python:  {py_avg_time:.2f}s")
    print(f"  Rust:    {rust_avg_time:.2f}s")
    
    py_avg_cpu = sum(float(row['CPU_Usage']) for row in python_data) / len(python_data)
    rust_avg_cpu = sum(float(row['CPU_Usage']) for row in rust_data) / len(rust_data)
    
    py_avg_ram = sum(float(row['RAM_Usage']) for row in python_data) / len(python_data)
    rust_avg_ram = sum(float(row['RAM_Usage']) for row in rust_data) / len(rust_data)
    
    print(f"\nAverage CPU Usage:")
    print(f"  Python:  {py_avg_cpu:.1f}%")
    print(f"  Rust:    {rust_avg_cpu:.1f}%")
    
    print(f"\nAverage RAM Usage:")
    print(f"  Python:  {py_avg_ram:.1f}%")
    print(f"  Rust:    {rust_avg_ram:.1f}%")
    
    # Generate plots
    print("\n📈 GENERATING COMPARISON PLOTS...")
    ensure_graphs_dir()
    
    epochs_py = [int(row['Epoch']) for row in python_data]
    epochs_rust = [int(row['Epoch']) for row in rust_data]
    
    # Plot 1: Loss (MSE)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['Loss']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=6)
    plt.plot(epochs_rust, [float(row['Loss']) for row in rust_data], 's-', label='Rust (Rusty-Axon)', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Regression: Loss (MSE) Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/regression_loss.png', dpi=150)
    plt.close()
    print("  ✓ Saved: python-tests/graphs/regression_loss.png")
    
    # Plot 2: RMSE
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['RMSE']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=6)
    plt.plot(epochs_rust, [float(row['RMSE']) for row in rust_data], 's-', label='Rust (Rusty-Axon)', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Regression: RMSE Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/regression_rmse.png', dpi=150)
    plt.close()
    print("  ✓ Saved: python-tests/graphs/regression_rmse.png")
    
    # Plot 3: CPU Usage
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['CPU_Usage']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=6)
    plt.plot(epochs_rust, [float(row['CPU_Usage']) for row in rust_data], 's-', label='Rust (Rusty-Axon)', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('CPU Usage (%)', fontsize=12)
    plt.title('Regression: CPU Usage Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/regression_cpu_usage.png', dpi=150)
    plt.close()
    print("  ✓ Saved: python-tests/graphs/regression_cpu_usage.png")
    
    # Plot 4: RAM Usage
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['RAM_Usage']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=6)
    plt.plot(epochs_rust, [float(row['RAM_Usage']) for row in rust_data], 's-', label='Rust (Rusty-Axon)', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('RAM Usage (%)', fontsize=12)
    plt.title('Regression: RAM Usage Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/regression_ram_usage.png', dpi=150)
    plt.close()
    print("  ✓ Saved: python-tests/graphs/regression_ram_usage.png")
    
    # Plot 5: Epoch Time
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['Time_s']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=6)
    plt.plot(epochs_rust, [float(row['Time_s']) for row in rust_data], 's-', label='Rust (Rusty-Axon)', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Regression: Epoch Time Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/regression_epoch_time.png', dpi=150)
    plt.close()
    print("  ✓ Saved: python-tests/graphs/regression_epoch_time.png")
    
    print("\n")

def main():
    """Main comparison function"""
    if len(sys.argv) > 1:
        benchmark = sys.argv[1].lower()
        if benchmark == 'classification':
            compare_classification()
        elif benchmark == 'regression':
            compare_regression()
        else:
            print(f"Unknown benchmark: {benchmark}")
            print("Usage: python compare_results.py [classification|regression]")
    else:
        # Run both comparisons
        compare_classification()
        print("\n")
        compare_regression()
    
    print("\n" + "=" * 80)
    print("💡 KEY TAKEAWAYS")
    print("=" * 80)
    print("• Rust should be 5-10x faster due to compiled, optimized code")
    print("• Model quality (accuracy/loss) should be similar (±5%)")
    print("• Minor differences expected due to random initialization")
    print("• Rust provides type safety & memory safety guarantees")
    print("=" * 80)

if __name__ == "__main__":
    main()

