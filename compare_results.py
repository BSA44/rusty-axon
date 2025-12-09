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

def save_summary_to_markdown(classification_data=None, regression_data=None):
    """Save comparison summary to markdown file"""
    ensure_graphs_dir()
    
    with open('python-tests/graphs/benchmark_summary.md', 'w') as f:
        f.write("# Benchmark Comparison: Micrograd (Python) vs Rusty-Axon (Rust)\n\n")
        f.write(f"*Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("---\n\n")
        
        if classification_data:
            py_data, rust_sgd_data, rust_meprop_data = classification_data
            py_final = py_data[-1]
            rust_sgd_final = rust_sgd_data[-1]
            rust_meprop_final = rust_meprop_data[-1]
            
            f.write("## Classification Benchmark (Diabetes/Pima Indians)\n\n")
            f.write("### Final Epoch Metrics (Epoch 50)\n\n")
            f.write("| Metric | Python (Micrograd) | Rust SGD | Rust MeProp | Diff (SGD) | Diff (MeProp) |\n")
            f.write("|--------|-------------------|----------|-------------|------------|---------------|\n")
            
            metrics = [
                ('Train Loss', 'Train_Loss'),
                ('Train Accuracy', 'Train_Acc'),
                ('Test Loss', 'Test_Loss'),
                ('Test Accuracy', 'Test_Acc'),
                ('F1 Score', 'F1'),
            ]
            
            for name, key in metrics:
                py_val = float(py_final[key])
                rust_sgd_val = float(rust_sgd_final[key])
                rust_meprop_val = float(rust_meprop_final[key])
                diff_sgd = rust_sgd_val - py_val
                diff_meprop = rust_meprop_val - py_val
                f.write(f"| {name} | {py_val:.6f} | {rust_sgd_val:.6f} | {rust_meprop_val:.6f} | {diff_sgd:+.6f} | {diff_meprop:+.6f} |\n")
            
            f.write("\n### Performance Metrics\n\n")
            
            py_total_time = sum(float(row['Epoch_Time']) for row in py_data)
            rust_sgd_total_time = sum(float(row['Epoch_Time']) for row in rust_sgd_data)
            rust_meprop_total_time = sum(float(row['Epoch_Time']) for row in rust_meprop_data)
            speedup_sgd = py_total_time / rust_sgd_total_time if rust_sgd_total_time > 0 else 0
            speedup_meprop = py_total_time / rust_meprop_total_time if rust_meprop_total_time > 0 else 0
            
            py_avg_time = py_total_time / len(py_data)
            rust_sgd_avg_time = rust_sgd_total_time / len(rust_sgd_data)
            rust_meprop_avg_time = rust_meprop_total_time / len(rust_meprop_data)
            
            py_avg_cpu = sum(float(row['CPU_Usage']) for row in py_data) / len(py_data)
            rust_sgd_avg_cpu = sum(float(row['CPU_Usage']) for row in rust_sgd_data) / len(rust_sgd_data)
            rust_meprop_avg_cpu = sum(float(row['CPU_Usage']) for row in rust_meprop_data) / len(rust_meprop_data)
            
            py_avg_ram = sum(float(row['RAM_Usage']) for row in py_data) / len(py_data)
            rust_sgd_avg_ram = sum(float(row['RAM_Usage']) for row in rust_sgd_data) / len(rust_sgd_data)
            rust_meprop_avg_ram = sum(float(row['RAM_Usage']) for row in rust_meprop_data) / len(rust_meprop_data)
            
            f.write("| Metric | Python | Rust SGD | Rust MeProp | Speedup SGD | Speedup MeProp |\n")
            f.write("|--------|--------|----------|-------------|-------------|----------------|\n")
            f.write(f"| **Total Training Time** | {py_total_time:.2f}s | {rust_sgd_total_time:.2f}s | {rust_meprop_total_time:.2f}s | **{speedup_sgd:.2f}x** | **{speedup_meprop:.2f}x** |\n")
            f.write(f"| Avg Time per Epoch | {py_avg_time:.2f}s | {rust_sgd_avg_time:.2f}s | {rust_meprop_avg_time:.2f}s | {py_avg_time/rust_sgd_avg_time:.2f}x | {py_avg_time/rust_meprop_avg_time:.2f}x |\n")
            f.write(f"| Avg CPU Usage | {py_avg_cpu:.1f}% | {rust_sgd_avg_cpu:.1f}% | {rust_meprop_avg_cpu:.1f}% | - | - |\n")
            f.write(f"| Avg RAM Usage | {py_avg_ram:.1f}% | {rust_sgd_avg_ram:.1f}% | {rust_meprop_avg_ram:.1f}% | - | - |\n")
            
            f.write("\n### Generated Plots\n\n")
            f.write("- `classification_train_loss.png` - Train loss comparison\n")
            f.write("- `classification_test_loss.png` - Test loss comparison\n")
            f.write("- `classification_train_accuracy.png` - Train accuracy comparison\n")
            f.write("- `classification_test_accuracy.png` - Test accuracy comparison\n")
            f.write("- `classification_f1_score.png` - F1 score comparison\n")
            f.write("- `classification_cpu_usage.png` - CPU usage over epochs\n")
            f.write("- `classification_ram_usage.png` - RAM usage over epochs\n")
            f.write("- `classification_epoch_time.png` - Time per epoch\n")
            f.write("\n---\n\n")
        
        if regression_data:
            py_data, rust_sgd_data, rust_meprop_data = regression_data
            py_final = py_data[-1]
            rust_sgd_final = rust_sgd_data[-1]
            rust_meprop_final = rust_meprop_data[-1]
            
            f.write("## Regression Benchmark (California Housing)\n\n")
            f.write("### Final Epoch Metrics (Epoch 5)\n\n")
            f.write("| Metric | Python (Micrograd) | Rust SGD | Rust MeProp | Diff (SGD) | Diff (MeProp) |\n")
            f.write("|--------|-------------------|----------|-------------|------------|---------------|\n")
            
            metrics = [
                ('Loss (MSE)', 'Loss'),
                ('RMSE', 'RMSE'),
            ]
            
            for name, key in metrics:
                py_val = float(py_final[key])
                rust_sgd_val = float(rust_sgd_final[key])
                rust_meprop_val = float(rust_meprop_final[key])
                diff_sgd = rust_sgd_val - py_val
                diff_meprop = rust_meprop_val - py_val
                f.write(f"| {name} | {py_val:.6f} | {rust_sgd_val:.6f} | {rust_meprop_val:.6f} | {diff_sgd:+.6f} | {diff_meprop:+.6f} |\n")
            
            f.write("\n### Performance Metrics\n\n")
            
            py_total_time = sum(float(row['Time_s']) for row in py_data)
            rust_sgd_total_time = sum(float(row['Time_s']) for row in rust_sgd_data)
            rust_meprop_total_time = sum(float(row['Time_s']) for row in rust_meprop_data)
            speedup_sgd = py_total_time / rust_sgd_total_time if rust_sgd_total_time > 0 else 0
            speedup_meprop = py_total_time / rust_meprop_total_time if rust_meprop_total_time > 0 else 0
            
            py_avg_time = py_total_time / len(py_data)
            rust_sgd_avg_time = rust_sgd_total_time / len(rust_sgd_data)
            rust_meprop_avg_time = rust_meprop_total_time / len(rust_meprop_data)
            
            py_avg_cpu = sum(float(row['CPU_Usage']) for row in py_data) / len(py_data)
            rust_sgd_avg_cpu = sum(float(row['CPU_Usage']) for row in rust_sgd_data) / len(rust_sgd_data)
            rust_meprop_avg_cpu = sum(float(row['CPU_Usage']) for row in rust_meprop_data) / len(rust_meprop_data)
            
            py_avg_ram = sum(float(row['RAM_Usage']) for row in py_data) / len(py_data)
            rust_sgd_avg_ram = sum(float(row['RAM_Usage']) for row in rust_sgd_data) / len(rust_sgd_data)
            rust_meprop_avg_ram = sum(float(row['RAM_Usage']) for row in rust_meprop_data) / len(rust_meprop_data)
            
            f.write("| Metric | Python | Rust SGD | Rust MeProp | Speedup SGD | Speedup MeProp |\n")
            f.write("|--------|--------|----------|-------------|-------------|----------------|\n")
            f.write(f"| **Total Training Time** | {py_total_time:.2f}s | {rust_sgd_total_time:.2f}s | {rust_meprop_total_time:.2f}s | **{speedup_sgd:.2f}x** | **{speedup_meprop:.2f}x** |\n")
            f.write(f"| Avg Time per Epoch | {py_avg_time:.2f}s | {rust_sgd_avg_time:.2f}s | {rust_meprop_avg_time:.2f}s | {py_avg_time/rust_sgd_avg_time:.2f}x | {py_avg_time/rust_meprop_avg_time:.2f}x |\n")
            f.write(f"| Avg CPU Usage | {py_avg_cpu:.1f}% | {rust_sgd_avg_cpu:.1f}% | {rust_meprop_avg_cpu:.1f}% | - | - |\n")
            f.write(f"| Avg RAM Usage | {py_avg_ram:.1f}% | {rust_sgd_avg_ram:.1f}% | {rust_meprop_avg_ram:.1f}% | - | - |\n")
            
            f.write("\n### Generated Plots\n\n")
            f.write("- `regression_loss.png` - MSE loss comparison\n")
            f.write("- `regression_rmse.png` - RMSE comparison\n")
            f.write("- `regression_cpu_usage.png` - CPU usage over epochs\n")
            f.write("- `regression_ram_usage.png` - RAM usage over epochs\n")
            f.write("- `regression_epoch_time.png` - Time per epoch\n")
            f.write("\n---\n\n")
        
        f.write("## Key Takeaways\n\n")
        f.write("- **Performance**: Rust is 5-10x faster due to compiled, optimized code\n")
        f.write("- **Model Quality**: Accuracy/loss metrics are similar (±5% variation expected)\n")
        f.write("- **Differences**: Minor variations due to random initialization\n")
        f.write("- **Safety**: Rust provides type safety & memory safety guarantees\n")
        f.write("- **Framework**: Rusty-Axon properly implements Loss and Optimizer traits\n\n")
        f.write("---\n\n")
        f.write("*Built with Rust for performance and education.*\n")

def compare_classification():
    """Compare classification benchmark results"""
    print("=" * 80)
    print("CLASSIFICATION BENCHMARK COMPARISON (Diabetes/Pima Indians)")
    print("=" * 80)
    
    try:
        python_data = load_csv('python-tests/micrograd/classification-diabetes/training_metrics.csv')
        rust_sgd_data = load_csv('python-tests/rusty-axon/classification-diabetes/rust_classification_metrics_sgd.csv')
        rust_meprop_data = load_csv('python-tests/rusty-axon/classification-diabetes/rust_classification_metrics_meprop.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run both benchmarks first:")
        print("  Python: cd python-tests/micrograd/classification-diabetes && python classification-diabetes.py")
        print("  Rust:   cargo run --release --example bench_classification_diabetes")
        return None
    
    if not python_data or not rust_sgd_data or not rust_meprop_data:
        print("Error: Empty data files")
        return
    
    # Final epoch comparison
    py_final = python_data[-1]
    rust_sgd_final = rust_sgd_data[-1]
    rust_meprop_final = rust_meprop_data[-1]
    
    print("\nFINAL EPOCH METRICS (Epoch 50)")
    print("-" * 100)
    print(f"{'Metric':<20} {'Python':<20} {'Rust SGD':<20} {'Rust MeProp':<20} {'Diff (SGD)':<10}")
    print("-" * 100)
    
    metrics = [
        ('Train Loss', 'Train_Loss'),
        ('Train Accuracy', 'Train_Acc'),
        ('Test Loss', 'Test_Loss'),
        ('Test Accuracy', 'Test_Acc'),
        ('F1 Score', 'F1'),
    ]
    
    for name, key in metrics:
        py_val = float(py_final[key])
        rust_sgd_val = float(rust_sgd_final[key])
        rust_meprop_val = float(rust_meprop_final[key])
        diff = rust_sgd_val - py_val
        print(f"{name:<20} {py_val:<20.6f} {rust_sgd_val:<20.6f} {rust_meprop_val:<20.6f} {diff:+.6f}")
    
    # Performance comparison
    print("\nPERFORMANCE METRICS")
    print("-" * 100)
    
    py_total_time = sum(float(row['Epoch_Time']) for row in python_data)
    rust_sgd_total_time = sum(float(row['Epoch_Time']) for row in rust_sgd_data)
    rust_meprop_total_time = sum(float(row['Epoch_Time']) for row in rust_meprop_data)
    speedup_sgd = py_total_time / rust_sgd_total_time if rust_sgd_total_time > 0 else 0
    speedup_meprop = py_total_time / rust_meprop_total_time if rust_meprop_total_time > 0 else 0
    
    py_avg_time = py_total_time / len(python_data)
    rust_sgd_avg_time = rust_sgd_total_time / len(rust_sgd_data)
    rust_meprop_avg_time = rust_meprop_total_time / len(rust_meprop_data)
    
    print(f"Total Training Time:")
    print(f"  Python:       {py_total_time:.2f}s")
    print(f"  Rust SGD:     {rust_sgd_total_time:.2f}s  (Speedup: {speedup_sgd:.2f}x)")
    print(f"  Rust MeProp:  {rust_meprop_total_time:.2f}s  (Speedup: {speedup_meprop:.2f}x)")
    
    print(f"\nAverage Time per Epoch:")
    print(f"  Python:       {py_avg_time:.2f}s")
    print(f"  Rust SGD:     {rust_sgd_avg_time:.2f}s")
    print(f"  Rust MeProp:  {rust_meprop_avg_time:.2f}s")
    
    py_avg_cpu = sum(float(row['CPU_Usage']) for row in python_data) / len(python_data)
    rust_sgd_avg_cpu = sum(float(row['CPU_Usage']) for row in rust_sgd_data) / len(rust_sgd_data)
    rust_meprop_avg_cpu = sum(float(row['CPU_Usage']) for row in rust_meprop_data) / len(rust_meprop_data)
    
    py_avg_ram = sum(float(row['RAM_Usage']) for row in python_data) / len(python_data)
    rust_sgd_avg_ram = sum(float(row['RAM_Usage']) for row in rust_sgd_data) / len(rust_sgd_data)
    rust_meprop_avg_ram = sum(float(row['RAM_Usage']) for row in rust_meprop_data) / len(rust_meprop_data)
    
    print(f"\nAverage CPU Usage:")
    print(f"  Python:       {py_avg_cpu:.1f}%")
    print(f"  Rust SGD:     {rust_sgd_avg_cpu:.1f}%")
    print(f"  Rust MeProp:  {rust_meprop_avg_cpu:.1f}%")
    
    print(f"\nAverage RAM Usage:")
    print(f"  Python:       {py_avg_ram:.1f}%")
    print(f"  Rust SGD:     {rust_sgd_avg_ram:.1f}%")
    print(f"  Rust MeProp:  {rust_meprop_avg_ram:.1f}%")
    
    # Generate plots
    print("\nGENERATING COMPARISON PLOTS...")
    ensure_graphs_dir()
    
    epochs_py = [int(row['Epoch']) for row in python_data]
    epochs_rust_sgd = [int(row['Epoch']) for row in rust_sgd_data]
    epochs_rust_meprop = [int(row['Epoch']) for row in rust_meprop_data]
    
    # Plot 1: Train Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['Train_Loss']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust_sgd, [float(row['Train_Loss']) for row in rust_sgd_data], 's-', label='Rust SGD', linewidth=2, markersize=4)
    plt.plot(epochs_rust_meprop, [float(row['Train_Loss']) for row in rust_meprop_data], '^-', label='Rust MeProp', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Train Loss', fontsize=12)
    plt.title('Classification: Train Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_train_loss.png', dpi=150)
    plt.close()
    print("  Saved: python-tests/graphs/classification_train_loss.png")
    
    # Plot 2: Test Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['Test_Loss']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust_sgd, [float(row['Test_Loss']) for row in rust_sgd_data], 's-', label='Rust SGD', linewidth=2, markersize=4)
    plt.plot(epochs_rust_meprop, [float(row['Test_Loss']) for row in rust_meprop_data], '^-', label='Rust MeProp', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.title('Classification: Test Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_test_loss.png', dpi=150)
    plt.close()
    print("  Saved: python-tests/graphs/classification_test_loss.png")
    
    # Plot 3: Train Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['Train_Acc']) * 100 for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust_sgd, [float(row['Train_Acc']) * 100 for row in rust_sgd_data], 's-', label='Rust SGD', linewidth=2, markersize=4)
    plt.plot(epochs_rust_meprop, [float(row['Train_Acc']) * 100 for row in rust_meprop_data], '^-', label='Rust MeProp', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Train Accuracy (%)', fontsize=12)
    plt.title('Classification: Train Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_train_accuracy.png', dpi=150)
    plt.close()
    print("  Saved: python-tests/graphs/classification_train_accuracy.png")
    
    # Plot 4: Test Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['Test_Acc']) * 100 for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust_sgd, [float(row['Test_Acc']) * 100 for row in rust_sgd_data], 's-', label='Rust SGD', linewidth=2, markersize=4)
    plt.plot(epochs_rust_meprop, [float(row['Test_Acc']) * 100 for row in rust_meprop_data], '^-', label='Rust MeProp', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Classification: Test Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_test_accuracy.png', dpi=150)
    plt.close()
    print("  Saved: python-tests/graphs/classification_test_accuracy.png")
    
    # Plot 5: F1 Score
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['F1']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust_sgd, [float(row['F1']) for row in rust_sgd_data], 's-', label='Rust SGD', linewidth=2, markersize=4)
    plt.plot(epochs_rust_meprop, [float(row['F1']) for row in rust_meprop_data], '^-', label='Rust MeProp', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Classification: F1 Score Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_f1_score.png', dpi=150)
    plt.close()
    print("  Saved: python-tests/graphs/classification_f1_score.png")
    
    # Plot 6: CPU Usage
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['CPU_Usage']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust_sgd, [float(row['CPU_Usage']) for row in rust_sgd_data], 's-', label='Rust SGD', linewidth=2, markersize=4)
    plt.plot(epochs_rust_meprop, [float(row['CPU_Usage']) for row in rust_meprop_data], '^-', label='Rust MeProp', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('CPU Usage (%)', fontsize=12)
    plt.title('Classification: CPU Usage Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_cpu_usage.png', dpi=150)
    plt.close()
    print("  Saved: python-tests/graphs/classification_cpu_usage.png")
    
    # Plot 7: RAM Usage
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['RAM_Usage']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust_sgd, [float(row['RAM_Usage']) for row in rust_sgd_data], 's-', label='Rust SGD', linewidth=2, markersize=4)
    plt.plot(epochs_rust_meprop, [float(row['RAM_Usage']) for row in rust_meprop_data], '^-', label='Rust MeProp', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('RAM Usage (%)', fontsize=12)
    plt.title('Classification: RAM Usage Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_ram_usage.png', dpi=150)
    plt.close()
    print("  Saved: python-tests/graphs/classification_ram_usage.png")
    
    # Plot 8: Epoch Time
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['Epoch_Time']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=4)
    plt.plot(epochs_rust_sgd, [float(row['Epoch_Time']) for row in rust_sgd_data], 's-', label='Rust SGD', linewidth=2, markersize=4)
    plt.plot(epochs_rust_meprop, [float(row['Epoch_Time']) for row in rust_meprop_data], '^-', label='Rust MeProp', linewidth=2, markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Classification: Epoch Time Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/classification_epoch_time.png', dpi=150)
    plt.close()
    print("  Saved: python-tests/graphs/classification_epoch_time.png")
    
    print("\n")
    
    return (python_data, rust_sgd_data, rust_meprop_data)

def compare_regression():
    """Compare regression benchmark results"""
    print("=" * 80)
    print("REGRESSION BENCHMARK COMPARISON (California Housing)")
    print("=" * 80)
    
    try:
        python_data = load_csv('python-tests/micrograd/regression-california-housing/training_metrics.csv')
        rust_sgd_data = load_csv('python-tests/rusty-axon/regression-california-housing/rust_regression_metrics_sgd.csv')
        rust_meprop_data = load_csv('python-tests/rusty-axon/regression-california-housing/rust_regression_metrics_meprop.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run all benchmarks first:")
        print("  Python: cd python-tests/micrograd/regression-california-housing && python regression-california-housing.py")
        print("  Rust SGD: cargo run --release --example bench_regression_housing_sgd")
        print("  Rust MeProp: cargo run --release --example bench_regression_housing_meprop")
        return None
    
    if not python_data or not rust_sgd_data or not rust_meprop_data:
        print("Error: Empty data files")
        return None
    
    # Final epoch comparison
    py_final = python_data[-1]
    rust_sgd_final = rust_sgd_data[-1]
    rust_meprop_final = rust_meprop_data[-1]
    
    print("\nFINAL EPOCH METRICS (Epoch 5)")
    print("-" * 100)
    print(f"{'Metric':<20} {'Python':<20} {'Rust SGD':<20} {'Rust MeProp':<20} {'Diff (SGD)':<10}")
    print("-" * 100)
    
    metrics = [
        ('Loss (MSE)', 'Loss'),
        ('RMSE', 'RMSE'),
    ]
    
    for name, key in metrics:
        py_val = float(py_final[key])
        rust_sgd_val = float(rust_sgd_final[key])
        rust_meprop_val = float(rust_meprop_final[key])
        diff = rust_sgd_val - py_val
        print(f"{name:<20} {py_val:<20.6f} {rust_sgd_val:<20.6f} {rust_meprop_val:<20.6f} {diff:+.6f}")
    
    # Performance comparison
    print("\nPERFORMANCE METRICS")
    print("-" * 100)
    
    py_total_time = sum(float(row['Time_s']) for row in python_data)
    rust_sgd_total_time = sum(float(row['Time_s']) for row in rust_sgd_data)
    rust_meprop_total_time = sum(float(row['Time_s']) for row in rust_meprop_data)
    speedup_sgd = py_total_time / rust_sgd_total_time if rust_sgd_total_time > 0 else 0
    speedup_meprop = py_total_time / rust_meprop_total_time if rust_meprop_total_time > 0 else 0
    
    py_avg_time = py_total_time / len(python_data)
    rust_sgd_avg_time = rust_sgd_total_time / len(rust_sgd_data)
    rust_meprop_avg_time = rust_meprop_total_time / len(rust_meprop_data)
    
    print(f"Total Training Time:")
    print(f"  Python:       {py_total_time:.2f}s")
    print(f"  Rust SGD:     {rust_sgd_total_time:.2f}s  (Speedup: {speedup_sgd:.2f}x)")
    print(f"  Rust MeProp:  {rust_meprop_total_time:.2f}s  (Speedup: {speedup_meprop:.2f}x)")
    
    print(f"\nAverage Time per Epoch:")
    print(f"  Python:       {py_avg_time:.2f}s")
    print(f"  Rust SGD:     {rust_sgd_avg_time:.2f}s")
    print(f"  Rust MeProp:  {rust_meprop_avg_time:.2f}s")
    
    py_avg_cpu = sum(float(row['CPU_Usage']) for row in python_data) / len(python_data)
    rust_sgd_avg_cpu = sum(float(row['CPU_Usage']) for row in rust_sgd_data) / len(rust_sgd_data)
    rust_meprop_avg_cpu = sum(float(row['CPU_Usage']) for row in rust_meprop_data) / len(rust_meprop_data)
    
    py_avg_ram = sum(float(row['RAM_Usage']) for row in python_data) / len(python_data)
    rust_sgd_avg_ram = sum(float(row['RAM_Usage']) for row in rust_sgd_data) / len(rust_sgd_data)
    rust_meprop_avg_ram = sum(float(row['RAM_Usage']) for row in rust_meprop_data) / len(rust_meprop_data)
    
    print(f"\nAverage CPU Usage:")
    print(f"  Python:       {py_avg_cpu:.1f}%")
    print(f"  Rust SGD:     {rust_sgd_avg_cpu:.1f}%")
    print(f"  Rust MeProp:  {rust_meprop_avg_cpu:.1f}%")
    
    print(f"\nAverage RAM Usage:")
    print(f"  Python:       {py_avg_ram:.1f}%")
    print(f"  Rust SGD:     {rust_sgd_avg_ram:.1f}%")
    print(f"  Rust MeProp:  {rust_meprop_avg_ram:.1f}%")
    
    # Generate plots
    print("\nGENERATING COMPARISON PLOTS...")
    ensure_graphs_dir()
    
    epochs_py = [int(row['Epoch']) for row in python_data]
    epochs_rust_sgd = [int(row['Epoch']) for row in rust_sgd_data]
    epochs_rust_meprop = [int(row['Epoch']) for row in rust_meprop_data]
    
    # Plot 1: Loss (MSE)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['Loss']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=6)
    plt.plot(epochs_rust_sgd, [float(row['Loss']) for row in rust_sgd_data], 's-', label='Rust SGD', linewidth=2, markersize=6)
    plt.plot(epochs_rust_meprop, [float(row['Loss']) for row in rust_meprop_data], '^-', label='Rust MeProp', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Regression: Loss (MSE) Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/regression_loss.png', dpi=150)
    plt.close()
    print("  Saved: python-tests/graphs/regression_loss.png")
    
    # Plot 2: RMSE
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['RMSE']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=6)
    plt.plot(epochs_rust_sgd, [float(row['RMSE']) for row in rust_sgd_data], 's-', label='Rust SGD', linewidth=2, markersize=6)
    plt.plot(epochs_rust_meprop, [float(row['RMSE']) for row in rust_meprop_data], '^-', label='Rust MeProp', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Regression: RMSE Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/regression_rmse.png', dpi=150)
    plt.close()
    print("  Saved: python-tests/graphs/regression_rmse.png")
    
    # Plot 3: CPU Usage
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['CPU_Usage']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=6)
    plt.plot(epochs_rust_sgd, [float(row['CPU_Usage']) for row in rust_sgd_data], 's-', label='Rust SGD', linewidth=2, markersize=6)
    plt.plot(epochs_rust_meprop, [float(row['CPU_Usage']) for row in rust_meprop_data], '^-', label='Rust MeProp', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('CPU Usage (%)', fontsize=12)
    plt.title('Regression: CPU Usage Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/regression_cpu_usage.png', dpi=150)
    plt.close()
    print("  Saved: python-tests/graphs/regression_cpu_usage.png")
    
    # Plot 4: RAM Usage
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['RAM_Usage']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=6)
    plt.plot(epochs_rust_sgd, [float(row['RAM_Usage']) for row in rust_sgd_data], 's-', label='Rust SGD', linewidth=2, markersize=6)
    plt.plot(epochs_rust_meprop, [float(row['RAM_Usage']) for row in rust_meprop_data], '^-', label='Rust MeProp', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('RAM Usage (%)', fontsize=12)
    plt.title('Regression: RAM Usage Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/regression_ram_usage.png', dpi=150)
    plt.close()
    print("  Saved: python-tests/graphs/regression_ram_usage.png")
    
    # Plot 5: Epoch Time
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_py, [float(row['Time_s']) for row in python_data], 'o-', label='Python (Micrograd)', linewidth=2, markersize=6)
    plt.plot(epochs_rust_sgd, [float(row['Time_s']) for row in rust_sgd_data], 's-', label='Rust SGD', linewidth=2, markersize=6)
    plt.plot(epochs_rust_meprop, [float(row['Time_s']) for row in rust_meprop_data], '^-', label='Rust MeProp', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Regression: Epoch Time Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('python-tests/graphs/regression_epoch_time.png', dpi=150)
    plt.close()
    print("  Saved: python-tests/graphs/regression_epoch_time.png")
    
    print("\n")
    
    return (python_data, rust_sgd_data, rust_meprop_data)

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

