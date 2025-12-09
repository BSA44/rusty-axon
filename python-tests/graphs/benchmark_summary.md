# Benchmark Comparison: Micrograd (Python) vs Rusty-Axon (Rust)

*Generated: 2025-12-09 22:55:29*

---

## Classification Benchmark (Diabetes/Pima Indians)

### Final Epoch Metrics (Epoch 50)

| Metric | Python (Micrograd) | Rust SGD | Rust MeProp | Diff (SGD) | Diff (MeProp) |
|--------|-------------------|----------|-------------|------------|---------------|
| Train Loss | 0.376764 | 0.241679 | 0.229542 | -0.135085 | -0.147222 |
| Train Accuracy | 0.791531 | 0.781759 | 0.781759 | -0.009772 | -0.009772 |
| Test Loss | 0.430995 | 0.245882 | 0.251990 | -0.185113 | -0.179005 |
| Test Accuracy | 0.740260 | 0.779221 | 0.772727 | +0.038961 | +0.032467 |
| F1 Score | 0.600000 | 0.613636 | 0.615385 | +0.013636 | +0.015385 |

### Performance Metrics

| Metric | Python | Rust SGD | Rust MeProp | Speedup SGD | Speedup MeProp |
|--------|--------|----------|-------------|-------------|----------------|
| **Total Training Time** | 52.28s | 5.38s | 5.61s | **9.73x** | **9.33x** |
| Avg Time per Epoch | 1.05s | 0.11s | 0.11s | 9.73x | 9.33x |
| Avg CPU Usage | 12.8% | 14.2% | 14.5% | - | - |
| Avg RAM Usage | 65.5% | 65.9% | 67.8% | - | - |

### Generated Plots

- `classification_train_loss.png` - Train loss comparison
- `classification_test_loss.png` - Test loss comparison
- `classification_train_accuracy.png` - Train accuracy comparison
- `classification_test_accuracy.png` - Test accuracy comparison
- `classification_f1_score.png` - F1 score comparison
- `classification_cpu_usage.png` - CPU usage over epochs
- `classification_ram_usage.png` - RAM usage over epochs
- `classification_epoch_time.png` - Time per epoch

---

## Regression Benchmark (California Housing)

### Final Epoch Metrics (Epoch 5)

| Metric | Python (Micrograd) | Rust SGD | Rust MeProp | Diff (SGD) | Diff (MeProp) |
|--------|-------------------|----------|-------------|------------|---------------|
| Loss (MSE) | 0.000109 | 0.000169 | 0.000288 | +0.000060 | +0.000179 |
| RMSE | 0.010433 | 0.012908 | 0.016519 | +0.002475 | +0.006086 |

### Performance Metrics

| Metric | Python | Rust SGD | Rust MeProp | Speedup SGD | Speedup MeProp |
|--------|--------|----------|-------------|-------------|----------------|
| **Total Training Time** | 96.84s | 5.00s | 5.10s | **19.37x** | **18.99x** |
| Avg Time per Epoch | 9.68s | 0.50s | 0.51s | 19.37x | 18.99x |
| Avg CPU Usage | 21.2% | 15.4% | 15.2% | - | - |
| Avg RAM Usage | 66.1% | 66.3% | 67.7% | - | - |

### Generated Plots

- `regression_loss.png` - MSE loss comparison
- `regression_rmse.png` - RMSE comparison
- `regression_cpu_usage.png` - CPU usage over epochs
- `regression_ram_usage.png` - RAM usage over epochs
- `regression_epoch_time.png` - Time per epoch

---

## Key Takeaways

- **Performance**: Rust is 5-10x faster due to compiled, optimized code
- **Model Quality**: Accuracy/loss metrics are similar (±5% variation expected)
- **Differences**: Minor variations due to random initialization
- **Safety**: Rust provides type safety & memory safety guarantees
- **Framework**: Rusty-Axon properly implements Loss and Optimizer traits

---

*Built with Rust for performance and education.*
