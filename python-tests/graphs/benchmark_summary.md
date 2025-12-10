# Benchmark Comparison: Micrograd (Python) vs Rusty-Axon (Rust)

<<<<<<< HEAD
*Generated: 2025-12-10 21:54:00*
=======
*Generated: 2025-12-10 17:39:32*
>>>>>>> master

---

## Classification Benchmark (Diabetes/Pima Indians)

### Final Epoch Metrics (Epoch 50)

| Metric | Python (Micrograd) | Rust SGD | Rust MeProp | Diff (SGD) | Diff (MeProp) |
|--------|-------------------|----------|-------------|------------|---------------|
| Train Loss | 0.383098 | 0.243230 | 0.232257 | -0.139868 | -0.150841 |
| Train Accuracy | 0.768730 | 0.757329 | 0.781759 | -0.011401 | +0.013029 |
| Test Loss | 0.416294 | 0.245174 | 0.264327 | -0.171120 | -0.151967 |
| Test Accuracy | 0.753247 | 0.740260 | 0.753247 | -0.012987 | +0.000000 |
| F1 Score | 0.586957 | 0.726027 | 0.577778 | +0.139070 | -0.009179 |

### Performance Metrics

| Metric | Python | Rust SGD | Rust MeProp | Speedup SGD | Speedup MeProp |
|--------|--------|----------|-------------|-------------|----------------|
| **Total Training Time** | 35.31s | 2.31s | 2.31s | **15.31x** | **15.27x** |
| Avg Time per Epoch | 0.71s | 0.05s | 0.05s | 15.31x | 15.27x |
| Avg CPU Usage | 13.2% | 14.8% | 15.6% | - | - |
| Avg RAM Usage | 19.5% | 16.7% | 16.7% | - | - |

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
| Loss (MSE) | 0.001892 | 0.000277 | 0.000020 | -0.001615 | -0.001872 |
| RMSE | 0.043498 | 0.016608 | 0.004503 | -0.026890 | -0.038995 |

### Performance Metrics

| Metric | Python | Rust SGD | Rust MeProp | Speedup SGD | Speedup MeProp |
|--------|--------|----------|-------------|-------------|----------------|
| **Total Training Time** | 42.09s | 3.00s | 3.00s | **14.03x** | **14.03x** |
| Avg Time per Epoch | 4.21s | 0.30s | 0.30s | 14.03x | 14.03x |
| Avg CPU Usage | 11.8% | 12.8% | 11.8% | - | - |
| Avg RAM Usage | 19.4% | 16.7% | 16.8% | - | - |

### Generated Plots

- `regression_loss.png` - MSE loss comparison
- `regression_rmse.png` - RMSE comparison
- `regression_cpu_usage.png` - CPU usage over epochs
- `regression_ram_usage.png` - RAM usage over epochs
- `regression_epoch_time.png` - Time per epoch

---

