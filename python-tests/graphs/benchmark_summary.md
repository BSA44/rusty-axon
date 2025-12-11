# Benchmark Comparison: Micrograd (Python) vs Rusty-Axon (Rust)

*Generated: 2025-12-10 22:35:16*

---

## Classification Benchmark (Diabetes/Pima Indians)

### Final Epoch Metrics


| Metric | Python (Micrograd) | Rust SGD | Rust MeProp | Diff vs Python (Micrograd) |
|---|---|---|---|---|
| Train Loss | 0.383098 | 0.243230 | 0.232257 | -0.139868 / -0.150841 |
| Train Accuracy | 0.768730 | 0.757329 | 0.781759 | -0.011401 / +0.013029 |
| Test Loss | 0.416294 | 0.245174 | 0.264327 | -0.171120 / -0.151967 |
| Test Accuracy | 0.753247 | 0.740260 | 0.753247 | -0.012987 / +0.000000 |
| F1 Score | 0.586957 | 0.726027 | 0.577778 | +0.139070 / -0.009179 |
### Performance Metrics

| Metric | Python (Micrograd) | Rust SGD | Speedup | Rust MeProp | Speedup |
|---|---|---|---|---|---|
| Total Training Time (s) | 35.31 | 2.31 | 15.31x | 2.31 | 15.27x
| Avg Time per Epoch (s) | 0.71 | 0.05 | 15.31x | 0.05 | 15.27x
| Avg CPU Usage (%) | 13.2 | 14.8 | - | 15.6 | - |
| Avg RAM Usage (%) | 19.5 | 16.7 | - | 16.7 | - |

### Generated Plots

- `classification_train_loss.png`
- `classification_test_loss.png`
- `classification_train_accuracy.png`
- `classification_test_accuracy.png`
- `classification_f1_score.png`
- `classification_cpu_usage.png`
- `classification_ram_usage.png`
- `classification_epoch_time.png`

---

## Regression Benchmark (California Housing)

### Final Epoch Metrics


| Metric | Python (Micrograd) | Rust SGD | Rust MeProp | Diff vs Python (Micrograd) |
|---|---|---|---|---|
| Loss (MSE) | 0.001892 | 0.000277 | 0.000020 | -0.001615 / -0.001872 |
| RMSE | 0.043498 | 0.016608 | 0.004503 | -0.026890 / -0.038995 |
### Performance Metrics

| Metric | Python (Micrograd) | Rust SGD | Speedup | Rust MeProp | Speedup |
|---|---|---|---|---|---|
| Total Training Time (s) | 42.09 | 3.00 | 14.03x | 3.00 | 14.03x
| Avg Time per Epoch (s) | 4.21 | 0.30 | 14.03x | 0.30 | 14.03x
| Avg CPU Usage (%) | 11.8 | 12.8 | - | 11.8 | - |
| Avg RAM Usage (%) | 19.4 | 16.7 | - | 16.8 | - |

### Generated Plots

- `regression_loss.png`
- `regression_rmse.png`
- `regression_cpu_usage.png`
- `regression_ram_usage.png`
- `regression_epoch_time.png`

---

