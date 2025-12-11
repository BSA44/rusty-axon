# Benchmark Comparison: Micrograd (Python) vs Rusty-Axon (Rust)

*Generated: 2025-12-11 07:51:21*

---

## Classification Benchmark (Diabetes/Pima Indians)

### Final Epoch Metrics


| Metric | Python (Micrograd) | Rust SGD | Rust MeProp | Rust SGD Rasp. Pi | Rust MeProp Rasp. Pi | Diff vs Python (Micrograd) |
|---|---|---|---|---|---|---|
| Train Loss | 0.383098 | 0.243230 | 0.232257 | 0.236532 | 0.247883 | -0.139868 / -0.150841 / -0.146566 / -0.135215 |
| Train Accuracy | 0.768730 | 0.757329 | 0.781759 | 0.778502 | 0.773616 | -0.011401 / +0.013029 / +0.009772 / +0.004886 |
| Test Loss | 0.416294 | 0.245174 | 0.264327 | 0.237936 | 0.270961 | -0.171120 / -0.151967 / -0.178358 / -0.145333 |
| Test Accuracy | 0.753247 | 0.740260 | 0.753247 | 0.772727 | 0.759740 | -0.012987 / +0.000000 / +0.019480 / +0.006493 |
| F1 Score | 0.586957 | 0.726027 | 0.577778 | 0.615385 | 0.610526 | +0.139070 / -0.009179 / +0.028428 / +0.023569 |
### Performance Metrics

| Metric | Python (Micrograd) | Rust SGD | Speedup | Rust MeProp | Speedup | Rust SGD Rasp. Pi | Speedup | Rust MeProp Rasp. Pi | Speedup |
|---|---|---|---|---|---|---|---|---|---|
| Total Training Time (s) | 35.31 | 2.31 | 15.31x | 2.31 | 15.27x | 4.74 | 7.44x | 4.80 | 7.36x
| Avg Time per Epoch (s) | 0.71 | 0.05 | 15.31x | 0.05 | 15.27x | 0.09 | 7.44x | 0.10 | 7.36x
| Avg CPU Usage (%) | 13.2 | 14.8 | - | 15.6 | - | 31.6 | - | 32.4 | - |
| Avg RAM Usage (%) | 19.5 | 16.7 | - | 16.7 | - | 10.2 | - | 10.2 | - |

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


| Metric | Python (Micrograd) | Rust SGD | Rust MeProp | Rust SGD Rasp. Pi | Rust MeProp Rasp. Pi | Diff vs Python (Micrograd) |
|---|---|---|---|---|---|---|
| Loss (MSE) | 0.001892 | 0.000277 | 0.000020 | 0.000079 | 0.000019 | -0.001615 / -0.001872 / -0.001813 / -0.001873 |
| RMSE | 0.043498 | 0.016608 | 0.004503 | 0.008634 | 0.004799 | -0.026890 / -0.038995 / -0.034864 / -0.038699 |
### Performance Metrics

| Metric | Python (Micrograd) | Rust SGD | Speedup | Rust MeProp | Speedup | Rust SGD Rasp. Pi | Speedup | Rust MeProp Rasp. Pi | Speedup |
|---|---|---|---|---|---|---|---|---|---|
| Total Training Time (s) | 42.09 | 3.00 | 14.03x | 3.00 | 14.03x | 9.00 | 4.68x | 9.20 | 4.57x
| Avg Time per Epoch (s) | 4.21 | 0.30 | 14.03x | 0.30 | 14.03x | 0.90 | 4.68x | 0.92 | 4.57x
| Avg CPU Usage (%) | 11.8 | 12.8 | - | 11.8 | - | 26.1 | - | 26.1 | - |
| Avg RAM Usage (%) | 19.4 | 16.7 | - | 16.8 | - | 11.7 | - | 11.7 | - |

### Generated Plots

- `regression_loss.png`
- `regression_rmse.png`
- `regression_cpu_usage.png`
- `regression_ram_usage.png`
- `regression_epoch_time.png`

---

