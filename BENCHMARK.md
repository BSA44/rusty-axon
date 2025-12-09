# Benchmark: Micrograd (Python) vs Rusty-Axon (Rust)

## Quick Run

### Python (Micrograd)
```bash
cd python-tests/micrograd/classification-diabetes
python classification-diabetes.py

cd ../regression-california-housing
python regression-california-housing.py
```

### Rust (Rusty-Axon)
```bash
cargo run --release --example bench_classification_diabetes_sgd
cargo run --release --example bench_classification_diabetes_meprop
cargo run --release --example bench_regression_housing_sgd
cargo run --release --example bench_regression_housing_meprop
```

⚠️ **Use `--release`** - debug builds are 10-100x slower!

## Results

| Benchmark | Dataset | Arch | Metric |
|-----------|---------|------|--------|
| **Classification** | Diabetes (768) | 8→8→4→2 | Accuracy, F1 |
| **Regression** | Housing (2000) | 8→16→8→1 | MSE, RMSE |

CSV outputs:
- Python: `python-tests/micrograd/{benchmark}/training_metrics.csv`
- Rust: `examples/rust_{benchmark}_metrics.csv`

## Compare Results

Run the comparison script to see side-by-side metrics and generate plots:

```bash
python compare_results.py
```

This will:
- Print comparison tables for both benchmarks
- Generate plots in `python-tests/graphs/`:
  - Loss curves (train/test)
  - Accuracy curves
  - F1 scores
  - CPU usage over time
  - RAM usage over time
  - Epoch time comparison
  - Create a comparison summary in `python-tests/graphs/benchmark_summary.md`

**Note:** Requires `matplotlib`: `pip install matplotlib`
