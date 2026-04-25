# Benchmark Comparison: Micrograd (Python) vs Rusty-Axon (Rust)

This document describes how to run head-to-head performance comparisons between the Python micrograd implementation and the Rust rusty-axon implementation.

## 📊 Benchmarks

### 1. Classification: Diabetes/Pima Indians Dataset
- **Dataset**: Pima Indians Diabetes (768 samples, 8 features)
- **Architecture**: 8 → 8 → 4 → 2 (binary classification)
- **Loss**: Binary Cross-Entropy
- **Metrics**: Train/Test Loss, Train/Test Accuracy, F1 Score, Time, CPU%
- **Training**: 50 epochs, batch size 32, learning rate 0.01

### 2. Regression: California Housing Dataset
- **Dataset**: California Housing (2000 samples, 8 features)
- **Architecture**: 8 → 16 → 8 → 1
- **Loss**: Mean Squared Error (MSE)
- **Metrics**: Loss, RMSE, CPU%, RAM%, Time
- **Training**: 5 epochs, batch size 64, learning rate 0.01

## 🚀 Running the Benchmarks

### Python (Micrograd)

```bash
# Classification
cd python-tests/micrograd/classification-diabetes
python classification-diabetes.py

# Regression
cd python-tests/micrograd/regression-california-housing
python regression-california-housing.py
```

**Output**: `training_metrics.csv` in each directory

### Rust (Rusty-Axon)

```bash
# Classification
cargo run --release --example bench_classification_diabetes

# Regression
cargo run --release --example bench_regression_housing
```

**Output**: 
- `examples/rust_classification_metrics.csv`
- `examples/rust_regression_metrics.csv`

## 📈 Comparing Results

### Key Metrics to Compare

#### Classification (Diabetes)
- **Training convergence**: Train/Test Loss curves
- **Model quality**: Final Test Accuracy and F1 Score
- **Performance**: Time per epoch, Total training time
- **Resource usage**: CPU%

#### Regression (Housing)
- **Training convergence**: Loss and RMSE curves
- **Model quality**: Final RMSE
- **Performance**: Time per epoch, Total training time
- **Resource usage**: CPU%, RAM%

### Expected Differences

1. **Random Initialization**: 
   - Different random seeds → different initial weights
   - Results will vary but should converge to similar accuracy

2. **Performance**:
   - Rust should be **5-10x faster** per epoch
   - Lower CPU usage expected in Rust

3. **Numerical Precision**:
   - Minor differences due to:
     - Floating point arithmetic order
     - Different exp/log implementations
     - Sigmoid approximations

## 🔬 Analysis Tips

1. **Plot the curves** from both CSV files to visualize convergence
2. **Compare final metrics** (last epoch values)
3. **Measure total time** to assess speed advantage
4. **Check stability** - both should converge smoothly

## 📝 Notes

- Use `--release` flag for Rust (crucial for fair comparison)
- Run Python with optimized settings if available
- Ensure same dataset files are used for both
- Close other applications to reduce CPU/RAM noise

## 🎯 Success Criteria

✅ Both implementations should:
- Converge to similar accuracy levels
- Show smooth loss curves (no instability)
- Produce similar final metrics (±5%)

✅ Rust should demonstrate:
- Faster training time
- Lower or similar CPU usage
- Type safety and memory safety guarantees

