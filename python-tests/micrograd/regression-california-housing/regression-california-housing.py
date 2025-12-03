import csv
import math
import time
import psutil
import numpy as np
from micrograd.engine import Value
from micrograd.nn import MLP

# -----------------------------
# Sigmoid helper
# -----------------------------
def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

# -----------------------------
# Load CSV dataset
# -----------------------------
def load_csv(path, limit=None):
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            # numeric features only (skip ocean_proximity)
            features = [
                row["longitude"],
                row["latitude"],
                row["housing_median_age"],
                row["total_rooms"],
                row["total_bedrooms"],
                row["population"],
                row["households"],
                row["median_income"]
            ]
            target = row["median_house_value"]
            data.append(features + [target])

    data = np.array(data, dtype=object)
    data[data == ''] = np.nan
    data = data.astype(np.float32)

    # Replace NaN with column mean
    col_mean = np.nanmean(data[:, :-1], axis=0)
    inds = np.where(np.isnan(data[:, :-1]))
    data[inds[0], inds[1]] = np.take(col_mean, inds[1])

    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)

    # Apply sigmoid normalization to scale everything 0-1
    X = sigmoid(X)
    y = sigmoid(y)

    return X, y

# -----------------------------
# MSE Loss
# -----------------------------
def mse_loss(preds, targets):
    batch_size = len(preds)
    loss = Value(0.0)
    for p, t in zip(preds, targets):
        loss += (p - t[0])**2
    return loss / batch_size

# -----------------------------
# Micrograd model wrapper
# -----------------------------
class MicrogradRegressor:
    def __init__(self, input_dim=8, hidden=[16, 8], output_dim=1):
        self.model = MLP(input_dim, hidden + [output_dim])

    def forward(self, X):
        out = []
        for x in X:
            vals = self.model(x.tolist())
            if isinstance(vals, Value):
                out.append(vals)
            else:
                out.append(vals[0])
        return out

    def backward(self, X, y, lr):
        for p in self.model.parameters():
            p.grad = 0.0
        preds = self.forward(X)
        loss = mse_loss(preds, y)
        loss.backward()
        for p in self.model.parameters():
            p.data -= lr * p.grad
        return loss.data

# -----------------------------
# Training loop with CSV logging
# -----------------------------
def train(path, epochs=10, batch_size=32, lr=0.01, limit=None, csv_file="training_metrics.csv"):
    X, y = load_csv(path, limit)
    n = len(X)
    model = MicrogradRegressor()

    # Prepare CSV file
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "RMSE", "CPU%", "RAM%", "Time_s"])

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        perm = np.random.permutation(n)
        losses = []

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            Xb = X[idx]
            yb = y[idx]

            loss = model.backward(Xb, yb, lr)
            losses.append(loss)

            # Measure CPU and RAM
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent

        rmse = math.sqrt(np.mean([l for l in losses]))
        epoch_time = time.time() - t0
        mean_loss = np.mean(losses)

        print(f"Epoch {epoch}/{epochs} | Loss={mean_loss:.6f} | RMSE={rmse:.6f} | CPU={cpu:.1f}% | RAM={ram:.1f}% | Time={epoch_time:.1f}s")

        # Append metrics to CSV
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, mean_loss, rmse, cpu, ram, epoch_time])

    return model

# -----------------------------
if __name__ == "__main__":
    model = train("dataset.csv", epochs=5, batch_size=64, lr=0.01, limit=2000)
