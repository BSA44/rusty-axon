import csv
import random
import time
import psutil
from micrograd.engine import Value
from micrograd.nn import MLP

# ------------------------ Micrograd exp approximation ------------------------
def micrograd_exp(v: Value):
    base = Value(1.0) + v / 1024
    out = base
    for _ in range(10):
        out = out * out
    return out

# ------------------------ Micrograd log approximation ------------------------
def log_approx(v: Value):
    t = v - Value(1.0)
    res = t - (t*t)/2 + (t*t*t)/3
    return res

Value.log_approx = log_approx

# ------------------------ Sigmoid ------------------------
def sigmoid(v: Value):
    if v.data < -100: v = Value(-100)
    if v.data > 100:  v = Value(100)
    base = Value(1.0) + (-v) / 1024
    exp_neg_v = base
    for _ in range(10):
        exp_neg_v = exp_neg_v * exp_neg_v
    return Value(1.0) / (Value(1.0) + exp_neg_v)

# ------------------------ Binary Cross Entropy ------------------------
def binary_cross_entropy(logits, true_label):
    z = logits[1] - logits[0]
    y = Value(true_label)
    s = sigmoid(z)
    eps = 1e-8
    log_s = (s + Value(eps)).log_approx()
    log_1_s = (Value(1.0)-s + Value(eps)).log_approx()
    loss = -(y * log_s + (Value(1.0)-y) * log_1_s)
    return loss

# ------------------------ Load Pima Dataset ------------------------
def load_pima(path):
    X, y = [], []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            try:
                features = list(map(float, row[:-1]))
                label = int(float(row[-1]))
            except ValueError:
                continue
            X.append(features)
            y.append(label)
    return X, y

# ------------------------ Normalize ------------------------
def normalize_features(X):
    max_vals = [17, 200, 122, 99, 846, 67.1, 2.42, 100]
    X_norm = [[x/m for x,m in zip(row,max_vals)] for row in X]
    return X_norm

# ------------------------ Prediction ------------------------
def predict(model, x):
    logits = model([Value(v) for v in x])
    return 1 if logits[1].data > logits[0].data else 0

# ------------------------ F1 Score ------------------------
def f1_score(y_true, y_pred):
    tp = sum(int(yt==1 and yp==1) for yt, yp in zip(y_true, y_pred))
    fp = sum(int(yt==0 and yp==1) for yt, yp in zip(y_true, y_pred))
    fn = sum(int(yt==1 and yp==0) for yt, yp in zip(y_true, y_pred))
    
    # Avoid division by zero
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

# ------------------------ Train/Test Split ------------------------
def train_test_split(X, y, test_ratio=0.2):
    data = list(zip(X, y))
    random.shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    train = data[:split]
    test = data[split:]
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    return list(X_train), list(y_train), list(X_test), list(y_test)

# ------------------------ MAIN ------------------------
if __name__ == "__main__":
    # Load dataset
    X, y = load_pima("dataset.csv")
    X = normalize_features(X)

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
    print("Train samples:", len(X_train), "Test samples:", len(X_test))

    # MLP: 8 -> 16 -> 2
    model = MLP(len(X_train[0]), [8, 4])
    lr = 0.01
    epochs = 50
    batch_size = 32

    # Prepare CSV
    csv_file = open("training_metrics.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["Epoch","Train_Loss","Train_Acc","Test_Loss","Test_Acc","F1","Epoch_Time","CPU_Usage","RAM_Usage"])

    start_total = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        combined = list(zip(X_train, y_train))
        random.shuffle(combined)

        # Batch training
        for i in range(0, len(combined), batch_size):
            batch = combined[i:i+batch_size]
            # zero grads
            for p in model.parameters():
                p.grad = 0.0
            batch_loss = Value(0.0)
            for feat, label in batch:
                logits = model([Value(v) for v in feat])
                loss = binary_cross_entropy(logits, label)
                batch_loss += loss
            batch_loss.backward()
            for p in model.parameters():
                p.data -= lr * p.grad
            total_loss += batch_loss.data
        train_loss = total_loss / len(X_train)

        # Train accuracy
        y_train_pred = [predict(model, x) for x in X_train]
        train_acc = sum(yp==yt for yp, yt in zip(y_train_pred, y_train)) / len(y_train)

        # Evaluate test loss
        test_loss = 0.0
        for x, label in zip(X_test, y_test):
            logits = model([Value(v) for v in x])
            test_loss += binary_cross_entropy(logits, label).data
        test_loss /= len(X_test)

        # Evaluate test metrics
        y_pred = [predict(model, x) for x in X_test]
        test_acc = sum(yp==yt for yp, yt in zip(y_pred, y_test))/len(y_test)
        f1 = f1_score(y_test, y_pred)

        epoch_time = time.time() - epoch_start
        cpu_usage = psutil.cpu_percent(interval=None)
        ram_usage = psutil.virtual_memory().percent

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
              f"F1: {f1:.4f} | Time: {epoch_time:.2f}s | CPU: {cpu_usage:.1f}% | RAM: {ram_usage:.1f}%")

        # Save to CSV
        writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc, f1, epoch_time, cpu_usage, ram_usage])

    csv_file.close()
    print("Total training time:", time.time() - start_total)
    print("Metrics saved to training_metrics.csv")
