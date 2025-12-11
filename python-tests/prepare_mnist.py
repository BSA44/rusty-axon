#!/usr/bin/env python3
"""
Prepare MNIST dataset for rusty-axon demo.
Downloads MNIST with multiple fallback sources, selects subset, saves to CSV.

Output format:
- First column: label (0-9)
- Remaining 784 columns: pixel values normalized to [0, 1]
"""

import os
import numpy as np
import gzip
import struct
import urllib.request
import time

def download_file(url, filepath, retries=3):
    """Download file with retry logic"""
    for attempt in range(retries):
        try:
            print(f"    Downloading from {url}...")
            urllib.request.urlretrieve(url, filepath)
            return True
        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return False

def load_mnist_images(filepath):
    """Load MNIST image file (IDX format)"""
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
    return images

def load_mnist_labels(filepath):
    """Load MNIST label file (IDX format)"""
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def download_mnist_direct():
    """Download MNIST directly from multiple mirror sources"""
    # Create cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), "mnist_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Multiple mirror URLs
    mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",  # PyTorch mirror
        "http://yann.lecun.com/exdb/mnist/",               # Original
        "https://storage.googleapis.com/cvdf-datasets/mnist/",  # Google mirror
    ]
    
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    
    downloaded = {}
    
    for key, filename in files.items():
        filepath = os.path.join(cache_dir, filename)
        downloaded[key] = filepath
        
        # Skip if already cached
        if os.path.exists(filepath):
            print(f"    Using cached {filename}")
            continue
        
        # Try each mirror
        success = False
        for mirror in mirrors:
            url = mirror + filename
            if download_file(url, filepath, retries=2):
                success = True
                break
        
        if not success:
            raise RuntimeError(f"Failed to download {filename} from any mirror")
    
    # Load the data
    print("[+] Loading downloaded files...")
    X_train = load_mnist_images(downloaded["train_images"])
    y_train = load_mnist_labels(downloaded["train_labels"])
    X_test = load_mnist_images(downloaded["test_images"])
    y_test = load_mnist_labels(downloaded["test_labels"])
    
    # Combine train and test for balanced sampling
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])
    
    return X, y

def download_mnist_keras():
    """Fallback: use Keras/TensorFlow"""
    try:
        print("[+] Trying Keras/TensorFlow...")
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        return X, y
    except ImportError:
        return None, None

def download_mnist():
    """Download MNIST with fallback sources"""
    print("[+] Downloading MNIST dataset...")
    
    # Try direct download first (most reliable)
    try:
        X, y = download_mnist_direct()
        if X is not None:
            return X, y
    except Exception as e:
        print(f"    Direct download failed: {e}")
    
    # Fallback to Keras
    X, y = download_mnist_keras()
    if X is not None:
        return X, y
    
    raise RuntimeError("Could not download MNIST from any source")

def prepare_balanced_subset(X, y, samples_per_class, seed=42):
    """Select balanced subset with equal samples per class"""
    np.random.seed(seed)
    
    indices = []
    for digit in range(10):
        digit_indices = np.where(y == digit)[0]
        selected = np.random.choice(digit_indices, size=samples_per_class, replace=False)
        indices.extend(selected)
    
    # Shuffle the selected indices
    np.random.shuffle(indices)
    
    return X[indices], y[indices]

def save_to_csv(X, y, filename):
    """Save dataset to CSV: label, pixel_0, pixel_1, ..., pixel_783"""
    print(f"[+] Saving {len(y)} samples to {filename}...")
    
    # Normalize pixels to [0, 1]
    X_normalized = X / 255.0
    
    # Combine label and pixels
    data = np.column_stack([y, X_normalized])
    
    # Create header
    header = "label," + ",".join([f"pixel_{i}" for i in range(784)])
    
    # Save with full precision for normalized values
    np.savetxt(filename, data, delimiter=',', header=header, comments='', fmt='%g')
    
    print(f"    Shape: {X.shape}, Labels: {np.unique(y)}")

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "mnist")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("MNIST Dataset Preparation for rusty-axon")
    print("=" * 50)
    
    # Download full MNIST
    X, y = download_mnist()
    print(f"[+] Full dataset: {X.shape[0]} images, {X.shape[1]} pixels each")
    
    # Prepare training set: 800 images (80 per class)
    X_train, y_train = prepare_balanced_subset(X, y, samples_per_class=80, seed=42)
    
    # Prepare test set: 200 images (20 per class) with different seed
    X_test, y_test = prepare_balanced_subset(X, y, samples_per_class=20, seed=123)
    
    # Save to CSV
    save_to_csv(X_train, y_train, os.path.join(output_dir, "mnist_train.csv"))
    save_to_csv(X_test, y_test, os.path.join(output_dir, "mnist_test.csv"))
    
    print()
    print("=" * 50)
    print("Dataset ready!")
    print(f"  Training: mnist/mnist_train.csv (800 samples)")
    print(f"  Testing:  mnist/mnist_test.csv (200 samples)")
    print("=" * 50)
    
    # Show sample distribution
    print()
    print("Label distribution (training):")
    for digit in range(10):
        count = np.sum(y_train == digit)
        print(f"  Digit {digit}: {count} samples")

if __name__ == "__main__":
    main()
