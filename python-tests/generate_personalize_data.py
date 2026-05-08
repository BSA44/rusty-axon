#!/usr/bin/env python3
"""
Generate "personalized" MNIST data for the on-device fine-tune demo (Phase 11).

The base classifier is trained on un-augmented MNIST. To simulate a single
end-user whose handwriting drifts from the training-set average, we apply a
*fixed* affine + photometric transform to a held-out subset of MNIST test
images. The user persona is fully described by USER_PROFILE below
(rotation, translation, brightness, contrast, jitter); change the seed or
profile to simulate a different user.

Outputs three CSVs at python-tests/mnist/:
  mnist_personalize_train.csv  - 200 augmented samples (fine-tune set)
  mnist_personalize_test.csv   - 500 augmented samples (eval set)
  mnist_personalize_clean.csv  - same 500 indices, un-augmented, for the
                                 domain-shift baseline row in the paper

Plus a preview image at python-tests/mnist/personalize_preview.png with
N_PREVIEW clean-vs-augmented sample pairs so you can eyeball the persona.

CSV format matches prepare_mnist.py:
  label, pixel_0, pixel_1, ..., pixel_783    (pixels normalized to [0, 1])

Requires: numpy, scipy, matplotlib.
"""

import os
import sys
import gzip
import struct
import numpy as np
from scipy.ndimage import rotate as ndi_rotate
from scipy.ndimage import shift as ndi_shift
import matplotlib.pyplot as plt


USER_PROFILE = {
    "name":             "user_a",
    "rotation_deg":     -9.0,        # consistent pen tilt
    "shift_xy_px":      (2.5, 0.5),  # (col, row) translation in pixels
    "brightness":        0.90,       # darker overall
    "contrast":          1.10,       # slightly higher contrast
    "noise_std":         0.02,       # per-sample Gaussian noise on normalized pixels
    "jitter_rot_deg":    1.5,        # per-sample rotation jitter (sd)
    "jitter_shift_px":   0.5,        # per-sample shift jitter (sd)
}
SEED       = 7
N_TRAIN    = 200
N_TEST     = 500
N_PREVIEW  = 8   # samples shown in the preview PNG


def load_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)


def load_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def augment_one(img_uint8, rng, profile):
    """Apply the user's affine + photometric transform with small per-sample jitter.

    Input:  img_uint8 -- (28, 28) uint8 in [0, 255]
    Output: (28, 28) float32 in [0, 1]
    """
    rot = profile["rotation_deg"] + rng.normal(0.0, profile["jitter_rot_deg"])
    sx  = profile["shift_xy_px"][0] + rng.normal(0.0, profile["jitter_shift_px"])
    sy  = profile["shift_xy_px"][1] + rng.normal(0.0, profile["jitter_shift_px"])

    x = img_uint8.astype(np.float32) / 255.0
    x = ndi_rotate(x, angle=rot, reshape=False, order=1, mode="constant", cval=0.0)
    x = ndi_shift(x, shift=(sy, sx), order=1, mode="constant", cval=0.0)

    # Contrast around the 0.5 midpoint, then brightness scale.
    x = (x - 0.5) * profile["contrast"] + 0.5
    x = x * profile["brightness"]

    x = x + rng.normal(0.0, profile["noise_std"], size=x.shape).astype(np.float32)
    return np.clip(x, 0.0, 1.0)


def stratified_pool(labels, total, rng):
    """Return `total` indices balanced across the 10 classes."""
    per_class = total // 10
    chunks = []
    for d in range(10):
        idx = np.where(labels == d)[0]
        chunks.append(rng.choice(idx, size=per_class, replace=False))
    out = np.concatenate(chunks)
    rng.shuffle(out)
    return out


def to_csv(images_norm, labels, path):
    """images_norm: (N, 784) float32 in [0,1]. labels: (N,) uint8."""
    data = np.column_stack([labels.astype(np.float32), images_norm])
    header = "label," + ",".join(f"pixel_{i}" for i in range(784))
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%g")
    print(f"  wrote {path} ({len(labels)} samples)")


def save_preview(clean, augmented, labels, profile, path):
    """Side-by-side preview: top row = clean, bottom row = augmented."""
    n = len(labels)
    fig, axes = plt.subplots(2, n, figsize=(1.4 * n, 3.2))
    for i in range(n):
        axes[0, i].imshow(clean[i].reshape(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, i].set_title(f"{labels[i]}", fontsize=10)
        axes[0, i].axis("off")
        axes[1, i].imshow(augmented[i].reshape(28, 28), cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("clean", fontsize=9)
    axes[1, 0].set_ylabel("augmented", fontsize=9)
    suptitle = (
        f"persona '{profile['name']}': "
        f"rot {profile['rotation_deg']:+.1f} deg, "
        f"shift {profile['shift_xy_px']} px, "
        f"brightness x{profile['brightness']}, "
        f"contrast x{profile['contrast']}"
    )
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  wrote {path} ({n} sample pairs)")


def main():
    here    = os.path.dirname(os.path.abspath(__file__))
    cache   = os.path.join(here, "mnist_cache")
    out_dir = os.path.join(here, "mnist")
    os.makedirs(out_dir, exist_ok=True)

    img_path = os.path.join(cache, "t10k-images-idx3-ubyte.gz")
    lbl_path = os.path.join(cache, "t10k-labels-idx1-ubyte.gz")
    if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
        sys.exit(
            f"MNIST test set not found in {cache}.\n"
            f"  Run python-tests/prepare_mnist.py first to populate the cache."
        )

    print(f"[+] Loading MNIST test images from {cache}")
    images = load_idx_images(img_path)  # (10000, 28, 28)
    labels = load_idx_labels(lbl_path)  # (10000,)

    p = USER_PROFILE
    print(f"[+] User persona: {p['name']}  "
          f"rot={p['rotation_deg']} deg  "
          f"shift={p['shift_xy_px']} px  "
          f"brightness x{p['brightness']}  "
          f"contrast x{p['contrast']}  "
          f"noise sigma={p['noise_std']}")

    rng = np.random.default_rng(SEED)

    # One stratified pool, then split — guarantees no overlap between train and test.
    pool      = stratified_pool(labels, N_TRAIN + N_TEST, rng)
    train_idx = pool[:N_TRAIN]
    test_idx  = pool[N_TRAIN:]

    print(f"[+] Augmenting {N_TRAIN} train + {N_TEST} test samples")
    train_aug = np.empty((N_TRAIN, 784), dtype=np.float32)
    for i, k in enumerate(train_idx):
        train_aug[i] = augment_one(images[k], rng, p).reshape(-1)

    test_aug = np.empty((N_TEST, 784), dtype=np.float32)
    for i, k in enumerate(test_idx):
        test_aug[i] = augment_one(images[k], rng, p).reshape(-1)

    test_clean = (images[test_idx].astype(np.float32) / 255.0).reshape(N_TEST, 784)

    to_csv(train_aug,  labels[train_idx], os.path.join(out_dir, "mnist_personalize_train.csv"))
    to_csv(test_aug,   labels[test_idx],  os.path.join(out_dir, "mnist_personalize_test.csv"))
    to_csv(test_clean, labels[test_idx],  os.path.join(out_dir, "mnist_personalize_clean.csv"))

    # Preview: take the first N_PREVIEW samples of the test pool with diverse labels if possible.
    preview_idx = []
    seen = set()
    for i, lab in enumerate(labels[test_idx]):
        if lab not in seen:
            preview_idx.append(i)
            seen.add(int(lab))
        if len(preview_idx) == N_PREVIEW:
            break
    while len(preview_idx) < N_PREVIEW:  # fall back if fewer than 8 unique classes appeared first
        for i in range(N_TEST):
            if i not in preview_idx:
                preview_idx.append(i)
                if len(preview_idx) == N_PREVIEW:
                    break

    save_preview(
        clean=test_clean[preview_idx],
        augmented=test_aug[preview_idx],
        labels=labels[test_idx][preview_idx],
        profile=p,
        path=os.path.join(out_dir, "personalize_preview.png"),
    )

    print()
    print("Done. Files for the Phase 11 personalization demo:")
    print(f"  {out_dir}/mnist_personalize_clean.csv  -- 500 un-augmented samples (clean baseline)")
    print(f"  {out_dir}/mnist_personalize_test.csv   -- 500 augmented samples   (domain-shifted eval)")
    print(f"  {out_dir}/mnist_personalize_train.csv  -- 200 augmented samples   (fine-tune set)")
    print(f"  {out_dir}/personalize_preview.png      -- clean vs augmented side-by-side")


if __name__ == "__main__":
    main()
