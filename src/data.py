"""Data acquisition, preprocessing, and stratified splitting for Fashion-MNIST.

Ensures official 10,000-sample test set remains strictly isolated
and provides reproducible, ultra-fast caching and subsampling for benchmark execution.
"""

from typing import Tuple
import gzip
import os
import urllib.request
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

CLASS_NAMES = [
    "T-shirt/top",  # Class 0
    "Trouser",      # Class 1
    "Pullover",     # Class 2
    "Dress",        # Class 3
    "Coat",         # Class 4
    "Sandal",       # Class 5
    "Shirt",        # Class 6
    "Sneaker",      # Class 7
    "Bag",          # Class 8
    "Ankle boot",   # Class 9
]

# Official high-speed mirrors for Fashion-MNIST
MIRRORS = [
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
]

FILES = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_lbl": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_lbl": "t10k-labels-idx1-ubyte.gz",
}


def _download_and_parse_idx(filename: str, cache_dir: str = "data/fashion_mnist") -> np.ndarray:
    """Download an IDX file from reliable mirrors and cache locally."""
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, filename)

    if not os.path.exists(local_path):
        downloaded = False
        for base_url in MIRRORS:
            url = base_url + filename
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    with open(local_path, "wb") as f:
                        f.write(resp.read())
                downloaded = True
                break
            except Exception:
                continue
        if not downloaded:
            raise RuntimeError(f"Failed to download {filename} from all mirrors.")

    with gzip.open(local_path, "rb") as gz:
        data = gz.read()

    # Parse IDX header
    magic = int.from_bytes(data[:4], "big")
    if magic == 2051:  # Image file (idx3)
        num_images = int.from_bytes(data[4:8], "big")
        rows = int.from_bytes(data[8:12], "big")
        cols = int.from_bytes(data[12:16], "big")
        return np.frombuffer(data[16:], dtype=np.uint8).reshape(num_images, rows * cols).astype(np.float32)
    elif magic == 2049:  # Label file (idx1)
        num_labels = int.from_bytes(data[4:8], "big")
        return np.frombuffer(data[8:], dtype=np.uint8).astype(np.int64)
    else:
        raise ValueError(f"Unknown IDX magic number {magic} in {filename}")


def load_raw_data(cache_dir: str = "data/fashion_mnist") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Fashion-MNIST from local cache or official mirrors into numpy arrays.

    Returns:
        X_train_full (60000, 784): Full training pixel array in float32.
        y_train_full (60000,): Training integer labels (0-9).
        X_test (10000, 784): Official holdout test pixel array in float32.
        y_test (10000,): Official holdout integer labels (0-9).
    """
    X_train_full = _download_and_parse_idx(FILES["train_img"], cache_dir)
    y_train_full = _download_and_parse_idx(FILES["train_lbl"], cache_dir)
    X_test = _download_and_parse_idx(FILES["test_img"], cache_dir)
    y_test = _download_and_parse_idx(FILES["test_lbl"], cache_dir)

    return X_train_full, y_train_full, X_test, y_test


def get_stratified_split(
    quick: bool = True,
    quick_samples_per_class: int = 1200,
    random_state: int = 42,
    cache_dir: str = "data/fashion_mnist",
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Retrieve training and holdout test partitions.

    In quick mode, extracts a stratified subset of 12,000 samples (1,200 per class)
    from the 60,000 training set to ensure full 5-fold cross-validation completes
    in <90 seconds on standard CPU hardware. The official 10,000 test set is preserved untouched.

    Args:
        quick: If True, returns 12,000 stratified train samples. If False, returns all 60,000.
        quick_samples_per_class: Number of samples per class when quick=True (default 1200).
        random_state: Random seed for deterministic reproducibility.
        cache_dir: Directory to cache downloaded raw dataset files.

    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    X_train_full, y_train_full, X_test, y_test = load_raw_data(cache_dir=cache_dir)

    if not quick:
        return (X_train_full, y_train_full), (X_test, y_test)

    total_quick = quick_samples_per_class * len(CLASS_NAMES)
    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=total_quick,
        random_state=random_state,
    )
    train_idx, _ = next(sss.split(X_train_full, y_train_full))

    train_idx = np.sort(train_idx)
    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]

    return (X_train, y_train), (X_test, y_test)
