"""
Dataset loaders for CICIDS-2017 [25] and CIC-IoT-2023 [26].

Both datasets are publicly available through the Canadian Institute for
Cybersecurity (CIC). This module expects a pre-cleaned CSV (invalid/NaN/inf
rows already dropped, as described in the paper's Section 4.2) with a
`Label` column plus numeric flow-feature columns, and partitions clients
using a Dirichlet distribution to emulate the non-IID setting (Eq. 3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset


@dataclass
class FlowDatasetConfig:
    csv_path: str
    label_column: str = "Label"
    seq_len: int = 8          # flows grouped into short sequences for the Transformer stream
    test_size: float = 0.15
    val_size: float = 0.15


class FlowSequenceDataset(Dataset):
    """Wraps flow-feature rows into fixed-length sequences of shape (T, F)
    for the Transformer stream, matching x_j in Eq. (1)."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int = 8):
        self.seq_len = seq_len
        n_complete = len(features) // seq_len
        features = features[: n_complete * seq_len]
        labels = labels[: n_complete * seq_len]
        self.sequences = features.reshape(n_complete, seq_len, -1)
        # sequence label = majority label within the window
        self.seq_labels = np.array(
            [np.bincount(labels[i * seq_len : (i + 1) * seq_len]).argmax() for i in range(n_complete)]
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.seq_labels[idx], dtype=torch.long),
        )


def load_and_preprocess(config: FlowDatasetConfig):
    """Loads a cleaned CSV, encodes labels, standardizes features, and
    returns train/val/test splits as numpy arrays."""
    df = pd.read_csv(config.csv_path)
    df = df.dropna()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[config.label_column].values)
    X = df.drop(columns=[config.label_column]).select_dtypes(include=[np.number]).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n = len(X)
    idx = np.random.permutation(n)
    n_test = int(n * config.test_size)
    n_val = int(n * config.val_size)

    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    return {
        "train": (X[train_idx], y[train_idx]),
        "val": (X[val_idx], y[val_idx]),
        "test": (X[test_idx], y[test_idx]),
        "num_classes": len(label_encoder.classes_),
        "num_features": X.shape[1],
        "class_names": list(label_encoder.classes_),
    }


def dirichlet_partition(X: np.ndarray, y: np.ndarray, num_clients: int, alpha: float = 0.5, seed: int = 42):
    """Partitions (X, y) across `num_clients` using a Dirichlet(alpha)
    distribution per class, producing the non-IID setting of Eq. (3).
    Lower alpha => more heterogeneous (non-IID) client distributions."""
    rng = np.random.default_rng(seed)
    num_classes = y.max() + 1
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(y == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        splits = np.split(class_idx, split_points)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    return [(X[idx], y[idx]) for idx in client_indices]
