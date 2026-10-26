"""
Communication graph construction: builds G_i = (V_i, E_i, X_i^v, X_i^e)
(Eq. 2) from a batch of flows using k-NN over IP/device feature embeddings,
with k defined by the `knn_k` hyperparameter (Table 3, default 7).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(
    device_features: np.ndarray,
    edge_features: np.ndarray | None = None,
    k: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        device_features: (N, d_v) per-device/node feature matrix
        edge_features: optional (N, N, d_e) precomputed pairwise edge
            attributes (e.g. protocol one-hot, byte-rate, port similarity).
            If None, a simple feature-difference edge attribute is used.
        k: number of nearest neighbours per node
    Returns:
        node_features: (N, d_v) tensor
        adj: (N, N) binary adjacency tensor
        edge_attr: (N, N, d_e) tensor
    """
    N = device_features.shape[0]
    k = min(k, max(N - 1, 1))

    nn_model = NearestNeighbors(n_neighbors=k + 1).fit(device_features)
    _, indices = nn_model.kneighbors(device_features)

    adj = np.zeros((N, N), dtype=np.float32)
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:
                adj[i, j] = 1.0
                adj[j, i] = 1.0  # symmetrize

    if edge_features is None:
        d_v = device_features.shape[1]
        edge_attr = np.zeros((N, N, d_v), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if adj[i, j] > 0:
                    edge_attr[i, j] = np.abs(device_features[i] - device_features[j])
    else:
        edge_attr = edge_features

    return (
        torch.tensor(device_features, dtype=torch.float32),
        torch.tensor(adj, dtype=torch.float32),
        torch.tensor(edge_attr, dtype=torch.float32),
    )
