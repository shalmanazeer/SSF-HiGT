"""
Adaptive cosine-similarity federated aggregation, replacing FedAvg.

Implements Eq. (19)-(20):

    omega_i = [n_i^alpha * exp(beta * cos(dTheta_i, dTheta_g))] /
              sum_j [n_j^alpha * exp(beta * cos(dTheta_j, dTheta_g))]

    Theta_g^(t+1) = Theta_g^(t) + eta * sum_i omega_i * dTheta_i^(t)
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def _flatten(params: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([p.ravel() for p in params])


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a_flat, b_flat = _flatten(a) if isinstance(a, list) else a, _flatten(b) if isinstance(b, list) else b
    num = float(np.dot(a_flat, b_flat))
    denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + eps
    return num / denom


def compute_adaptive_weights(
    client_deltas: List[List[np.ndarray]],
    global_delta_direction: List[np.ndarray],
    client_sizes: List[int],
    alpha: float = 0.5,
    beta: float = 5.0,
) -> np.ndarray:
    """Computes omega_i for each client per Eq. (19).

    Args:
        client_deltas: list of N clients, each a list of per-layer weight-delta arrays (dTheta_i)
        global_delta_direction: reference direction dTheta_g (e.g. previous round's
            aggregated delta, or the mean of current-round deltas on the first round)
        client_sizes: n_i, number of local training samples per client
        alpha: dataset-size weighting exponent
        beta: cosine-similarity weighting sharpness
    Returns:
        (N,) array of normalized aggregation weights omega_i, sums to 1.
    """
    global_flat = _flatten(global_delta_direction)
    scores = []
    for delta, n_i in zip(client_deltas, client_sizes):
        delta_flat = _flatten(delta)
        sim = cosine_sim(delta_flat, global_flat)
        score = (max(n_i, 1) ** alpha) * np.exp(beta * sim)
        scores.append(score)

    scores = np.array(scores, dtype=np.float64)
    total = scores.sum()
    if total <= 0 or not np.isfinite(total):
        # fall back to uniform weighting if degenerate
        return np.ones(len(scores)) / len(scores)
    return scores / total


def aggregate(
    global_params: List[np.ndarray],
    client_deltas: List[List[np.ndarray]],
    weights: np.ndarray,
    eta: float = 1.0,
) -> List[np.ndarray]:
    """Applies Eq. (20): Theta_g^(t+1) = Theta_g^(t) + eta * sum_i omega_i * dTheta_i."""
    new_params = [p.copy() for p in global_params]
    for w, delta in zip(weights, client_deltas):
        for layer_idx, d in enumerate(delta):
            new_params[layer_idx] = new_params[layer_idx] + eta * w * d
    return new_params


def compute_deltas(local_params: List[np.ndarray], global_params: List[np.ndarray]) -> List[np.ndarray]:
    """dTheta_i = Theta_i - Theta_g."""
    return [lp - gp for lp, gp in zip(local_params, global_params)]


def average_direction(client_deltas: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Fallback global-direction estimate: simple mean of client deltas,
    used to seed dTheta_g before any aggregation history exists."""
    num_layers = len(client_deltas[0])
    return [
        np.mean([delta[layer_idx] for delta in client_deltas], axis=0) for layer_idx in range(num_layers)
    ]
