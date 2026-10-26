"""
Flower Strategy implementing the adaptive cosine-similarity aggregation
of Eq. (19)-(20), as a drop-in replacement for flwr.server.strategy.FedAvg.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from src.federated.aggregation import (
    average_direction,
    compute_adaptive_weights,
    compute_deltas,
)


class AdaptiveCosineAggregation(fl.server.strategy.FedAvg):
    """Custom strategy: on each round, computes per-client weight updates
    (deltas), estimates the global update direction, derives adaptive
    weights omega_i via cosine similarity + dataset-size scaling, and
    aggregates accordingly (Eq. 19-20) instead of plain weighted averaging.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 5.0, eta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self._prev_global_params: Optional[List[np.ndarray]] = None
        self._prev_direction: Optional[List[np.ndarray]] = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], dict]:
        if not results:
            return None, {}

        client_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        client_sizes = [fit_res.num_examples for _, fit_res in results]

        if self._prev_global_params is None:
            # first round: nothing to compute deltas against yet, fall back to FedAvg once
            self._prev_global_params = client_params[0]

        client_deltas = [compute_deltas(cp, self._prev_global_params) for cp in client_params]

        direction = self._prev_direction if self._prev_direction is not None else average_direction(client_deltas)

        weights = compute_adaptive_weights(
            client_deltas, direction, client_sizes, alpha=self.alpha, beta=self.beta
        )

        new_global = [p.copy() for p in self._prev_global_params]
        for w, delta in zip(weights, client_deltas):
            for layer_idx, d in enumerate(delta):
                new_global[layer_idx] = new_global[layer_idx] + self.eta * w * d

        self._prev_direction = average_direction(client_deltas)
        self._prev_global_params = new_global

        metrics = {"agg_weights_std": float(np.std(weights))}
        return ndarrays_to_parameters(new_global), metrics
