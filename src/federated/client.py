"""
Flower NumPyClient wrapper for a single SSF-HiGT client. Supports both
Phase I (self-supervised pre-training) and Phase II (supervised fine-tuning)
local-training modes, matching Algorithm 1 in the paper.
"""
from __future__ import annotations

from typing import Dict, List

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.losses import total_loss
from src.utils.masking import mask_features
from src.models.ssf_higt import SSFHiGT


def get_model_params(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_params(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(state_dict, strict=True)


class SSFHiGTClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: SSFHiGT,
        train_loader: DataLoader,
        graph_batch,  # (node_features, adj, edge_attr) tuple for this client's local graph
        device: torch.device,
        loss_weights: Dict,
        mode: str = "pretrain",  # "pretrain" | "finetune"
        local_epochs: int = 5,
        lr: float = 1e-4,
        masking_ratio: float = 0.30,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.node_features, self.adj, self.edge_attr = [t.to(device) for t in graph_batch]
        self.device = device
        self.loss_weights = loss_weights
        self.mode = mode
        self.local_epochs = local_epochs
        self.lr = lr
        self.masking_ratio = masking_ratio
        self.num_samples = len(train_loader.dataset)

    def get_parameters(self, config):
        return get_model_params(self.model)

    def set_parameters(self, params):
        set_model_params(self.model, params)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.local_epochs):
            for x_seq, y in self.train_loader:
                x_seq, y = x_seq.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                if self.mode == "pretrain":
                    masked_x, mask = mask_features(x_seq, ratio=self.masking_ratio)
                    recon = self.model.forward_pretrain(masked_x)

                    z_t, z_g, z_g_nodes = self.model.encode_streams(
                        x_seq, self.node_features, self.adj, self.edge_attr
                    )
                    # simple augmentation for the positive contrastive view: light dropout
                    z_pos = torch.nn.functional.dropout(z_t, p=0.1, training=True)

                    loss, _ = total_loss(
                        logits=None,
                        targets=None,
                        reconstructed=recon,
                        original=x_seq,
                        recon_mask=mask,
                        z_anchor=z_t,
                        z_positive=z_pos,
                        node_embeddings=z_g_nodes,
                        adj=self.adj,
                        weights=self.loss_weights,
                    )
                else:  # finetune
                    out = self.model(x_seq, self.node_features, self.adj, self.edge_attr)
                    masked_x, mask = mask_features(x_seq, ratio=self.masking_ratio)
                    recon = self.model.forward_pretrain(masked_x)
                    z_pos = torch.nn.functional.dropout(out["z_t"], p=0.1, training=True)

                    loss, _ = total_loss(
                        logits=out["logits"],
                        targets=y,
                        reconstructed=recon,
                        original=x_seq,
                        recon_mask=mask,
                        z_anchor=out["z_t"],
                        z_positive=z_pos,
                        node_embeddings=out["z_g_nodes"],
                        adj=self.adj,
                        model=self.model,
                        weights=self.loss_weights,
                    )

                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), self.num_samples, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x_seq, y in self.train_loader:
                x_seq, y = x_seq.to(self.device), y.to(self.device)
                out = self.model(x_seq, self.node_features, self.adj, self.edge_attr)
                loss = torch.nn.functional.cross_entropy(out["logits"], y)
                loss_sum += loss.item() * x_seq.size(0)
                correct += (out["logits"].argmax(dim=-1) == y).sum().item()
                total += x_seq.size(0)
        accuracy = correct / max(total, 1)
        return loss_sum / max(total, 1), total, {"accuracy": accuracy}
