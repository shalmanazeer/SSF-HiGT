"""
Phase I: Self-supervised federated pre-training (Algorithm 1, lines 2-16).

Usage:
    python scripts/run_pretrain.py --config configs/default.yaml \
        --num_clients 10 --rounds 50 --dataset data/cicids2017/clean.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import flwr as fl
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FlowDatasetConfig, FlowSequenceDataset, dirichlet_partition, load_and_preprocess
from src.data.graph_builder import build_knn_graph
from src.federated.client import SSFHiGTClient, get_model_params
from src.federated.server import AdaptiveCosineAggregation
from src.models.ssf_higt import SSFHiGT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dataset", type=str, required=True, help="Path to cleaned flow CSV")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--dirichlet_alpha", type=float, default=0.5)
    p.add_argument("--checkpoint_out", type=str, default="checkpoints/pretrain_final.pt")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_and_preprocess(FlowDatasetConfig(csv_path=args.dataset))
    X_train, y_train = data["train"]
    num_features, num_classes = data["num_features"], data["num_classes"]

    client_splits = dirichlet_partition(X_train, y_train, args.num_clients, alpha=args.dirichlet_alpha)

    cfg["model"]["num_features"] = num_features
    cfg["model"]["num_classes"] = num_classes

    def client_fn(cid: str) -> SSFHiGTClient:
        idx = int(cid)
        X_c, y_c = client_splits[idx]
        seq_ds = FlowSequenceDataset(X_c, y_c, seq_len=cfg["model"]["transformer"]["max_seq_len"] // 4 or 8)
        loader = DataLoader(seq_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True)

        # local communication graph: treat each flow-sequence window as a "device" node
        device_feats = X_c[: min(len(X_c), 200)]
        node_features, adj, edge_attr = build_knn_graph(device_feats, k=cfg["graph"]["knn_k"])

        model = SSFHiGT(
            num_features=num_features,
            num_classes=num_classes,
            d_model=cfg["model"]["d_model"],
            num_heads=cfg["model"]["transformer"]["num_heads"],
            num_transformer_layers=cfg["model"]["transformer"]["num_layers"],
            num_gat_layers=cfg["model"]["gat"]["num_layers"],
            gat_heads=cfg["model"]["gat"]["num_heads"],
            edge_dim=node_features.shape[-1],
            dropout=cfg["model"]["transformer"]["dropout"],
            max_seq_len=seq_ds.seq_len,
        )

        return SSFHiGTClient(
            model=model,
            train_loader=loader,
            graph_batch=(node_features, adj, edge_attr),
            device=device,
            loss_weights=cfg["loss_weights"] | {"nt_xent_temperature": cfg["self_supervised"]["nt_xent_temperature"]},
            mode="pretrain",
            local_epochs=cfg["training"]["local_epochs"],
            lr=cfg["training"]["learning_rate"],
            masking_ratio=cfg["self_supervised"]["masking_ratio"],
        )

    strategy = AdaptiveCosineAggregation(
        alpha=cfg["federated_aggregation"]["alpha"],
        beta=cfg["federated_aggregation"]["beta"],
        eta=1.0,
        fraction_fit=1.0,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
    print(f"Phase I pre-training complete. Save/export global params via strategy history as needed.")


if __name__ == "__main__":
    main()
