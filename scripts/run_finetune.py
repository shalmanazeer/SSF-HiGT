"""
Phase II: Supervised federated fine-tuning (Algorithm 1, lines 18-27).
Unfreezes the classification head and jointly optimizes
L_CE + lambda1*L_Mask + lambda2*L_Contrast.

Usage:
    python scripts/run_finetune.py --config configs/default.yaml \
        --num_clients 10 --rounds 50 --dataset data/cicids2017/clean.csv \
        --init_checkpoint checkpoints/pretrain_final.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import flwr as fl
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FlowDatasetConfig, FlowSequenceDataset, dirichlet_partition, load_and_preprocess
from src.data.graph_builder import build_knn_graph
from src.federated.client import SSFHiGTClient
from src.federated.server import AdaptiveCosineAggregation
from src.models.ssf_higt import SSFHiGT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--dirichlet_alpha", type=float, default=0.5)
    p.add_argument("--init_checkpoint", type=str, default=None)
    p.add_argument("--checkpoint_out", type=str, default="checkpoints/finetune_final.pt")
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

    def client_fn(cid: str) -> SSFHiGTClient:
        idx = int(cid)
        X_c, y_c = client_splits[idx]
        seq_len = cfg["model"]["transformer"]["max_seq_len"] // 4 or 8
        seq_ds = FlowSequenceDataset(X_c, y_c, seq_len=seq_len)
        loader = DataLoader(seq_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True)

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
            max_seq_len=seq_len,
        )

        if args.init_checkpoint and Path(args.init_checkpoint).exists():
            state = torch.load(args.init_checkpoint, map_location="cpu")
            model.load_state_dict(state, strict=False)  # SSL checkpoint may lack classifier weights

        return SSFHiGTClient(
            model=model,
            train_loader=loader,
            graph_batch=(node_features, adj, edge_attr),
            device=device,
            loss_weights=cfg["loss_weights"] | {"nt_xent_temperature": cfg["self_supervised"]["nt_xent_temperature"]},
            mode="finetune",
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
    print("Phase II fine-tuning complete.")


if __name__ == "__main__":
    main()
