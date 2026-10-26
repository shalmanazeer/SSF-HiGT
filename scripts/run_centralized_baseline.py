"""
Centralized (non-federated) training of SSF-HiGT on pooled data, matching
the "w/o Federated (centralised)" row of the ablation study (Table 4).

Usage:
    python scripts/run_centralized_baseline.py --config configs/default.yaml \
        --dataset data/cicids2017/clean.csv --epochs 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FlowDatasetConfig, FlowSequenceDataset, load_and_preprocess
from src.data.graph_builder import build_knn_graph
from src.losses import total_loss
from src.models.ssf_higt import SSFHiGT
from src.utils.masking import mask_features


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--checkpoint_out", type=str, default="checkpoints/centralized_final.pt")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_and_preprocess(FlowDatasetConfig(csv_path=args.dataset))
    X_train, y_train = data["train"]
    num_features, num_classes = data["num_features"], data["num_classes"]

    seq_len = cfg["model"]["transformer"]["max_seq_len"] // 4 or 8
    train_ds = FlowSequenceDataset(X_train, y_train, seq_len=seq_len)
    loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True)

    device_feats = X_train[: min(len(X_train), 500)]
    node_features, adj, edge_attr = build_knn_graph(device_feats, k=cfg["graph"]["knn_k"])
    node_features, adj, edge_attr = node_features.to(device), adj.to(device), edge_attr.to(device)

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
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_weights = cfg["loss_weights"] | {"nt_xent_temperature": cfg["self_supervised"]["nt_xent_temperature"]}

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for x_seq, y in loader:
            x_seq, y = x_seq.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x_seq, node_features, adj, edge_attr)
            masked_x, mask = mask_features(x_seq, ratio=cfg["self_supervised"]["masking_ratio"])
            recon = model.forward_pretrain(masked_x)
            z_pos = torch.nn.functional.dropout(out["z_t"], p=0.1, training=True)

            loss, components = total_loss(
                logits=out["logits"],
                targets=y,
                reconstructed=recon,
                original=x_seq,
                recon_mask=mask,
                z_anchor=out["z_t"],
                z_positive=z_pos,
                node_embeddings=out["z_g_nodes"],
                adj=adj,
                model=model,
                weights=loss_weights,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {epoch_loss / len(loader):.4f}")

    Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint_out)
    print(f"Saved centralized baseline checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
